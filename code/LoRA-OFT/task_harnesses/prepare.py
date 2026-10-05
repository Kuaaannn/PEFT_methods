"""Freeze source datasets and evaluation banks on a CUDA machine."""
import argparse
import gzip
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from .coding.protocol import validate_benchmarks
from .shared.data import group_key, split_groups
from .shared.io import ROOT, REPO, atomic_json, digest, exclusive, file_hash, read_json, read_rows, write_rows, writable_output
from .shared.runtime import require_cuda

EVALPLUS_COMMIT = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"


def retention_rows(path):
    """Read the existing math bank without changing its order or text."""
    if Path(path).suffix == ".jsonl":
        rows = read_rows(path)
    else:
        value = read_json(path)
        if isinstance(value, dict) and "rows" in value:
            import hashlib
            texts = value["rows"]
            expected = hashlib.sha256("\u0000".join(texts).encode()).hexdigest()
            if value.get("n") != 200 or len(texts) != 200 or value.get("sha256") != expected:
                raise ValueError("Original math retention bank failed its count/content checksum")
            rows = [{"id": str(i), "text": text} for i, text in enumerate(texts)]
        else:
            rows = value
    if not isinstance(rows, list):
        raise ValueError("Invalid retention bank")
    return rows






def coding_sources(revision, human_version, mbpp_version):
    if len(revision) != 40 or revision == "main":
        raise ValueError("Pin the PiSSA dataset revision")
    from .download_data import PISSA_REVISION, RELEASES
    raw = ROOT / "data/sources"
    downloads = read_json(raw / "downloads.json")
    if revision != PISSA_REVISION or downloads["pissa_revision"] != revision:
        raise ValueError("Downloaded PiSSA source revision mismatch")
    if downloads["releases"] != RELEASES or (human_version, mbpp_version) != tuple(RELEASES.values()):
        raise ValueError("Downloaded EvalPlus versions mismatch")
    for name, sha in downloads["files"].items():
        if file_hash(raw / name) != sha:
            raise ValueError(f"Downloaded data changed: {name}")
    from evalplus.data.utils import CACHE_DIR
    cache = Path(CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    for name, version in RELEASES.items():
        target = cache / f"{name}-{version}.jsonl"
        # Preparation owns this task-specific cache; never reuse an unverified file.
        temporary = target.with_suffix(".partial")
        with gzip.open(raw / f"{name}-{version}.jsonl.gz", "rb") as source, temporary.open("wb") as out:
            shutil.copyfileobj(source, out)
        temporary.replace(target)
    from evalplus.data import (get_human_eval_plus, get_human_eval_plus_hash,
                               get_mbpp_plus, get_mbpp_plus_hash)
    training = read_rows(raw / "pissa/python/train.json")
    rows = [{**dict(row), "id": f"python/{index}"} for index, row in enumerate(training)]
    tests, information = [], {}
    for name, version, getter, hasher in (
        ("humaneval", human_version, get_human_eval_plus, get_human_eval_plus_hash),
        ("mbpp", mbpp_version, get_mbpp_plus, get_mbpp_plus_hash)):
        if version == "default" or not version.startswith("v"):
            raise ValueError("Pin explicit EvalPlus dataset releases")
        data = getter(version=version)
        information[name] = {"version": version, "hash": hasher(version=version), "n": len(data)}
        for task_id, row in data.items():
            tests.append({"id": f"{name}/{task_id}", "benchmark": name, "task_id": task_id,
                          "prompt": row["prompt"], "entry_point": row["entry_point"]})
    validate_benchmarks(tests)
    return rows, tests, {"repository": "fxmeng/pissa-dataset", "revision": revision,
                         "data_dir": "python", "split": "train", "rows": len(rows)}, information


def ensure_bank(task, path):
    """First smoke prepares the canonical bank once; later runs only read it."""
    import sys
    from .shared.data import load_bank
    require_cuda()
    path = Path(path).resolve()
    canonical = ROOT / "data" / task / "manifest.json"
    if path != canonical:
        # Custom/frozen banks must be prepared explicitly; no guessed paths.
        load_bank(path, task)
        return
    with exclusive(path.parent, lock_name="prepare.lock", blocking=True):
        if not path.exists():
            argv = [sys.executable, "-m", "task_harnesses.prepare", "--task", task,
                    "--output", str(path.parent), "--retention",
                    str(REPO / "experiments/results/retention_bank.json")]
            subprocess.run(argv, cwd=REPO, check=True)
        load_bank(path, task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("coding",), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retention", required=True, help="Existing frozen 200-document JSON/JSONL bank")
    parser.add_argument("--pissa-revision", default="d4746ceca8314940af8a61333bc2d395d9e259c9")
    parser.add_argument("--humaneval-version", default="v0.1.10")
    parser.add_argument("--mbpp-version", default="v0.2.0")
    args = parser.parse_args()
    require_cuda()
    output = writable_output(args.output)
    with exclusive(output):
        if (output / "manifest.json").exists():
            raise FileExistsError("Bank already frozen; use a new output for a protocol revision")
        if not args.pissa_revision:
            parser.error("--pissa-revision is required")
        training, tests, source, information = coding_sources(
            args.pissa_revision, args.humaneval_version, args.mbpp_version)
        train, dev, audit = split_groups(training, tests, response_policy="generative")
        print("DATA_AUDIT", {k: v for k, v in audit.items()
                             if k not in ("conflicting_groups", "alternative_response_choices")}, flush=True)
        documents = retention_rows(args.retention)
        if len(documents) != 200:
            raise ValueError("Supply exactly the existing 200 retention documents; no fresh sampling")
        documents = [{"id": str(row.get("id", i)), "text": row["text"]} for i, row in enumerate(documents)]
        if len({row["id"] for row in documents}) != 200 or any(not row["text"].strip() for row in documents):
            raise ValueError("Invalid retention IDs/text")
        files = {}
        for name, rows in (("train", train), ("dev", dev), ("test", tests), ("retention", documents)):
            path = output / f"{name}.jsonl"
            write_rows(path, rows)
            files[name] = {"path": path.name, "sha256": file_hash(path), "n": len(rows)}
        atomic_json(output / "manifest.json", {"schema": 1, "task": args.task, "files": files,
            "source": source, "split_seed": 2027, "audit": audit,
            "retention_source_sha256": file_hash(args.retention), "evalplus": information,
            "profile": "budget20k-v2-audited-source-duplicates-not-full-paper-reproduction"})


if __name__ == "__main__":
    main()

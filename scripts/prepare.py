"""Download public inputs and prepare the unchanged experiment data pipelines."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from common import CONFIGS, FLUX, HARNESS, LLM, MODELS, ROOT, execute, python, read, shared_options, write

EVALPLUS_COMMIT = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"


def image_hash(path):
    h = hashlib.sha256()
    for p in sorted(q for q in path.rglob("*") if q.is_file() and q.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        name = p.relative_to(path).as_posix().encode()
        h.update(len(name).to_bytes(8, "big")); h.update(name)
        h.update(p.stat().st_size.to_bytes(8, "big"))
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                h.update(block)
    return h.hexdigest()


def download_objects():
    from huggingface_hub import snapshot_download
    design = read(CONFIGS / "objects.json")
    source = Path(snapshot_download(design["dataset_repository"], repo_type="dataset", revision=design["dataset_revision"]))
    for subject in design["subjects"]:
        destination = FLUX / subject["training_config"]["dataset_id"]
        original = source / "dataset" / subject["id"]
        if not original.is_dir():
            original = source / subject["id"]
        if not original.is_dir():
            raise FileNotFoundError(f"The pinned dataset is missing subject {subject['id']}")
        if not destination.exists():
            shutil.copytree(original, destination)
        if image_hash(destination) != subject["dataset_sha"]:
            raise RuntimeError(f"Image bytes/names differ from the paper for {subject['id']}")


def download(a):
    from huggingface_hub import snapshot_download
    root = a.output.resolve()
    path = root / "data/assets.json"
    assets = read(path) if path.exists() else {}
    models = [a.model] if a.model else ["qwen", "llama"] if a.task in ("math", "coding") else ["flux"]
    for model in models:
        repo, revision = MODELS[model]
        snapshot = str(Path(snapshot_download(repo, revision=revision)).resolve())
        if model in assets and assets[model] != snapshot:
            raise RuntimeError("Prepared assets use a different cache location. Use a new --output directory.")
        assets[model] = snapshot
    write(path, assets)
    bank = read(ROOT / "data/retention_bank.json")
    assert bank["n"] == len(bank["rows"]) == 200
    assert hashlib.sha256("\0".join(bank["rows"]).encode()).hexdigest() == bank["sha256"]
    write(LLM / "results/retention_bank.json", bank, immutable=True)
    if a.task == "objects":
        download_objects()
    if a.task in ("cat", "objects"):
        config = read(FLUX / "default_training_params.json")
        config["model_id"] = assets["flux"]
        write(root / "data/flux-cache-config.json", config, immutable=True)
    if a.task == "coding":
        vendor = HARNESS / "task_harnesses/vendor/evalplus"
        if not vendor.exists():
            subprocess.run(["git", "clone", "https://github.com/evalplus/evalplus.git", str(vendor)], check=True)
            subprocess.run(["git", "-C", str(vendor), "checkout", "--detach", EVALPLUS_COMMIT], check=True)
        head = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
        if head != EVALPLUS_COMMIT:
            raise RuntimeError("Existing EvalPlus checkout is not the pinned revision")
        execute([sys.executable, "-m", "pip", "install", "--no-deps", str(vendor)])
        execute([sys.executable, "-m", "task_harnesses.download_data"], cwd=HARNESS)
    print(f"Prepared pinned model snapshots and inputs for {a.task}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("math", "coding", "cat", "objects"))
    p.add_argument("--stage", choices=("download", "data", "sandbox"), default="download")
    p.add_argument("--model", choices=("qwen", "llama", "flux"))
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    shared_options(p)
    a = p.parse_args()
    env = "train" if a.task in ("math", "coding") else "flux"
    if a.model and ((a.model == "flux") != (env == "flux")):
        p.error("Choose a language model for math/coding or flux for images")
    root = a.output.resolve()
    if a.stage == "download":
        if a.worker:
            download(a)
        else:
            command = [python(env), Path(__file__).resolve(), a.task, "--worker", "--output", root]
            if a.model:
                command += ["--model", a.model]
            execute(command, env_name=env, dry_run=a.dry_run)
        return
    if a.stage == "sandbox":
        if a.task != "coding":
            p.error("The execution sandbox is only needed for coding")
        execute([python("train"), "-m", "task_harnesses.prepare_sandbox", "--source",
                 HARNESS / "task_harnesses/vendor/evalplus", "--output", root / "sandbox"],
                cwd=HARNESS, dry_run=a.dry_run)
    elif a.task == "coding":
        execute([python("train"), "-m", "task_harnesses.prepare", "--task", "coding", "--retention",
                 ROOT / "data/retention_bank.json", "--output", root / "data/coding"], cwd=HARNESS, dry_run=a.dry_run)
    elif a.task in ("cat", "objects"):
        execute([python("flux"), "build_forgetting_caches.py", "--config", root / "data/flux-cache-config.json",
                 "--seeds", "0,1,2", "--num-val-samples", "200"], cwd=FLUX, env_name="flux", dry_run=a.dry_run)
    else:
        assets = read(root / "data/assets.json") if (root / "data/assets.json").exists() else {}
        for model in ([a.model] if a.model else ["qwen", "llama"]):
            suffix = "qwen25_7b" if model == "qwen" else "llama31_8b"
            execute([python("train"), "-m", "scripts.build_base_reference", "--model", assets.get(model, f"<prepared-{model}-snapshot>"),
                     "--batch-size", 2, "--max-length", 768, "--out", LLM / f"results/base_reference_{suffix}.json"],
                    cwd=LLM, dry_run=a.dry_run)


if __name__ == "__main__":
    main()

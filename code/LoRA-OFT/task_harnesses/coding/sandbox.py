"""Fail-closed EvalPlus execution. No generated program executes on the host."""
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from ..shared.io import digest, file_hash, read_json, write_rows
from ..shared.runtime import require_cuda


def runtime():
    found = shutil.which("apptainer") or shutil.which("singularity")
    if not found:
        raise RuntimeError("Singularity/Apptainer is required for isolated code evaluation")
    return found


def bundle_hash(root):
    root = Path(root)
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()})


@lru_cache(maxsize=4)
def validate_sandbox(path):
    spec = read_json(path)
    if spec.get("schema") != 2 or not {"image", "sha256", "evalplus_commit", "bundle", "bundle_sha256", "isolation_verified"} <= set(spec):
        raise ValueError("Sandbox requires the provisioned schema-2 image/bundle manifest")
    if spec["evalplus_commit"] != "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e":
        raise ValueError("Sandbox EvalPlus revision differs from the frozen protocol")
    image = Path(spec["image"])
    if not image.is_absolute() or not image.is_file() or file_hash(image) != spec["sha256"]:
        raise ValueError("Missing or changed sandbox image")
    bundle = Path(spec["bundle"])
    if not bundle.is_absolute() or not bundle.is_dir() or bundle_hash(bundle) != spec["bundle_sha256"]:
        raise ValueError("Missing or changed read-only evaluator/data bundle")
    if spec["isolation_verified"] is not True:
        raise ValueError("Container isolation was not verified")
    runtime()
    return spec


def container_command(spec, work, command, *, network=False, writable_bundle=False):
    """Only the owned work directory is writable during evaluation."""
    work = Path(work).resolve()
    # --containall otherwise puts /tmp in Singularity's small session filesystem.
    # Reuse the same owned directory, not the host's /tmp or any broader bind.
    argv = [runtime(), "exec", "--containall", "--cleanenv", "--no-home",
            "--no-mount", "hostfs,tmp", "--bind", f"{work}:/work:rw",
            "--bind", f"{work}:/tmp:rw", "--bind", f"{work}:/var/tmp:rw",
            "--bind", f"{spec['bundle']}:/opt/evalplus:{'rw' if writable_bundle else 'ro'}",
            "--pwd", "/work", "--env",
            "PYTHONPATH=/opt/evalplus/python,PYTHONDONTWRITEBYTECODE=1,"
            "XDG_CACHE_HOME=/opt/evalplus/cache,TMPDIR=/tmp,TMP=/tmp,TEMP=/tmp"]
    if not network:
        argv += ["--net", "--network", "none"]
    return argv + [spec["image"], *command]


def clean_environment():
    # No credentials or inherited container bind/environment overrides.
    return {"PATH": os.environ["PATH"]}


def ensure_sandbox(path=None):
    """Prepare once inside the first coding smoke; never submit a setup job."""
    import sys
    from ..shared.io import ROOT, REPO, exclusive
    require_cuda()
    canonical = ROOT / "runtime/evalplus/sandbox.json"
    path = Path(path).resolve() if path else canonical
    if path != canonical:
        validate_sandbox(str(path))
        return str(path)
    with exclusive(path.parent, lock_name="prepare.lock", blocking=True):
        if not path.exists():
            subprocess.run([sys.executable, "-m", "task_harnesses.prepare_sandbox",
                            "--output", str(path.parent), "--source", str(ROOT / "vendor/evalplus")],
                           cwd=REPO, check=True)
        validate_sandbox(str(path))
    return str(path)


def summarize_results(results, expected_ids):
    table = results.get("eval", {})
    if not expected_ids or set(table) != set(expected_ids) or any(len(v) != 1 for v in table.values()):
        raise ValueError("EvalPlus must return exactly one completion for every frozen task")
    if any(not v[0].get("base_status") or not v[0].get("plus_status") for v in table.values()):
        raise ValueError("Missing base/plus test outcomes; incomplete execution is not zero accuracy")
    base = sum(v[0]["base_status"] == "pass" for v in table.values())
    plus = sum(v[0]["base_status"] == v[0]["plus_status"] == "pass" for v in table.values())
    return {"pass@1": base / len(table), "plus_pass@1": plus / len(table), "n": len(table)}


def score_samples(samples, benchmark, info, sandbox, output):
    require_cuda()
    spec = validate_sandbox(sandbox)
    if benchmark not in ("humaneval", "mbpp") or not re.fullmatch(r"v[0-9.]+", info["version"]):
        raise ValueError("Unpinned benchmark/version")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Only this temporary directory is writable/mounted, never home or checkpoints.
    with tempfile.TemporaryDirectory(prefix="evalplus-isolated-", dir=output) as temporary:
        work = Path(temporary)
        write_rows(work / "samples.jsonl", samples)
        argv = container_command(spec, work, ["python", "-m", "evalplus.evaluate",
                "--dataset", benchmark, "--samples", "/work/samples.jsonl",
                "--version", info["version"], "--parallel", "2", "--test-details",
                "--output-file", "/work/results.json"])
        with (output / f"{benchmark}.sandbox.log").open("w") as log:
            subprocess.run(argv, cwd=work, env=clean_environment(), check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        results = read_json(work / "results.json")
        if results.get("hash") != info["hash"]:
            raise ValueError("Sandbox benchmark hash differs from frozen generation bank")
        metrics = summarize_results(results, [s["task_id"] for s in samples])
        # Keep all test outcomes, not just aggregate pass@1.
        from ..shared.io import atomic_json
        atomic_json(output / f"{benchmark}.evalplus.json", results)
        return metrics

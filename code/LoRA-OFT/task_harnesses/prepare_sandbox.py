"""Provision pinned EvalPlus in a prebuilt Python SIF; no custom image build."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .coding.sandbox import (bundle_hash, clean_environment, container_command,
                             runtime, score_samples)
from .prepare import EVALPLUS_COMMIT
from .shared.io import ROOT, atomic_json, exclusive, file_hash, read_json, read_rows, writable_output
from .shared.runtime import require_cuda


def validate_control_results(controls, benchmarks):
    """Verify grader operation, not the perfection/speed of reference programs."""
    expected_tasks = {"humaneval": "HumanEval/0", "mbpp": "Mbpp/2"}
    if set(controls) != set(expected_tasks) or set(benchmarks) != set(expected_tasks):
        raise RuntimeError("Missing evaluator control benchmark")
    for name, task in expected_tasks.items():
        control = controls[name]
        if control.get("task_id") != task or control.get("hash") != benchmarks[name]["hash"]:
            raise RuntimeError("Evaluator control identity mismatch")
        if control.get("positive") != {"base": "pass", "plus": "pass"}:
            raise RuntimeError(f"Known-correct evaluator control failed: {name}")
        if control.get("negative") != {"base": "fail", "plus": "fail"}:
            raise RuntimeError(f"Deliberately wrong evaluator control was not rejected: {name}")


def run(argv, cwd=None):
    print("+", " ".join(map(str, argv)), flush=True)
    env = clean_environment() if Path(argv[0]).name in ("singularity", "apptainer") else None
    subprocess.run(list(map(str, argv)), cwd=cwd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    require_cuda()
    root = writable_output(args.output)
    with exclusive(root):
        if (root / "sandbox.json").exists():
            from .coding.sandbox import validate_sandbox
            validate_sandbox(str(root / "sandbox.json"))
            print("Sandbox already provisioned and verified", flush=True)
            return
        source = Path(args.source).resolve()
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != EVALPLUS_COMMIT:
            raise ValueError("Wrong EvalPlus source commit")
        image = root / "python312.sif"
        if not image.exists():
            run([runtime(), "pull", str(image), "docker://python:3.12-slim-bookworm"], root)
        bundle, work = root / "bundle", root / "setup-work"
        bundle.mkdir(exist_ok=True)
        work.mkdir(exist_ok=True)
        spec = {"schema": 2, "image": str(image), "sha256": file_hash(image),
                "bundle": str(bundle), "evalplus_commit": EVALPLUS_COMMIT,
                "image_source": "docker://python:3.12-slim-bookworm"}
        tools = bundle / "tools"
        tools.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "coding/container_prepare.py", tools / "container_prepare.py")
        # This stdlib-only check works before dependencies exist. Later isolation
        # checks repeat it with networking disabled, before executing solutions.
        run(container_command(spec, work, ["python", "/opt/evalplus/tools/container_prepare.py", "storage"],
                              network=True), work)
        wheel_dir = work / "wheel"
        wheel_dir.mkdir(exist_ok=True)
        # Building trusted package metadata is done in this GPU allocation, never on login.
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir",
             "--wheel-dir", str(wheel_dir), str(source)])
        wheels = list(wheel_dir.glob("evalplus-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected one pinned EvalPlus wheel")
        installed = bundle / "installed.json"
        requirements = ROOT / "coding/requirements-evaluator.txt"
        shutil.copy2(requirements, work / requirements.name)
        if not installed.exists():
            run(container_command(spec, work, ["python", "-m", "pip", "install", "--no-cache-dir",
                "--no-compile", "--upgrade", "--target", "/opt/evalplus/python",
                "-r", f"/work/{requirements.name}"],
                network=True, writable_bundle=True), work)
            run(container_command(spec, work, ["python", "-m", "pip", "install", "--no-deps",
                "--no-compile", "--upgrade", "--target", "/opt/evalplus/python",
                f"/work/wheel/{wheels[0].name}"], network=True, writable_bundle=True), work)
            atomic_json(installed, {"evalplus_commit": revision, "requirements_sha256": file_hash(requirements)})
        if read_json(installed) != {"evalplus_commit": revision, "requirements_sha256": file_hash(requirements)}:
            raise ValueError("Existing evaluator has different dependencies; use a fresh sandbox directory")
        from .download_data import RELEASES
        raw = ROOT / "data/sources"
        downloads = read_json(raw / "downloads.json")
        for name, version in RELEASES.items():
            filename = f"{name}-{version}.jsonl.gz"
            if file_hash(raw / filename) != downloads["files"][filename]:
                raise ValueError("Downloaded benchmark checksum mismatch")
            shutil.copy2(raw / filename, work / filename)
        # Verify isolation BEFORE executing even the trusted canonical solutions.
        (bundle / "cache").mkdir(exist_ok=True)
        run(container_command(spec, work, ["python", "/opt/evalplus/tools/container_prepare.py", "isolation"]), work)
        run(container_command(spec, work, ["python", "/opt/evalplus/tools/container_prepare.py", "cache"],
                              writable_bundle=True), work)
        # The exact evaluation launch flags must pass isolation before any generated code.
        run(container_command(spec, work, ["python", "/opt/evalplus/tools/container_prepare.py", "isolation"]), work)
        spec.update(bundle_sha256=bundle_hash(bundle), isolation_verified=True,
                    temporary_storage="owned_work_directory",
                    evalplus_wheel_sha256=file_hash(wheels[0]))
        pending = root / "sandbox.pending.json"
        atomic_json(pending, spec)
        benchmarks = read_json(work / "benchmarks.json")
        audit = read_json(work / "reference-audit.json")
        run(container_command(spec, work, ["python", "/opt/evalplus/tools/container_prepare.py", "controls"]), work)
        controls = read_json(work / "control-checks.json")
        validate_control_results(controls, benchmarks)
        # Preserve every attempt and run BOTH benchmarks before the launch gate.
        # The old blanket 100% gate was ours, not part of upstream EvalPlus.
        checks = root / "canonical-checks" / "controls"
        results = {}
        for name, info in benchmarks.items():
            metrics = score_samples(read_rows(work / f"{name}.canonical.jsonl"), name, info,
                                    str(pending), checks)
            results[name] = metrics
            print(f"Canonical {name}: {metrics}", flush=True)
        # Reference solutions may fail correctness or resource checks themselves.
        # Keep their exact scores and failures; never turn them into model-score
        # exceptions or require a non-upstream all-reference-solutions-pass gate.
        spec.update(benchmarks=benchmarks, canonical_validation=results,
                    canonical_checks_path=str(checks), polynomial_reference_audit=audit,
                    evaluator_controls=controls,
                    canonical_validation_policy="diagnostic_only; launch gated by isolation, integrity and positive/negative evaluator controls")
        atomic_json(root / "sandbox.json", spec)
        print("Pinned sandbox, isolation and evaluator controls passed; full canonical diagnostics saved", flush=True)


if __name__ == "__main__":
    main()

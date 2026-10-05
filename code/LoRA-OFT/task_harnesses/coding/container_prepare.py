"""Trusted preparation helper, executed ONLY inside the isolated container."""
import argparse
import json
import math
import os
import shutil
import socket
import tempfile
from pathlib import Path


def temporary_storage(work="/work", paths=("/tmp", "/var/tmp")):
    """Fail before package installation if temp writes still use session tmpfs."""
    for path in (*paths, tempfile.gettempdir()):
        if not Path(work).samefile(path):
            raise RuntimeError(f"Temporary storage is not the owned work directory: {path}")
    with tempfile.TemporaryFile(dir=paths[0]) as probe:
        probe.write(b"evalplus temporary storage probe\n")
        probe.flush()
    print(json.dumps({"temporary_storage": "owned_work_directory",
                      "paths": [str(path) for path in paths],
                      "filesystem_free_bytes": shutil.disk_usage(work).free}), flush=True)


def isolation():
    temporary_storage()
    for path in ("/home", "/Users", "/mnt"):
        # A dummy home ancestor is harmless; a mounted host filesystem is not.
        mounts = [line.split()[4] for line in Path('/proc/self/mountinfo').read_text().splitlines()]
        if path in mounts:
            raise RuntimeError(f"Host filesystem exposed inside container: {path}")
    if {name for _, name in socket.if_nameindex()} - {"lo"}:
        raise RuntimeError("Container network is not isolated")
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if os.environ.get(key):
            raise RuntimeError("Host credentials were inherited")
    for path in (Path("/opt/evalplus/python"), Path("/opt/evalplus/cache")):
        if not path.is_dir():
            raise RuntimeError(f"Missing evaluator directory: {path}")
        try:
            with (path / ".isolation-write-probe").open("x"):
                pass
        except OSError:
            continue
        raise RuntimeError("Evaluator bundle is writable")
    print(json.dumps({"isolation": "passed", "network": "loopback_only", "bundle": "read_only"}), flush=True)


def cache():
    import gzip
    from evalplus.data.utils import CACHE_DIR
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    for name, version in (("HumanEvalPlus", "v0.1.10"), ("MbppPlus", "v0.2.0")):
        with gzip.open(f"/work/{name}-{version}.jsonl.gz", "rb") as src, \
                (Path(CACHE_DIR) / f"{name}-{version}.jsonl").open("wb") as dest:
            shutil.copyfileobj(src, dest)
    from evalplus.data import (get_human_eval_plus, get_human_eval_plus_hash,
                               get_mbpp_plus, get_mbpp_plus_hash, write_jsonl)
    from evalplus.evaluate import get_groundtruth
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS, _poly
    information = {}
    reference_audit = {}
    for name, version, getter, hasher, special in (
        ("humaneval", "v0.1.10", get_human_eval_plus, get_human_eval_plus_hash, []),
        ("mbpp", "v0.2.0", get_mbpp_plus, get_mbpp_plus_hash, MBPP_OUTPUT_NOT_NONE_TASKS)):
        problems = getter(version=version)
        fingerprint = hasher(version=version)
        oracle = get_groundtruth(problems, fingerprint, special)
        if name == "humaneval":
            problem = problems["HumanEval/32"]
            if problem["entry_point"] != "find_zero":
                raise RuntimeError("Unexpected HumanEval/32 entry point")
            reference_audit = polynomial_reference_audit(problem, oracle["HumanEval/32"], _poly)
            reference_audit["dataset_hash"] = fingerprint
        write_jsonl(f"/work/{name}.canonical.jsonl", [
            {"task_id": task_id, "solution": p["prompt"] + p["canonical_solution"]}
            for task_id, p in problems.items()])
        information[name] = {"version": version, "hash": fingerprint, "n": len(problems)}
    Path("/work/benchmarks.json").write_text(json.dumps(information, indent=2) + "\n")
    Path("/work/reference-audit.json").write_text(json.dumps(reference_audit, indent=2, allow_nan=False) + "\n")


def polynomial_reference_audit(problem, oracle, polynomial):
    """Check the reference outputs against upstream's independent root oracle.

    get_groundtruth records the canonical Newton solver's return values without
    testing convergence. EvalPlus grades find_zero by |poly(x)| <= atol instead
    of equality to those values. A reference return is not necessarily a pass.
    This diagnoses reference solutions only; model scoring is never changed.
    """
    report = {"task_id": problem["task_id"], "entry_point": problem["entry_point"],
              "atol": problem["atol"], "failures": {}}
    for split in ("base", "plus"):
        inputs, outputs = problem[f"{split}_input"], oracle[split]
        if len(inputs) != len(outputs):
            raise RuntimeError("Incomplete reference outputs")
        failures = []
        for index, (inp, value) in enumerate(zip(inputs, outputs)):
            residual = abs(polynomial(*inp, value))
            if not math.isfinite(value) or not math.isfinite(residual):
                raise RuntimeError("Nonfinite polynomial reference output/residual")
            if residual > problem["atol"]:
                failures.append({"index": index, "input": inp, "output": value,
                                 "abs_residual": residual})
        report["failures"][split] = failures
        report[f"{split}_n"] = len(inputs)
    print(json.dumps({"polynomial_reference_audit": report}), flush=True)
    return report


def controls():
    """Run known-positive/negative controls through the actual upstream grader.

    These setup-only controls do not subset or change model evaluation. Avoid a
    false assumption that every upstream reference passes extreme tests within
    the same default time/memory limits applied to model completions.
    """
    from evalplus.data import (get_human_eval_plus, get_human_eval_plus_hash,
                               get_mbpp_plus, get_mbpp_plus_hash)
    from evalplus.evaluate import check_correctness, get_groundtruth
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    report = {}
    for name, task, version, getter, hasher, special in (
        ("humaneval", "HumanEval/0", "v0.1.10", get_human_eval_plus, get_human_eval_plus_hash, []),
        ("mbpp", "Mbpp/2", "v0.2.0", get_mbpp_plus, get_mbpp_plus_hash, MBPP_OUTPUT_NOT_NONE_TASKS)):
        problems = getter(version=version)
        fingerprint = hasher(version=version)
        oracle = get_groundtruth(problems, fingerprint, special)[task]
        problem = problems[task]
        entry = problem["entry_point"]
        if not entry.isidentifier():
            raise RuntimeError("Invalid control entry point")
        report[name] = {"task_id": task, "hash": fingerprint}
        for label, solution, expected in (
            ("positive", problem["prompt"] + problem["canonical_solution"], "pass"),
            ("negative", f"def {entry}(*args, **kwargs):\n    return None\n", "fail")):
            checked = check_correctness(name, 0, problem, solution, oracle, fast_check=False)
            report[name][label] = {split: checked[split][0] for split in ("base", "plus")}
            for split in ("base", "plus"):
                status, details = checked[split]
                if (status != expected or len(details) != len(problem[f"{split}_input"])
                        or any(bool(value) != (expected == "pass") for value in details)):
                    raise RuntimeError(f"Evaluator control failed: {name} {label} {split}")
        print(json.dumps({"evaluator_control": report[name], "benchmark": name}), flush=True)
    Path("/work/control-checks.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("cache", "isolation", "storage", "controls"))
    args = parser.parse_args()
    {"cache": cache, "isolation": isolation, "storage": temporary_storage, "controls": controls}[args.stage]()

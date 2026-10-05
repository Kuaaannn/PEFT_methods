"""Inspect or analyze selected Qwen/Llama checkpoints using shared SPECINT.

--dry-run reads only metadata. Execution requires frozen token banks and an
available CUDA GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import subprocess
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loraoft.checkpoint_analysis import inspect_run_resilient, shared_identity, digest
from loraoft.analysis_artifacts import prune_transient_artifacts, write_cleanup_report
from loraoft.paths import reject_quarantined
from loraoft.protocol_plan import resolve_plan, validate_plan


def terminate_cleanly(signum, _frame):
    """Let worker ``finally`` blocks prune tensors on termination."""
    raise SystemExit(128 + signum)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", default=[], help="explicit selected run (repeatable)")
    p.add_argument("--runs", action="append", default=[], help="root containing run directories")
    p.add_argument("--only", help="comma-separated validation-selected run directory names")
    p.add_argument("--steps", help="comma-separated saved steps; default: manifest eval cadence")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--banks", help="frozen token/outcome banks JSON for this backbone")
    p.add_argument("--base", help="local safetensors base snapshot; default: cached model_id")
    p.add_argument("--output", default="results/checkpoint_analysis")
    p.add_argument("--panel", choices=("core", "detailed"), default="core",
                   help="contract-expanded protocol panel; detailed includes all five draws")
    p.add_argument("--band-d-rel", type=float, default=0.01,
                   help="d/||s*|| for the detailed equal-energy band pilot")
    p.add_argument("--plan", help="JSON extra-cell list or exact hash-pinned source-cell subset")
    p.add_argument("--evaluation-id", default="v1", help="new ID for corrected/repeated evaluations")
    p.add_argument("--activation-layers",
                   help="layers for the optional frozen-input diagnostic only; interventions always use all adapted matrices")
    p.add_argument("--activation-inputs", action="store_true",
                   help="capture base/trained common inputs and measure output-edit energy")
    p.add_argument("--generation-batch-size", type=int, default=8,
                   help="GPU batch size for frozen GSM8K generation; failures bisect to one example")
    p.add_argument("--inference-dtype", choices=("float32", "bfloat16"), default="bfloat16",
                   help="forward compute dtype; KL log-softmax and geometry remain FP32")
    p.add_argument("--vllm-socket", help="local checkpoint vLLM worker socket for GSM8K")
    p.add_argument("--vllm-scratch", help="job-local transient directory for weight patches")
    p.add_argument("--verify-standard-parity", action="store_true",
                   help="pilot-only full-checkpoint versus patch loading check")
    p.add_argument("--factor-cache",
                   help="optional job-local directory for large regenerable SVD factors")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--retain-transient-artifacts", action="store_true",
                   help="debug only: retain large regenerable tensor caches")
    p.add_argument("--restoration-only", action="store_true", help="verify and measure restoration, then stop")
    a = p.parse_args()
    a.random_geometry = False
    if a.activation_inputs != bool(a.activation_layers):
        p.error("--activation-inputs and --activation-layers must be supplied together")
    if bool(a.vllm_socket) != bool(a.vllm_scratch):
        p.error("--vllm-socket and --vllm-scratch must be supplied together")
    try:
        plan, plan_metadata = resolve_plan(a.plan, panel=a.panel, band_d_rel=a.band_d_rel)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    if a.restoration_only:
        plan = plan[:1]
        plan_metadata["source_cell_ids"] = plan_metadata["source_cell_ids"][:1]
    if not a.run and not a.runs:
        p.error("supply --run or --runs")
    identity = shared_identity()
    runs = {Path(r).resolve() for r in a.run}
    discovery_failures = []
    for root in a.runs:
        try:
            runs.update(r.resolve() for r in Path(root).iterdir() if (r / "manifest.json").is_file())
        except OSError as exc:
            discovery_failures.append({"checkpoint_id": str(root), "status": "failed",
                                       "stage": "discovery", "status_reason": f"{type(exc).__name__}: {exc}"})
    if a.only:
        names = set(a.only.split(","))
        missing = names - {r.name for r in runs}
        for name in sorted(missing):
            discovery_failures.append({"checkpoint_id": name, "status": "failed",
                                       "stage": "discovery", "status_reason": "selected run not found"})
        runs = {r for r in runs if r.name in names}
    steps = [int(s) for s in a.steps.split(",")] if a.steps else None
    checkpoints = []
    for run in sorted(runs):
        try:
            checkpoints.extend(inspect_run_resilient(run, steps))
        except Exception as exc:
            discovery_failures.append({"checkpoint_id": run.name, "status": "failed",
                                       "stage": "discovery", "status_reason": f"{type(exc).__name__}: {exc}"})
    for failure in discovery_failures:
        print(json.dumps(failure), file=sys.stderr)
    if not checkpoints:
        p.error("no valid checkpoints selected; see discovery failures")
    print(json.dumps({**identity, "runs": len(runs), "checkpoints": len(checkpoints),
                      "measurement_plan": {**plan_metadata, "cells": plan}}))
    for ck in checkpoints:
        print(f"{ck['status']:6} {ck['checkpoint_id']} {ck['status_reason'] or ''}")
    if a.dry_run:
        return
    if not a.banks:
        p.error("execution requires --banks with frozen selection/target/general inputs")
    from loraoft.checkpoint_analysis import CheckpointAnalysis, require_gpu
    from loraoft.protocol_metrics import FrozenTokenEvaluator
    require_gpu()
    output = reject_quarantined(a.output)
    output.mkdir(parents=True, exist_ok=True)
    if discovery_failures:
        with (output / "failures.jsonl").open("a") as dst:
            for failure in discovery_failures:
                dst.write(json.dumps(failure) + "\n")
    if not a.worker:
        # A CUDA context cannot reliably recover after device-side errors.
        # Each checkpoint has a fresh process, so it cannot poison another run.
        failed = len(discovery_failures)
        for ck in checkpoints:
            cmd = [sys.executable, "-m", "scripts.run_interventions", "--worker",
                   "--run", ck["run"], "--steps", str(ck["step"]), "--banks", a.banks,
                   "--output", str(output), "--evaluation-id", a.evaluation_id]
            for flag, value in (("--base", a.base),
                                ("--activation-layers", a.activation_layers),
                                ("--plan", a.plan)):
                if value:
                    cmd.extend([flag, value])
            for flag, value in (("--vllm-socket", a.vllm_socket),
                                ("--vllm-scratch", a.vllm_scratch),
                                ("--factor-cache", a.factor_cache)):
                if value:
                    cmd.extend([flag, value])
            if a.restoration_only:
                cmd.append("--restoration-only")
            if a.retain_transient_artifacts:
                cmd.append("--retain-transient-artifacts")
            cmd.extend(["--panel", a.panel, "--band-d-rel", str(a.band_d_rel)])
            if a.activation_inputs:
                cmd.append("--activation-inputs")
            cmd.extend(["--generation-batch-size", str(a.generation_batch_size)])
            cmd.extend(["--inference-dtype", a.inference_dtype])
            if a.verify_standard_parity:
                cmd.append("--verify-standard-parity")
            completed = subprocess.run(cmd, check=False)
            failed += completed.returncode != 0
            print(json.dumps({"checkpoint_id": ck["checkpoint_id"], "worker_returncode": completed.returncode}), flush=True)
        raise SystemExit(1 if failed else 0)
    # The production wrapper owns termination and first kills this worker's
    # process group before pruning.  A direct invocation owns its own cleanup.
    wrapper_managed = bool(os.environ.get("LORA_OFT_WRAPPER_MANAGED"))
    if not wrapper_managed:
        signal.signal(signal.SIGTERM, terminate_cleanly)
        signal.signal(signal.SIGINT, terminate_cleanly)
    failures = 0
    for ck in checkpoints:
        analysis = None
        work = None
        failures_before_checkpoint = failures
        started = time.perf_counter()
        try:
            if ck["status"] != "ok":
                raise ValueError(ck["status_reason"])
            activation_layers = ([int(n) for n in a.activation_layers.split(",")]
                                 if a.activation_layers else None)
            analysis = CheckpointAnalysis(ck, output, FrozenTokenEvaluator(
                                          a.banks, generation_batch_size=a.generation_batch_size,
                                          inference_dtype=a.inference_dtype,
                                          vllm_socket=a.vllm_socket,
                                          vllm_scratch=a.vllm_scratch,
                                          verify_standard_parity=a.verify_standard_parity), a.base,
                                          a.evaluation_id, activation_layers, a.activation_inputs,
                                          a.factor_cache,
                                          evaluation_plan={**plan_metadata, "cells": plan})
            analysis.prepare()
            work = analysis.work
            analysis.prepare_references()
            for cell in plan:
                try:
                    rec = analysis.run_cell(cell["operator"], cell.get("params", {}))
                    print(f"{ck['checkpoint_id']} {cell['operator']}: {rec['status']}", flush=True)
                    failures += rec["execution_status"] != "success"
                    if cell["operator"] == "restore" and not analysis.restoration_success:
                        raise RuntimeError("Restoration failed; this checkpoint is gated, other checkpoints continue")
                except Exception as exc:
                    if cell["operator"] == "restore":
                        raise
                    failures += 1
                    print(f"FAILED CELL {cell}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                    with (output / "failures.jsonl").open("a") as dst:
                        dst.write(json.dumps({"checkpoint_id": ck["checkpoint_id"], "cell": cell,
                                              "status": "failed", "status_reason": str(exc)}) + "\n")
            # Optional diagnostic, never a prerequisite for the causal panel.
        except Exception as exc:
            failures += 1
            # Missing/failed checkpoints remain visible instead of shortening a trajectory.
            with (output / "failures.jsonl").open("a") as dst:
                dst.write(json.dumps({**identity,
                    "record_id": digest([ck["checkpoint_id"], a.evaluation_id, uuid.uuid4().hex]),
                    "checkpoint_id": ck["checkpoint_id"],
                    "checkpoint_hash": getattr(analysis, "checkpoint_hash", None),
                    "module_manifest_hash": getattr(analysis, "module_hash", None),
                    "operator_id": "trained", "operator_params": {"stage": "prepare"},
                    "rng_key": None, "dtype_profile": getattr(analysis, "profile", None),
                    "bank_hash": analysis.evaluator.bank_hash if analysis else None,
                    "status": "failed", "metrics": None, "geometry": None,
                    "status_reason": f"{type(exc).__name__}: {exc}",
                    "wall_seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                    "hardware": None}, allow_nan=False) + "\n")
            print(f"FAILED {ck['checkpoint_id']}: {exc}", file=sys.stderr)
        finally:
            if work is None and analysis is not None:
                work = getattr(analysis, "work", None)
            del analysis
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            if (work is not None and not a.retain_transient_artifacts
                    and not wrapper_managed):
                report = prune_transient_artifacts(work)
                write_cleanup_report(
                    work, report,
                    outcome=("analysis_success" if failures == failures_before_checkpoint
                             else "analysis_failed"),
                )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

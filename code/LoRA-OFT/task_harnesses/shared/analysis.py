"""Streaming analysis of every adapted matrix; math entry points remain untouched."""
from __future__ import annotations

import copy
import os
import re
import shutil
import tempfile
from pathlib import Path

from .checkpoint import adapted_layers, load_run
from .data import load_bank
from .evaluation import base_control, result_identity, save_evaluation, with_base
from .io import atomic_json, digest, exclusive, read_json, writable_output
from .runtime import base_model, bridge, require_cuda

DEFAULT_CELLS = (1, 3, 4, 22, 23, 24, 25, 26, 27)


def cell_specs(ids):
    bridge()
    from specint.plan import PLAN_40
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 1 <= i <= len(PLAN_40) for i in ids):
        raise ValueError("Choose unique canonical SPECINT cell IDs 1..40")
    return [(i, *PLAN_40[i - 1]) for i in ids]


def fixed_base(config, name, weight, root, torch):
    from singular_rotation import compute_base_svd, load_reference, reference_path, save_reference
    metadata = {"model_id": config.model_id, "revision": config.model_revision,
                "matrix_name": name, "svd_dtype": "float32", "source_dtype": "bfloat16",
                "svd_driver": "gesvd", "reference_schema": "new-task-fixed-gauge-v1"}
    path = reference_path(Path(root), f"{config.model_id}-{config.model_revision}", name)
    with exclusive(path.parent, lock_name=path.name + ".lock", blocking=True):
        if not path.exists():
            estimate = 4 * min(weight.shape) * (sum(weight.shape) + 1)
            if shutil.disk_usage(path.parent).free < estimate + (2 << 30):
                raise OSError("Insufficient space for immutable base SVD")
            save_reference(path, compute_base_svd(weight, "gesvd"), metadata, torch)
        return load_reference(path, weight, metadata, torch)


def install_trained(layers, reference, torch):
    """Exact-copy restore + deterministic native merge, never BF16 unmerge subtraction."""
    parameters = dict(reference.named_parameters())
    with torch.no_grad():
        for name, layer in layers.items():
            layer.get_base_layer().weight.copy_(parameters[name])
            layer.merged_adapters.clear()
            layer.merge(safe_merge=True)


def factor_pair(name, w0, trained, base, endpoint, torch):
    from specint.ops import Factors, rebuild, _fro
    f = Factors(name=name, W0=w0.float(), W_star=trained.float(),
                U=endpoint["U"], s_star=endpoint["singular_values"], Vh=endpoint["Vh"],
                s0=base["singular_values"], U0=base["U"], s0_basis=base["singular_values"],
                Vh0=base["Vh"], solver="gesvd")
    f.rebuild_floor = _fro(rebuild(f.U, f.s_star, f.Vh) - f.W_star) / max(f.W0_norm, 1e-30)
    f.base_rebuild_floor = base["relative_reconstruction_residual"]
    return f


def intervene(run, output, references, scratch, evaluator, *, split="test", cells=DEFAULT_CELLS, sandbox=None):
    torch = require_cuda()
    from .checkpoint import inspect_run
    config, _ = inspect_run(run)
    if config.task == "coding" and split == "test":
        if not sandbox:
            raise ValueError("Coding test intervention requires --sandbox")
        from ..coding.sandbox import validate_sandbox
        validate_sandbox(sandbox)
    bridge()
    import specint
    from specint.plan import build_edit, randomness_id
    from singular_rotation import compute_base_svd
    output, references, scratch = (writable_output(p) for p in (output, references, scratch))
    scratch.mkdir(parents=True, exist_ok=True)
    requests = cell_specs(cells)
    with exclusive(output):
        config, manifest, model, tokenizer = load_run(run, keep_adapter=True)
        _, bank = load_bank(config.data_manifest, config.task)
        identity = {**result_identity(config, manifest, split, sandbox), "specint_hash": specint.library_hash(),
                    "matrix_scope": "all_adapted", "cells": list(cells)}
        report = output / "analysis.json"
        if report.exists() and read_json(report)["identity"] != identity:
            raise ValueError("Refusing to mix incompatible intervention results")
        layers = adapted_layers(model, manifest)
        reference = base_model(config).eval()
        base_parameters = dict(reference.named_parameters())
        # Capacity estimate includes only one endpoint factor bank, never per-cell checkpoints.
        estimate = sum(4 * min(p.get_base_layer().weight.shape) *
                       (sum(p.get_base_layer().weight.shape) + 1) for p in layers.values())
        if shutil.disk_usage(scratch).free < int(estimate * 1.15) + (2 << 30):
            raise OSError(f"Need approximately {estimate / 2**30:.1f} GiB temporary endpoint factors")
        atomic_json(report, {"identity": identity, "status": "running", "factor_cache_bytes_estimate": estimate})
        baseline = base_control(config, tokenizer, reference, bank, split, evaluator, sandbox)
        install_trained(layers, reference, torch)
        model.config.use_cache = True
        # TemporaryDirectory removes only this run-owned cache on normal completion/errors.
        # A hard SIGKILL can leave it behind; scratch inventory is supplied by preflight.
        with tempfile.TemporaryDirectory(prefix="task-factors-", dir=scratch) as temporary:
            cache = Path(temporary)
            atomic_json(cache / "owner.json", {"run": str(run), "output": str(output),
                                               "job_id": None,
                                               "identity": identity})
            completed = []
            for cell_id, operator, params in requests:
                cell_output = output / f"cell_{cell_id:02d}_{operator}"
                cell_identity = {**identity, "cell_id": cell_id, "operator": operator, "params": params}
                result_file = cell_output / "metrics.json"
                if result_file.exists():
                    if read_json(result_file)["identity_hash"] != digest(cell_identity):
                        raise ValueError("Cached cell has a different identity")
                    completed.append(cell_id)
                    continue
                install_trained(layers, reference, torch)
                geometry = []
                feasible = True
                with torch.no_grad():
                    for name, layer in layers.items():
                        weight = layer.get_base_layer().weight
                        if operator == "trained":
                            continue
                        if operator == "base":
                            weight.copy_(base_parameters[name])
                            continue
                        w0 = base_parameters[name].detach()
                        base = fixed_base(config, name, w0, references, torch)
                        endpoint_path = cache / (digest(name) + ".pt")
                        if endpoint_path.exists():
                            endpoint = torch.load(endpoint_path, weights_only=True, map_location="cuda:0")
                        else:
                            endpoint = compute_base_svd(weight, "gesvd")
                            torch.save(endpoint, endpoint_path)
                        factors = factor_pair(name, w0, weight, base, endpoint, torch)
                        edit = build_edit(factors, operator, params, randomness_id(identity["bank_hash"]))
                        if not edit.feasible:
                            geometry.append({"matrix": name, "feasible": False, "reason": edit.reason})
                            feasible = False
                            break
                        weight.copy_(edit.W.to(weight.dtype))
                        realized = weight.float()
                        # Distances of installed BF16 weights, without another per-cell SVD.
                        geometry.append({"matrix": name, "feasible": True,
                            "edit_norm": float(torch.linalg.vector_norm(realized - factors.W_star)),
                            "base_distance": float(torch.linalg.vector_norm(realized - factors.W0)),
                            "base_norm": factors.W0_norm, "rebuild_floor": factors.rebuild_floor,
                            "metadata": edit.info,
                            "installed_spectrum": None,
                            "spectrum_note": "not re-SVDed; no claim of exact post-quantization spectrum"})
                        del factors, edit, realized, base, endpoint
                atomic_json(cell_output / "geometry.json", geometry)
                if not feasible:
                    atomic_json(cell_output / "infeasible.json", {"identity": cell_identity, "geometry": geometry})
                    raise RuntimeError(f"Cell {cell_id} is infeasible; no partially edited model was evaluated")
                metrics, records = evaluator(model, tokenizer, config, bank, split, reference=reference,
                                             output=cell_output, sandbox=sandbox)
                save_evaluation(cell_output, with_base(metrics, baseline), records, cell_identity)
                completed.append(cell_id)
                atomic_json(report, {"identity": identity, "status": "running", "completed_cells": completed})
        atomic_json(report, {"identity": identity, "status": "complete", "completed_cells": completed,
                             "temporary_factors_removed": True})




def intervention_main(task, evaluator):
    import argparse
    parser = argparse.ArgumentParser(description=f"All-matrix SPECINT intervention: {task}")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--cells", type=int, nargs="+", default=list(DEFAULT_CELLS))
    parser.add_argument("--sandbox")
    args = parser.parse_args()
    from .checkpoint import inspect_run
    config, _ = inspect_run(args.run)
    if config.task != task:
        raise ValueError("Wrong task intervention wrapper")
    intervene(args.run, args.output, args.references, args.scratch, evaluator,
              split=args.split, cells=args.cells, sandbox=args.sandbox)

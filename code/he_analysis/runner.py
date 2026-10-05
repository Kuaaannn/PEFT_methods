"""GPU-only checkpoint execution. Standard native merges; no evaluator or SVD."""
from __future__ import annotations

from collections import defaultdict
import copy
import fcntl
import importlib.metadata
import math
import os
from pathlib import Path
import time

from .plan import atomic_json, canonical_name, check, digest, file_hash, read, require


def require_gpu():
    import torch
    require(torch.cuda.is_available(), "CUDA unavailable; no CPU fallback")
    return torch


def merge_pair(layer, *, safe_merge):
    """Copy one native PEFT layer; never merge/unmerge the live base in place."""
    require(not layer.merged, "Cannot capture W0 from an already merged layer")
    copied = copy.deepcopy(layer)
    base = copied.get_base_layer().weight.detach().clone()
    copied.merge(safe_merge=safe_merge)
    adapted = copied.get_base_layer().weight.detach().clone()
    del copied
    return base, adapted


def load_layers(plan, task, torch):
    from peft import PeftModel
    snapshot = plan["bases"][task["model"]]["path"]
    if plan["domain"] == "llm":
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            snapshot, dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", local_files_only=True).eval()
        # Same default adapter autocast policy as the causal-22 LLM runner.
        model = PeftModel.from_pretrained(base, task["adapter"], is_trainable=False).eval()
    else:
        from diffusers import Flux2Transformer2DModel
        base = Flux2Transformer2DModel.from_pretrained(
            snapshot, subfolder="transformer", torch_dtype=torch.bfloat16,
            local_files_only=True).to("cuda:0").eval()
        model = PeftModel.from_pretrained(base, task["adapter"], torch_device="cuda:0",
                                          autocast_adapter_dtype=True).eval()
    layers = {canonical_name(name + ".weight"): layer for name, layer in model.named_modules()
              if hasattr(layer, "get_base_layer") and hasattr(layer, "merge")}
    expected = {e["name"]: e["shape"] for e in plan["modules"][task["model"]]}
    require(set(layers) == set(expected), "Loaded model is missing/unexpectedly adapting matrices")
    for name, layer in layers.items():
        weight = layer.get_base_layer().weight
        require(weight.is_cuda and weight.dtype == torch.bfloat16 and list(weight.shape) == expected[name],
                f"Wrong base shape/dtype/device: {name}")
    return model, layers


def audit_names(entries):
    groups = defaultdict(list)
    for entry in entries:
        groups[(entry["block_type"], entry["projection_type"])].append(entry)
    selected = set()
    for group in groups.values():
        ordered = sorted(group, key=lambda row: row["depth"])
        for pos in (0, len(ordered) // 2, len(ordered) - 1):
            selected.add(ordered[pos]["name"])
    return selected


def precision_audit(W0, W, count, torch):
    from specint.hyperspherical import compare_hyperspherical_energy as compare
    rows = torch.linspace(0, W.shape[0] - 1, min(count, W.shape[0]), device=W.device).round().long()
    base, adapted = W0[rows], W[rows]
    fast = compare(base, adapted, block_size=256, sample_count=0)
    audit = compare(base, adapted, block_size=256, sample_count=0, matmul_dtype="float64")
    result = {"scope": "fixed_evenly_spaced_row_subset_not_full_matrix_error_bound",
              "row_indices": rows.tolist(), "fp32": fast, "fp64": audit}
    if fast["status"] == audit["status"] == "ok":
        result["absolute_relative_he_disagreement"] = abs(fast["relative_he_change"] - audit["relative_he_change"])
        result["cosine_rms_disagreement"] = abs(fast["pair_cosine_rms_change"] - audit["pair_cosine_rms_change"])
    return result, rows


def mechanism_audit(layer, W0, rows, method, torch):
    """Raw transform only. Never polar-project, never substitute evaluated weights."""
    from specint.hyperspherical import compare_hyperspherical_energy as compare
    active = list(layer.active_adapters)
    require(len(active) == 1, "Exactly one active adapter required")
    name = active[0]
    source = W0[rows].float()
    with torch.autocast(device_type="cuda", enabled=False):
        if method == "oft":
            rotation = layer.get_delta_weight(name).detach().float()
            width = int(layer.oft_block_size[name])
            blocks = torch.stack([rotation[i:i + width, i:i + width]
                                  for i in range(0, rotation.shape[0], width)])
            gram = blocks.transpose(-1, -2) @ blocks
            identity = torch.eye(width, device=source.device)
            defect = float(torch.linalg.vector_norm((gram - identity).double()))
            adapted = (rotation @ source.T).T
            convention = "W = W0 R.T"
        else:
            from peft.tuners.hra.layer import _cwy_factors, _right_multiply_hra
            require(not layer.hra_apply_GS[name], "Expected saved non-GS HRA")
            vectors = layer.hra_u[name].detach().float()
            require(bool(torch.isfinite(vectors).all()) and bool((vectors.norm(dim=0) > 0).all()),
                    "Invalid Householder vectors")
            adapted = _right_multiply_hra(source, vectors, False, reverse=False, cast_input=False)
            u, t = _cwy_factors(vectors)
            gram, t = u.double().T @ u.double(), t.double()
            middle = -t - t.T + t.T @ gram @ t
            product = middle @ gram
            defect = float((product * product.T).sum().clamp_min(0).sqrt())
            convention = "W = W0 Q"
    return {"scope": "fixed_row_subset_raw_transform_diagnostic",
            "convention": convention, "polar_projection": False,
            "orthogonality_fro": defect, "changes_primary_weights": False,
            "energy": compare(source, adapted, block_size=256, sample_count=0)}


def aggregate(records):
    good = [row for row in records if row["measurement"]["status"] == "ok"]
    result = {"matrix_count": len(records), "valid_matrix_count": len(good),
              "invalid_matrix_count": len(records) - len(good)}
    if not good:
        return result
    metrics = [row["measurement"] for row in good]
    for key in ("relative_he_change", "absolute_relative_he_change", "pair_cosine_rms_change",
                "pair_kernel_relative_l1_change"):
        result["macro_" + key] = sum(row[key] for row in metrics) / len(metrics)
    base = sum(row["base"]["he"] for row in metrics)
    # This is energy-weighted (not the primary equal-matrix aggregate).
    result["pooled_relative_he_change"] = sum(
        row["base"]["he"] * row["relative_he_change"] for row in metrics) / base
    pairs = sum(row["unordered_pair_count"] for row in metrics)
    result["pair_weighted_cosine_rms_change"] = (sum(
        row["unordered_pair_count"] * row["pair_cosine_rms_change"] ** 2 for row in metrics) / pairs) ** .5
    return result


def task_directory(plan, index):
    return Path(plan["output_root"]) / f"task_{index:03d}"


def valid_record(path, identity, name):
    if not path.exists():
        return False
    row = read(path)
    require(row["identity"] == identity and row["entry"]["name"] == name, "Refuse mixed HE matrix records")
    measurement = row["measurement"]
    n, d = row["entry"]["shape"]
    require(measurement["shape"] == [n, d] and measurement["neuron_axis"] == "rows"
            and measurement["all_pairs"] and measurement["ordered_pair_count"] == n * (n - 1),
            "Malformed HE matrix record")
    if measurement["status"] == "ok":
        for key in ("relative_he_change", "absolute_relative_he_change", "pair_cosine_rms_change",
                    "pair_kernel_relative_l1_change"):
            require(measurement[key] is not None and math.isfinite(measurement[key]), "Nonfinite HE result")
        require(measurement["base"]["he"] > 0 and measurement["adapted"]["he"] > 0, "Invalid finite HE")
    archive = row.get("sample_archive")
    if archive:
        require((path.parent / archive["name"]).is_file() and
                file_hash(path.parent / archive["name"]) == archive["sha256"], "Sample archive missing/corrupt")
    return True


def task_complete(plan, index):
    """Check identities AND every matrix/sample before skipping a completed pilot."""
    task = plan["tasks"][index]
    directory = task_directory(plan, index)
    path = directory / "metadata.json"
    if not path.exists():
        return False
    metadata = read(path)
    identity = digest({"plan": plan["identity"], "task": task})
    require(metadata["identity"] == identity, "Refuse to mix task identities")
    if metadata["status"] != "complete":
        return False
    entries = plan["modules"][task["model"]]
    return metadata["completed_matrices"] == len(entries) and all(
        valid_record(directory / f"matrix_{i:03d}.json", identity, entry["name"])
        for i, entry in enumerate(entries))


def run(plan, index):
    torch = require_gpu()
    from specint.hyperspherical import VERSION, compare_hyperspherical_energy, library_hash
    import numpy as np
    require(0 <= index < len(plan["tasks"]), "Task index out of range")
    check(plan, index=index)
    task = plan["tasks"][index]
    directory = task_directory(plan, index)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = digest({"plan": plan["identity"], "task": task})
        entries = plan["modules"][task["model"]]
        paths = [directory / f"matrix_{i:03d}.json" for i in range(len(entries))]
        done = [valid_record(path, identity, e["name"]) for path, e in zip(paths, entries)]
        start = time.monotonic()
        weight_sha = file_hash(Path(task["adapter"]) / "adapter_model.safetensors")
        require(not task["expected_weights_sha256"] or weight_sha == task["expected_weights_sha256"],
                "FLUX checkpoint content changed since causal-22 selection")
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            old = read(metadata_path)
            require(old["identity"] == identity and old["adapter_sha256"] == weight_sha,
                    "Refuse to reuse results from different adapter bytes")
            if all(done) and old["status"] == "complete" and (directory / "summary.json").is_file():
                print(f"Already complete and hash-validated: {directory}", flush=True)
                return
        metadata = {"identity": identity, "plan_identity": plan["identity"], "task": task,
                    "version": VERSION, "he_source_hash": library_hash(), "adapter_sha256": weight_sha,
                    "selection_sha256": plan["selection_sha256"],
                    "module_manifest_sha256": digest(entries), "base": plan["bases"][task["model"]],
                    "settings": plan["settings"], "hardware": torch.cuda.get_device_name(0),
                    "execution": "local_cuda",
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ("torch", "peft", "transformers", "numpy")},
                    "status": "running", "resumed_matrices": sum(done)}
        if plan["domain"] == "flux":
            metadata["packages"]["diffusers"] = importlib.metadata.version("diffusers")
        atomic_json(metadata_path, metadata)
        model = layers = None
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        audits = audit_names(entries)
        if not all(done):
            loading = time.monotonic()
            model, layers = load_layers(plan, task, torch)
            metadata["load_seconds"] = time.monotonic() - loading
        settings = plan["settings"]
        with torch.inference_mode():
            for i, (entry, path, complete) in enumerate(zip(entries, paths, done)):
                if complete:
                    print(f"[{i + 1}/{len(entries)}] reuse {entry['name']}", flush=True)
                    continue
                name = entry["name"]
                tick = time.monotonic()
                W0, W = merge_pair(layers[name], safe_merge=plan["domain"] == "llm")
                torch.cuda.synchronize()
                merge_seconds = time.monotonic() - tick
                key = digest({"base": metadata["base"]["revision"], "matrix": name,
                              "sample_protocol": "he-pair-v1"})
                kwargs = {k: settings[k] for k in ("block_size", "matmul_dtype", "repair_below", "sample_count")}
                measure_start = time.monotonic()
                measurement = compare_hyperspherical_energy(W0, W, sample_seed=int(key[:15], 16), **kwargs)
                samples = measurement.pop("samples")
                torch.cuda.synchronize()
                row = {"identity": identity, "entry": entry, "measurement": measurement,
                       "merge_seconds": merge_seconds, "measurement_seconds": time.monotonic() - measure_start}
                if name in audits:
                    row["precision_audit"], rows = precision_audit(W0, W, settings["audit_rows"], torch)
                    if task["method"] in {"oft", "hra"}:
                        row["mechanism_audit"] = mechanism_audit(layers[name], W0, rows, task["method"], torch)
                if samples:
                    sample_path = path.with_suffix(".npz")
                    tmp = sample_path.with_name(sample_path.name + ".tmp")
                    # Passive host serialization only; no CPU numerical analysis.
                    arrays = {k: v.detach().to(device="cpu").numpy().astype(
                        "int32" if k in {"row_i", "row_j"} else "float32", copy=False)
                              for k, v in samples.items()}
                    with tmp.open("wb") as stream:
                        np.savez_compressed(stream, **arrays)
                    tmp.replace(sample_path)
                    row["sample_archive"] = {"name": sample_path.name, "sha256": file_hash(sample_path)}
                row["wall_seconds"] = time.monotonic() - tick
                atomic_json(path, row)
                print(f"[{i + 1}/{len(entries)}] {name}: {measurement['status']}, "
                      f"relative_HE={measurement['relative_he_change']}, {row['wall_seconds']:.2f}s", flush=True)
                del W0, W, measurement, samples
        records = [read(path) for path in paths]
        summary = aggregate(records)
        groups = defaultdict(list)
        for row in records:
            groups[(row["entry"]["block_type"], row["entry"]["projection_type"])].append(row)
        atomic_json(directory / "summary.json", {"identity": identity, "all_matrices": summary,
                    "by_projection": {"/".join(key): aggregate(group) for key, group in groups.items()}})
        metadata.update(status="complete" if summary["invalid_matrix_count"] == 0 else "complete_with_issues",
                        completed_matrices=len(records), wall_seconds=time.monotonic() - start,
                        max_cuda_memory_bytes=torch.cuda.max_memory_allocated())
        atomic_json(metadata_path, metadata)
        del model, layers
        print(f"{metadata['status']}: {len(records)} matrices; {directory}", flush=True)

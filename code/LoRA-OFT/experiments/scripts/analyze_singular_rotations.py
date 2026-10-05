#!/usr/bin/env python3
"""Measure singular-vector rotations for selected LLM adapter checkpoints."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for import_root in (ROOT, WORKSPACE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from singular_rotation import (ANALYSIS_VERSION, CheckpointOutput, analyze_weight_pair,
                               apply_block_rotation_to_weight_right, choose_example_positions,
                               compute_base_svd, library_hash, load_reference,
                               project_oft_rotation, reference_path,
                               render_plots, save_reference, verify_oft_convention)

from loraoft.analysis_selection import expand_selection
from loraoft.checkpoint_analysis import inspect_run_resilient
from loraoft.methods import canonical_name


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    return torch


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_snapshot(checkpoint: dict) -> Path:
    from huggingface_hub import hf_hub_download

    model_id = checkpoint["manifest"]["model_id"]
    config = json.loads((Path(checkpoint["path"]) / "adapter_config.json").read_text())
    candidate = Path(model_id)
    if candidate.is_dir():
        return candidate
    return Path(hf_hub_download(model_id, "config.json", revision=config.get("revision"),
                                local_files_only=True)).parent


def entries_and_layers(wrapped, selected_layers, selected_projections):
    layers = {canonical_name(name + ".weight"): module
              for name, module in wrapped.named_modules()
              if hasattr(module, "get_base_layer") and hasattr(module, "merge")}
    entries = []
    for name, module in sorted(layers.items()):
        match = re.fullmatch(r"model.layers.(\d+).(self_attn|mlp).([a-z_]+).weight", name)
        if not match or module.get_base_layer().weight.ndim != 2:
            raise ValueError(f"Unsupported adapted matrix {name}")
        depth, block_type, role = int(match[1]), match[2], match[3]
        if selected_layers is not None and depth not in selected_layers:
            continue
        if selected_projections and role not in selected_projections:
            continue
        entries.append({"name": name, "depth": depth, "projection_type": role,
                        "block_type": block_type})
    if not entries:
        raise ValueError("The layer/projection selection contains no adapted matrices")
    return entries, layers


def merged_components(layer, is_oft: bool, torch, method=None):
    """Return FP32 scientific weights plus the untouched evaluation realization."""
    copied = copy.deepcopy(layer).to("cuda:0")
    W0_eval = copied.get_base_layer().weight.detach().clone()
    W0 = W0_eval.float()
    rotation = rotation_for_analysis = generator = polar_diagnostic = None
    oft_block_size = None
    active = copied.active_adapters
    if len(active) != 1:
        raise ValueError(f"Expected exactly one active adapter, got {active}")
    if is_oft:
        rotation = copied.get_delta_weight(active[0]).detach().clone()
        rotation_module = copied.oft_R[active[0]]
        oft_block_size = int(copied.oft_block_size[active[0]])
        rotation_for_analysis, polar_diagnostic = project_oft_rotation(
            rotation, oft_block_size)
        generator_blocks = rotation_module._pytorch_skew_symmetric(
            rotation_module.weight.detach().float(), oft_block_size)
        if generator_blocks.shape[0] == 1 and int(copied.r[active[0]]) > 1:
            generator_blocks = generator_blocks.expand(int(copied.r[active[0]]), -1, -1)
        generator = generator_blocks.detach()
        # This is the algebraic OFT endpoint. It is intentionally kept in FP32
        # for geometry; standard evaluation continues to use the normal merge.
        W = apply_block_rotation_to_weight_right(
            rotation_for_analysis, W0, oft_block_size)
    elif method == "hra":
        from peft.tuners.hra.layer import _right_multiply_hra
        if copied.hra_apply_GS[active[0]]:
            raise ValueError("Only the saved non-GS HRA convention is supported")
        with torch.autocast(device_type="cuda", enabled=False):
            W = _right_multiply_hra(W0, copied.hra_u[active[0]].detach().float(),
                                    False, reverse=False, cast_input=False)
    elif method == "dora":
        # get_delta_weight alone omits DoRA's magnitude/normalization. Use its
        # actual PEFT merge for geometry; keep the standard BF16 merge below.
        scientific = copy.deepcopy(copied).float()
        scientific.merge(safe_merge=True)
        W = scientific.get_base_layer().weight.detach().clone()
        del scientific
    else:
        delta = copied.get_delta_weight(active[0]).detach().float()
        W = W0 + delta
    copied.merge(safe_merge=True)
    W_eval = copied.get_base_layer().weight.detach().clone()
    del copied
    torch.cuda.empty_cache()
    return (W0, W, W0_eval, W_eval, rotation, rotation_for_analysis, generator,
            oft_block_size, polar_diagnostic)


def analyze_checkpoint(checkpoint, args, torch, selection_task=None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    adapter = Path(checkpoint["path"])
    base = base_snapshot(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        str(base), dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="eager", local_files_only=True).eval()
    wrapped = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).eval()
    entries, layers = entries_and_layers(wrapped, args.layers, set(args.projection or []))
    expected = json.loads((Path(checkpoint["run"]) / "target_matrices.json").read_text())
    if set(layers) != set(expected):
        raise ValueError("Adapter module coverage disagrees with target_matrices.json")
    example_positions = choose_example_positions(entries)
    manifest = checkpoint["manifest"]
    matched_spectral_width = ({"small": 32, "medium": 64, "large": 128}.get(
        selection_task.get("budget")) if selection_task
        and not args.no_matched_oft_sensitivity else None)
    if args.build_reference_only:
        built = []
        for index, entry in enumerate(entries, 1):
            name = entry["name"]
            path = reference_path(Path(args.reference_root), manifest["model_id"], name)
            W0 = layers[name].get_base_layer().weight.detach().float()
            reference_metadata = {
                "analysis_version": ANALYSIS_VERSION, "base_model": manifest["model_id"],
                "matrix_name": name, "shape": list(W0.shape), "svd_dtype": "float32",
                "svd_driver": args.svd_driver,
            }
            if path.exists():
                print(f"[{index}/{len(entries)}] validating existing fixed base SVD {name}",
                      flush=True)
                artifact = load_reference(path, W0, reference_metadata, torch)
            else:
                print(f"[{index}/{len(entries)}] building fixed base SVD {name}", flush=True)
                artifact = compute_base_svd(W0, args.svd_driver)
                save_reference(path, artifact, reference_metadata, torch)
            built.append({**reference_metadata, "path": str(path),
                          "relative_reconstruction_residual":
                              artifact["relative_reconstruction_residual"]})
            del W0, artifact
            torch.cuda.empty_cache()
        reference_dir = reference_path(Path(args.reference_root), manifest["model_id"],
                                       entries[0]["name"]).parent
        manifest_path = reference_dir / "manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"status": "complete", "entries": built},
                                        indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)
        print(f"completed fixed reference {manifest_path}", flush=True)
        return
    adapter_files = {path.name: file_hash(path) for path in sorted(adapter.iterdir()) if path.is_file()}
    reference_manifest = reference_path(Path(args.reference_root), manifest["model_id"],
                                        entries[0]["name"]).parent / "manifest.json"
    if not reference_manifest.is_file():
        raise FileNotFoundError(f"Missing completed fixed-reference manifest {reference_manifest}")
    metadata = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_library_hash": library_hash(),
        "adapter_source_hash": file_hash(Path(__file__)),
        "checkpoint_id": checkpoint["checkpoint_id"],
        "label": args.label or checkpoint["checkpoint_id"],
        "family": "llm",
        "method": manifest["method"],
        "base_model": manifest["model_id"],
        "step": checkpoint["step"],
        "schedule_fraction": checkpoint["schedule_fraction"],
        "capacity": manifest.get("capacity"),
        "budget": selection_task.get("budget") if selection_task else None,
        "learning_rate": manifest.get("learning_rate"),
        "seed": manifest.get("seed"),
        "run_manifest": manifest,
        "selected_module_manifest": entries,
        "adapter_path": str(adapter),
        "adapter_file_hashes": adapter_files,
        "fixed_base_svd_root": str(Path(args.reference_root)),
        "fixed_base_svd_manifest": str(reference_manifest),
        "fixed_base_svd_manifest_sha256": file_hash(reference_manifest),
        "settings": {"geometry_weight": "fp32 algebraic merge from native base and adapter",
                     "evaluation_weight": "unchanged standard PEFT merge in model dtype",
                     "svd_dtype": "float32", "svd_driver": args.svd_driver,
                     "repeated_rtol": args.repeated_rtol,
                     "repeated_atol": args.repeated_atol,
                     "spectral_block_size": args.spectral_block_size,
                     "reporting_band_count": args.reporting_bands,
                     "matched_oft_width_sensitivity": matched_spectral_width,
                     "top_block_size": args.top_block_size,
                     "block_semantics": {
                         "spectral": "contiguous singular-index subspaces with gap-adjusted boundaries",
                         "oft": "feature-coordinate blocks acted on by the OFT parameterization",
                         "relationship": "independent; equal numeric widths are a sensitivity view only"},
                     "heatmap_components": args.heatmap_components,
                     "pairwise_scope": "top/middle/tail windows for fixed early/middle/late matrices",
                     "primary_claim_scope": "spectral subspaces",
                     "per_index_scope": "leading spectral block only"},
        "example_matrices": example_positions,
        "hardware": torch.cuda.get_device_name(0),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "peft", "numpy")},
    }
    if manifest["method"] == "hra":
        metadata["hra_transport_source_hash"] = file_hash(ROOT / "loraoft/hra_orientation.py")
    output = CheckpointOutput(args.output, metadata)
    is_oft = "oft" in manifest["method"]
    for index, entry in enumerate(entries, 1):
        name = entry["name"]
        print(f"[{index}/{len(entries)}] {checkpoint['checkpoint_id']} {name}", flush=True)
        (W0, W, W0_eval, W_eval, raw_rotation, rotation, generator,
         oft_block_size, polar_diagnostic) = merged_components(
             layers[name], is_oft, torch, method=manifest["method"])
        fixed_svd_path = reference_path(Path(args.reference_root), manifest["model_id"], name)
        fixed_svd = load_reference(fixed_svd_path, W0, {
            "analysis_version": ANALYSIS_VERSION, "base_model": manifest["model_id"],
            "matrix_name": name, "shape": list(W0.shape), "svd_dtype": "float32",
            "svd_driver": args.svd_driver,
        }, torch)
        convention = None
        transport_options = {}
        if manifest["method"] == "hra":
            from loraoft.hra_orientation import transported_bases
            from peft.tuners.hra.layer import _right_multiply_hra
            active = layers[name].active_adapters[0]
            opt_u = layers[name].hra_u[active].detach()
            replay = _right_multiply_hra(W0_eval, opt_u, False).to(W_eval.dtype)
            error = float(torch.linalg.vector_norm(replay.float() - W_eval.float()) /
                          torch.linalg.vector_norm(W_eval.float()).clamp_min(1e-30))
            convention = {"method": "hra", "expected": "right", "verified": error <= 1e-6,
                          "standard_merge_replay_relative_error": error,
                          "primary_u": "fixed_U0", "primary_v": "Q.T @ V0"}
            if not convention["verified"]:
                raise RuntimeError(f"HRA right-side convention verification failed: {convention}")
            transport_options = {
                "right_basis_transport": lambda v: transported_bases(
                    {"U": fixed_svd["U"], "Vh": v.T}, opt_u)[1].T,
                "transport_method": "hra"}
            del replay
        if raw_rotation is not None:
            convention = verify_oft_convention(W0_eval, W_eval, raw_rotation,
                                               tolerance=args.convention_tolerance,
                                               block_size=oft_block_size)
            if not convention["verified"]:
                raise RuntimeError(f"OFT convention verification failed for {name}: {convention}")
            convention["polar_projection"] = polar_diagnostic
        result = analyze_weight_pair(
            W0, W, name=name, projection_type=entry["projection_type"], depth=entry["depth"],
            repeated_rtol=args.repeated_rtol, repeated_atol=args.repeated_atol,
            heatmap_components=args.heatmap_components,
            retain_pairwise=name in example_positions, driver=args.svd_driver,
            rotation=rotation, oft_generator=generator,
            oft_coordinate_block_size=oft_block_size,
            spectral_block_size=args.spectral_block_size,
            sensitivity_block_size=matched_spectral_width,
            reporting_band_count=args.reporting_bands,
            top_block_size=args.top_block_size, base_svd=fixed_svd, **transport_options)
        result["layer"]["example_positions"] = example_positions.get(name, [])
        result["layer"]["standard_eval_relative_to_fp32_geometry"] = float(
            torch.linalg.vector_norm(W_eval.float() - W) /
            torch.linalg.vector_norm(W).clamp_min(1e-30))
        output.add(result, convention)
        del W0, W, W0_eval, W_eval, raw_rotation, rotation, generator
        del polar_diagnostic, fixed_svd, result
        gc.collect()
        torch.cuda.empty_cache()
    directory = output.finish()
    del wrapped, model, layers
    gc.collect()
    torch.cuda.empty_cache()
    print(f"completed {directory}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append",
                        help="run directory; repeat to analyze multiple runs")
    parser.add_argument("--selection-manifest",
                        help="shared pinned selection used by intervention analysis")
    parser.add_argument("--index", type=int, help="task index within --selection-manifest")
    parser.add_argument("--step", dest="steps", action="append", type=int,
                        help="saved step; repeat as needed (default: every saved checkpoint)")
    parser.add_argument("--output", default="results/singular_rotations")
    parser.add_argument("--reference-root", default="results/singular_rotations/base_references_v2",
                        help="immutable FP32 base-SVD artifact root")
    parser.add_argument("--build-reference-only", action="store_true",
                        help="create the fixed reference and do not analyze checkpoints")
    parser.add_argument("--label", help="optional plot label (best with one checkpoint)")
    parser.add_argument("--budget", choices=("small", "medium", "large"),
                        help="Paper capacity group when running without a selection manifest")
    parser.add_argument("--layers", type=lambda x: {int(v) for v in x.split(",")})
    parser.add_argument("--projection", action="append",
                        help="projection role such as q_proj; repeat to select several")
    parser.add_argument("--repeated-rtol", type=float, default=1e-3)
    parser.add_argument("--repeated-atol", type=float, default=0.0)
    parser.add_argument("--heatmap-components", type=int, default=128)
    parser.add_argument("--spectral-block-size", type=int, default=128,
                        help="target singular-index width; independent of OFT coordinate blocks")
    parser.add_argument("--top-block-size", type=int, default=128,
                        help="maximum number of leading per-index rows")
    parser.add_argument("--reporting-bands", type=int, default=8,
                        help="number of directly measured broad paper bands")
    parser.add_argument("--no-matched-oft-sensitivity", action="store_true",
                        help="omit secondary coordinate-width blocks in full-grid runs")
    parser.add_argument("--svd-driver", choices=("gesvd", "gesvdj"), default="gesvd")
    parser.add_argument("--convention-tolerance", type=float, default=5e-3)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.selection_manifest) != (args.index is not None):
        raise SystemExit("--selection-manifest and --index must be supplied together")
    if args.selection_manifest and args.run:
        raise SystemExit("choose --run or --selection-manifest/--index, not both")
    if not args.selection_manifest and not args.run:
        raise SystemExit("supply --run or --selection-manifest/--index")
    torch = require_gpu()
    requests = []
    if args.selection_manifest:
        _, tasks = expand_selection(args.selection_manifest)
        if not 0 <= args.index < len(tasks):
            raise SystemExit(f"--index must be in 0..{len(tasks) - 1}")
        task = tasks[args.index]
        requests.append((task["run"], [task["checkpoint_step"]], task))
    else:
        requests.extend((run, args.steps, {"budget": args.budget} if args.budget else None) for run in args.run)
    for run, steps, selection_task in requests:
        for checkpoint in inspect_run_resilient(run, steps):
            if checkpoint["status"] != "ok":
                raise RuntimeError(f"{checkpoint['checkpoint_id']}: {checkpoint['status_reason']}")
            analyze_checkpoint(checkpoint, args, torch, selection_task)
    if not args.no_plots:
        render_plots(args.output)


if __name__ == "__main__":
    main()

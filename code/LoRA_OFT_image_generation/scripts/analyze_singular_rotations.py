#!/usr/bin/env python3
"""Measure singular-vector rotations for selected FLUX LoRA/OFT checkpoints."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, WORKSPACE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from singular_rotation import (ANALYSIS_VERSION, CheckpointOutput, analyze_weight_pair,
                               apply_block_rotation_to_weight_right, choose_example_positions,
                               compute_base_svd, library_hash, load_reference,
                               project_oft_rotation, reference_path,
                               render_plots, save_reference, verify_oft_convention)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    return torch


def selected_entries(manifest, layers, projections):
    entries = []
    for entry in manifest.entries:
        projection_type = f"{entry.block_type}:{entry.role}"
        if layers is not None and entry.layer not in layers:
            continue
        if projections and entry.role not in projections and projection_type not in projections:
            continue
        entries.append({"name": entry.name, "depth": entry.layer,
                        "projection_type": projection_type, "entry": entry})
    if not entries:
        raise ValueError("The layer/projection selection contains no adapted matrices")
    return entries


def analyze_checkpoint(checkpoint, args, torch):
    from pipeline import manifest as M
    from pipeline import merge as MG
    from pipeline import model as MD
    from utils import get_torch_dtype

    cfg = M.load_train_config(checkpoint)
    dtype = get_torch_dtype(cfg.dtype)
    transformer = MD.load_transformer(dtype=dtype, device="cuda:0")
    state = MD.adapter_state_dict(checkpoint)
    wrapped = MD.attach(checkpoint, transformer=transformer, dtype=dtype, device="cuda:0",
                        autocast_adapter_dtype=cfg.autocast_adapter_dtype).eval()
    module_manifest = MG.build_manifest(wrapped, state)
    entries = selected_entries(module_manifest, args.layers, set(args.projection or []))
    by_name = {entry["name"]: entry for entry in entries}
    example_positions = choose_example_positions(entries)
    peft_layers = MG._peft_layers(wrapped)
    matched_spectral_width = {"xs": 32, "small": 64, "medium": 128,
                              "large": 256}.get(args.budget)
    if args.build_reference_only:
        built = []
        for index, name in enumerate(sorted(by_name), 1):
            _, W0, _, merge_check = next(MG.iter_pairs(
                wrapped, module_manifest, device="cuda:0", check=True, names=[name]))
            if not merge_check.passed:
                raise RuntimeError(f"Native/merged check failed for {name}: {merge_check}")
            path = reference_path(Path(args.reference_root), M.BASE_MODEL, name)
            reference_metadata = {
                "analysis_version": ANALYSIS_VERSION, "base_model": M.BASE_MODEL,
                "matrix_name": name, "shape": list(W0.shape), "svd_dtype": "float32",
                "svd_driver": args.svd_driver,
            }
            if path.exists():
                print(f"[{index}/{len(by_name)}] validating existing fixed base SVD {name}",
                      flush=True)
                artifact = load_reference(path, W0.float(), reference_metadata, torch)
            else:
                print(f"[{index}/{len(by_name)}] building fixed base SVD {name}", flush=True)
                artifact = compute_base_svd(W0.float(), args.svd_driver)
                save_reference(path, artifact, reference_metadata, torch)
            built.append({**reference_metadata, "path": str(path),
                          "relative_reconstruction_residual":
                              artifact["relative_reconstruction_residual"]})
            del W0, artifact
            torch.cuda.empty_cache()
        reference_dir = reference_path(Path(args.reference_root), M.BASE_MODEL,
                                       sorted(by_name)[0]).parent
        manifest_path = reference_dir / "manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"status": "complete", "entries": built},
                                        indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)
        wrapped.unload()
        print(f"completed fixed reference {manifest_path}", flush=True)
        return
    reference_manifest = reference_path(
        Path(args.reference_root), M.BASE_MODEL, sorted(by_name)[0]).parent / "manifest.json"
    if not reference_manifest.is_file():
        raise FileNotFoundError(f"Missing completed fixed-reference manifest {reference_manifest}")
    metadata = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_library_hash": library_hash(),
        "adapter_source_hash": file_hash(Path(__file__)),
        "checkpoint_id": checkpoint.checkpoint_id,
        "label": args.label or checkpoint.checkpoint_id,
        "family": "diffusion",
        "method": checkpoint.method,
        "base_model": M.BASE_MODEL,
        "capacity_kind": checkpoint.capacity_kind,
        "capacity": checkpoint.capacity,
        "budget": args.budget,
        "learning_rate": checkpoint.lr,
        "seed": checkpoint.seed,
        "subject": checkpoint.subject,
        "max_steps": checkpoint.max_steps,
        "checkpoint_path": str(checkpoint.path()),
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "checkpoint_metadata": checkpoint.as_dict(),
        "module_manifest": module_manifest.as_dict(),
        "fixed_base_svd_root": str(Path(args.reference_root)),
        "fixed_base_svd_manifest": str(reference_manifest),
        "fixed_base_svd_manifest_sha256": file_hash(reference_manifest),
        "settings": {"geometry_weight": "fp32 algebraic merge from native base and adapter",
                     "evaluation_weight": "unchanged native diffusion merge",
                     "weight_dtype": cfg.dtype, "svd_dtype": "float32",
                     "svd_driver": args.svd_driver, "repeated_rtol": args.repeated_rtol,
                     "repeated_atol": args.repeated_atol,
                     "spectral_block_size": args.spectral_block_size,
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
                     for name in ("torch", "diffusers", "peft", "numpy")},
    }
    output = CheckpointOutput(args.output, metadata)
    for index, name in enumerate(sorted(by_name), 1):
        info = by_name[name]
        layer = peft_layers[name]
        print(f"[{index}/{len(by_name)}] {checkpoint.checkpoint_id} {name}", flush=True)
        active = layer.active_adapters
        if len(active) != 1:
            raise ValueError(f"Expected exactly one active adapter, got {active}")
        raw_rotation = rotation = generator = polar_diagnostic = None
        oft_block_size = None
        if checkpoint.method == "oft":
            raw_rotation = layer.get_delta_weight(active[0]).detach().clone()
            rotation_module = layer.oft_R[active[0]]
            oft_block_size = int(layer.oft_block_size[active[0]])
            rotation, polar_diagnostic = project_oft_rotation(raw_rotation, oft_block_size)
            generator_blocks = rotation_module._pytorch_skew_symmetric(
                rotation_module.weight.detach().float(), oft_block_size)
            if generator_blocks.shape[0] == 1 and int(layer.r[active[0]]) > 1:
                generator_blocks = generator_blocks.expand(int(layer.r[active[0]]), -1, -1)
            generator = generator_blocks.detach()
        _, W0_eval, W_eval, merge_check = next(MG.iter_pairs(
            wrapped, module_manifest, device="cuda:0", check=True, names=[name]))
        if not merge_check.passed:
            raise RuntimeError(f"Native/merged check failed for {name}: {merge_check}")
        W0 = W0_eval.float()
        if rotation is not None:
            W = apply_block_rotation_to_weight_right(rotation, W0, oft_block_size)
        else:
            W = W0 + layer.get_delta_weight(active[0]).detach().float()
        fixed_path = reference_path(Path(args.reference_root), M.BASE_MODEL, name)
        fixed_svd = load_reference(fixed_path, W0, {
            "analysis_version": ANALYSIS_VERSION, "base_model": M.BASE_MODEL,
            "matrix_name": name, "shape": list(W0.shape), "svd_dtype": "float32",
            "svd_driver": args.svd_driver,
        }, torch)
        convention = None
        if raw_rotation is not None:
            convention = verify_oft_convention(W0_eval, W_eval, raw_rotation,
                                               tolerance=args.convention_tolerance,
                                               block_size=oft_block_size)
            if not convention["verified"]:
                raise RuntimeError(f"OFT convention verification failed for {name}: {convention}")
            convention["polar_projection"] = polar_diagnostic
        result = analyze_weight_pair(
            W0, W, name=name, projection_type=info["projection_type"], depth=info["depth"],
            repeated_rtol=args.repeated_rtol, repeated_atol=args.repeated_atol,
            heatmap_components=args.heatmap_components,
            retain_pairwise=name in example_positions, driver=args.svd_driver,
            rotation=rotation, oft_generator=generator,
            oft_coordinate_block_size=oft_block_size,
            spectral_block_size=args.spectral_block_size,
            sensitivity_block_size=matched_spectral_width,
            top_block_size=args.top_block_size, base_svd=fixed_svd)
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
    wrapped.unload()
    del wrapped, transformer, peft_layers
    gc.collect()
    torch.cuda.empty_cache()
    print(f"completed {directory}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="manifest checkpoint_id; repeat for multiple checkpoints")
    parser.add_argument("--output", default="runs/singular_rotations")
    parser.add_argument("--reference-root", default="runs/singular_rotations/base_references_v2")
    parser.add_argument("--build-reference-only", action="store_true")
    parser.add_argument("--label", help="optional plot label (best with one checkpoint)")
    parser.add_argument("--budget", help="matched parameter-budget label for aggregation")
    parser.add_argument("--layers", type=lambda x: {int(v) for v in x.split(",")})
    parser.add_argument("--projection", action="append",
                        help="role or block_type:role; repeat to select several")
    parser.add_argument("--repeated-rtol", type=float, default=1e-3)
    parser.add_argument("--repeated-atol", type=float, default=0.0)
    parser.add_argument("--heatmap-components", type=int, default=128)
    parser.add_argument("--spectral-block-size", type=int, default=128,
                        help="target singular-index width; independent of OFT coordinate blocks")
    parser.add_argument("--top-block-size", type=int, default=128)
    parser.add_argument("--svd-driver", choices=("gesvd", "gesvdj"), default="gesvd")
    parser.add_argument("--convention-tolerance", type=float, default=5e-3)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch = require_gpu()
    from pipeline import manifest as M
    checkpoints = M.load()
    for checkpoint_id in args.checkpoint:
        analyze_checkpoint(M.get(checkpoint_id, checkpoints), args, torch)
    if not args.no_plots:
        render_plots(args.output)


if __name__ == "__main__":
    main()

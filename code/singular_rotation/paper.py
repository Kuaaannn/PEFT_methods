"""Paper-level aggregation for block-identifiable singular rotations.

The checkpoint is the replicate. Blocks are first combined within a matrix
using singular-dimension weights, matrices are combined within projection
families, and projection families receive equal weight. Non-identifiable
blocks contribute to coverage, never to a rotation estimate and never as zero.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


SIDES = ("u", "v")
DEPTH_LABELS = ("early", "middle", "late")
METHOD_LABELS = {"lora": "LoRA", "flashoft": "OFT"}
BUDGETS = ("small", "medium", "large")


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _model_label(model_id: str) -> str:
    return "Qwen2.5-7B" if "Qwen" in model_id else "Llama-3.1-8B"


def _mean(values):
    return sum(values) / len(values) if values else None


def _sample_sd(values):
    if len(values) < 2:
        return None
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def _weighted_rms(pairs):
    pairs = [(value, weight) for value, weight in pairs if value is not None and weight > 0]
    if not pairs:
        return None
    return math.sqrt(sum(weight * value * value for value, weight in pairs)
                     / sum(weight for _, weight in pairs))


def _endpoint_identifiable(row: dict) -> bool:
    return row.get("endpoint_identifiable", row.get("claim_eligible", False))


def _perturbation_certified(row: dict) -> bool:
    return row.get("perturbation_certified", row.get("stability_status") in (
        "exact_analytic_transport", "perturbation_certified_correspondence"))


def _hierarchical_rotation(rows, side: str) -> float | None:
    """Equal-weight projections and layers; dimension-weight blocks in a matrix."""
    matrices = defaultdict(list)
    for row in rows:
        if _endpoint_identifiable(row):
            matrices[(row["projection_type"], row["matrix_name"])].append(
                (row[side]["rms_sine"], row["size"]))
    projections = defaultdict(list)
    for (projection, _), values in matrices.items():
        value = _weighted_rms(values)
        if value is not None:
            projections[projection].append(value)
    projection_values = [_rms(values) for values in projections.values() if values]
    return _rms(projection_values)


def _hierarchical_scalar(layer_rows, field: str) -> float | None:
    projections = defaultdict(list)
    for row in layer_rows:
        projections[row["projection_type"]].append(float(row[field]))
    return _rms([_rms(values) for values in projections.values() if values])


def _coverage(rows, predicate=_endpoint_identifiable) -> float:
    projections = defaultdict(lambda: [0, 0])
    for row in rows:
        slot = projections[row["projection_type"]]
        slot[1] += row["size"]
        if predicate(row):
            slot[0] += row["size"]
    return _mean([eligible / total for eligible, total in projections.values() if total]) or 0.0


def _flatten_metadata(root: Path):
    for path in sorted(root.glob("*/metadata.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("status") != "complete" or not str(
                metadata.get("analysis_version", "")).startswith("2."):
            continue
        yield path.parent, metadata


def _load_checkpoint(directory: Path, metadata: dict, selected_metadata: dict[str, dict],
                     n_bands: int):
    layers = list(_iter_jsonl(directory / metadata["raw_files"]["layers"]))
    raw_blocks = list(_iter_jsonl(directory / metadata["raw_files"]["spectral_blocks"]))
    reporting = [row for row in raw_blocks
                 if row["block_kind"] == "spectral_reporting_band"
                 and row["resolution_role"] == "primary_paper_band"]
    blocks = reporting or [row for row in raw_blocks
                           if row["block_kind"] == "spectral_analysis_block"
                           and row["resolution_role"] == "primary_cross_method"]
    if not layers or not blocks:
        raise ValueError(f"Incomplete measurements in {directory}")
    if not reporting and {row["target_width"] for row in blocks} != {128}:
        raise ValueError(f"Primary cross-method blocks are not uniformly width 128: {directory}")
    n_singular = {row["matrix_name"]: row["n_singular"] for row in layers}
    max_depth = max(row["depth"] for row in layers)
    common = {
        "checkpoint_id": metadata["checkpoint_id"],
        "model": _model_label(metadata["base_model"]),
        "base_model": metadata["base_model"],
        "method": METHOD_LABELS.get(metadata["method"], metadata["method"]),
        "method_key": metadata["method"],
        "budget": metadata.get("budget"),
        "capacity": metadata.get("capacity"),
        "learning_rate": metadata.get("learning_rate"),
        "seed": metadata.get("seed"),
        "selected_best_lr": metadata["checkpoint_id"] in selected_metadata,
        "selected_mean_dev_accuracy": selected_metadata.get(
            metadata["checkpoint_id"], {}).get("mean_dev_accuracy"),
        "band_measurement": ("direct_broad_subspace" if reporting
                             else "legacy_128_block_fallback"),
    }
    checkpoint = {
        **common,
        "relative_update_rms": _hierarchical_scalar(layers, "relative_weight_change"),
        "endpoint_identifiable_coverage": _coverage(blocks),
        "perturbation_certified_coverage": _coverage(blocks, _perturbation_certified),
    }
    singular_shifts = [row["endpoint_svd_diagnostic"][
        "max_singular_value_shift_over_s0_max"] for row in layers]
    checkpoint["singular_value_shift_median"] = sorted(singular_shifts)[len(singular_shifts) // 2]
    for side in SIDES:
        value = _hierarchical_rotation(blocks, side)
        checkpoint[f"{side}_rotation_rms_sine"] = value
        checkpoint[f"{side}_rotation_equivalent_degrees"] = (
            math.degrees(math.asin(min(1.0, value))) if value is not None else None)
    u_value, v_value = (checkpoint["u_rotation_rms_sine"],
                        checkpoint["v_rotation_rms_sine"])
    denominator = u_value + v_value if u_value is not None and v_value is not None else None
    checkpoint["rotation_asymmetry"] = ((v_value - u_value) / denominator
                                         if denominator else None)

    module_rows = []
    for projection in sorted({row["projection_type"] for row in blocks}):
        selected = [row for row in blocks if row["projection_type"] == projection]
        entry = {**common, "projection_type": projection,
                 "endpoint_identifiable_coverage": _coverage(selected),
                 "perturbation_certified_coverage": _coverage(
                     selected, _perturbation_certified)}
        for side in SIDES:
            entry[f"{side}_rotation_rms_sine"] = _hierarchical_rotation(selected, side)
        module_rows.append(entry)

    depth_rows = []
    for depth_id, label in enumerate(DEPTH_LABELS):
        selected = [row for row in blocks
                    if min(2, 3 * row["depth"] // (max_depth + 1)) == depth_id]
        entry = {**common, "depth_group": label, "stable_coverage": _coverage(selected)}
        entry["endpoint_identifiable_coverage"] = entry.pop("stable_coverage")
        entry["perturbation_certified_coverage"] = _coverage(
            selected, _perturbation_certified)
        for side in SIDES:
            entry[f"{side}_rotation_rms_sine"] = _hierarchical_rotation(selected, side)
        depth_rows.append(entry)

    band_rows = []
    observed_band_count = (metadata.get("settings", {}).get("reporting_band_count", n_bands)
                           if reporting else n_bands)
    if reporting and observed_band_count != n_bands:
        raise ValueError(f"Requested {n_bands} bands but checkpoint stores "
                         f"{observed_band_count}: {directory}")
    for band in range(n_bands):
        if reporting:
            selected = [row for row in blocks if row["band_id"] == band]
        else:
            selected = []
            for row in blocks:
                midpoint = (row["start_index"] + row["stop_index"]) / 2
                observed = min(n_bands - 1, int(n_bands * midpoint
                                                 / n_singular[row["matrix_name"]]))
                if observed == band:
                    selected.append(row)
        entry = {**common, "band": band,
                 "normalized_band_center": (band + 0.5) / n_bands,
                 "endpoint_identifiable_coverage": _coverage(selected),
                 "perturbation_certified_coverage": _coverage(
                     selected, _perturbation_certified)}
        for side in SIDES:
            entry[f"{side}_rotation_rms_sine"] = _hierarchical_rotation(selected, side)
        band_rows.append(entry)
    return checkpoint, module_rows, depth_rows, band_rows


def _deduplicate_completed(roots):
    completed = {}
    for root in map(Path, roots):
        for directory, metadata in _flatten_metadata(root):
            checkpoint_id = metadata["checkpoint_id"]
            prior = completed.get(checkpoint_id)
            if prior and prior[1].get("analysis_library_hash") != metadata.get(
                    "analysis_library_hash"):
                raise ValueError(f"Conflicting analyses for {checkpoint_id}")
            completed[checkpoint_id] = (directory, metadata)
    return completed


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _seed_summary(rows: list[dict], group_fields: tuple[str, ...], metrics: tuple[str, ...]):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(group_fields, key))
        result["n_seeds"] = len({row["seed"] for row in values})
        for metric in metrics:
            observed = [float(row[metric]) for row in values if row.get(metric) is not None]
            result[f"{metric}_mean"] = _mean(observed)
            result[f"{metric}_sd"] = _sample_sd(observed)
            result[f"{metric}_min"] = min(observed) if observed else None
            result[f"{metric}_max"] = max(observed) if observed else None
        output.append(result)
    return output


def _set_rotation_axis(axis, values) -> None:
    """Adaptive nonnegative symlog scale that represents exact zeros honestly."""
    positive = [value for value in values if value is not None and value > 0]
    if not positive:
        axis.set_ylim(-1e-8, 1e-8)
        return
    threshold = min(positive) / 5
    axis.set_yscale("symlog", linthresh=threshold, linscale=0.7)
    axis.set_ylim(-threshold * 0.2, max(positive) * 1.3)


def _plot_bandwise(rows, path: Path, np, plt) -> None:
    selected = [row for row in rows if row["selected_best_lr"]]
    models = sorted({row["model"] for row in selected}, reverse=True)
    fig, axes = plt.subplots(len(models), 2, figsize=(10, 3.8 * len(models)),
                             squeeze=False, constrained_layout=True)
    colors = {"LoRA": "#2673b8", "OFT": "#d94841"}
    styles = {"small": ":", "medium": "--", "large": "-"}
    for row_id, model in enumerate(models):
        for col, side in enumerate(SIDES):
            axis = axes[row_id, col]
            panel_values = []
            for method in ("LoRA", "OFT"):
                for budget in BUDGETS:
                    grouped = defaultdict(list)
                    for row in selected:
                        if row["model"] == model and row["method"] == method \
                                and row["budget"] == budget:
                            value = row[f"{side}_rotation_rms_sine"]
                            if value is not None:
                                grouped[row["band"]].append(value)
                    if not grouped:
                        continue
                    total_bands = max(grouped) + 1
                    xs = [(band + 0.5) / total_bands for band in sorted(grouped)]
                    means = [_mean(grouped[band]) for band in sorted(grouped)]
                    lows = [min(grouped[band]) for band in sorted(grouped)]
                    highs = [max(grouped[band]) for band in sorted(grouped)]
                    panel_values.extend(means); panel_values.extend(lows); panel_values.extend(highs)
                    axis.plot(xs, means, color=colors[method], linestyle=styles[budget],
                              linewidth=2, label=f"{method} {budget}")
                    axis.fill_between(xs, lows, highs, color=colors[method], alpha=0.10)
            _set_rotation_axis(axis, panel_values)
            axis.set(title=f"{model}: {side.upper()} rotation",
                     xlabel="normalized singular-spectrum position",
                     ylabel="hierarchical RMS sine (adaptive symlog)", xlim=(0, 1))
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False)
    fig.suptitle("Bandwise singular-subspace rotation at designated operating points")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_full_lr(rows, path: Path, np, plt) -> None:
    models = sorted({row["model"] for row in rows}, reverse=True)
    fig, axes = plt.subplots(len(models), 2, figsize=(10, 3.8 * len(models)),
                             squeeze=False, constrained_layout=True)
    colors = {"LoRA": "#2673b8", "OFT": "#d94841"}
    markers = {"small": "o", "medium": "s", "large": "^"}
    for row_id, model in enumerate(models):
        for col, side in enumerate(SIDES):
            axis = axes[row_id, col]
            panel_values = []
            for method in ("LoRA", "OFT"):
                for budget in BUDGETS:
                    values = [row for row in rows if row["model"] == model
                              and row["method"] == method and row["budget"] == budget
                              and row["relative_update_rms"] is not None
                              and row[f"{side}_rotation_rms_sine"] is not None]
                    if not values:
                        continue
                    axis.scatter([row["relative_update_rms"] for row in values],
                                 [row[f"{side}_rotation_rms_sine"] for row in values],
                                 color=colors[method], marker=markers[budget], alpha=0.5,
                                 s=28, label=f"{method} {budget}")
                    panel_values.extend(row[f"{side}_rotation_rms_sine"] for row in values)
                    chosen = [row for row in values if row["selected_best_lr"]]
                    axis.scatter([row["relative_update_rms"] for row in chosen],
                                 [row[f"{side}_rotation_rms_sine"] for row in chosen],
                                 facecolors="none", edgecolors="black", marker=markers[budget],
                                 linewidths=0.9, s=60)
            axis.set_xscale("log")
            _set_rotation_axis(axis, panel_values)
            axis.set(title=f"{model}: {side.upper()} rotation",
                     xlabel=r"relative update $\rho$ (hierarchical RMS)",
                     ylabel="hierarchical RMS sine (adaptive symlog)")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False)
    fig.suptitle("Rotation versus realized update magnitude across learning rates")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_appendix(rows, category: str, path: Path, np, plt) -> None:
    selected = [row for row in rows if row["selected_best_lr"]]
    categories = list(dict.fromkeys(row[category] for row in selected))
    models = sorted({row["model"] for row in selected}, reverse=True)
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 3.8 * len(models)),
                             squeeze=False, constrained_layout=True)
    colors = {"LoRA": "#2673b8", "OFT": "#d94841"}
    offsets = {"LoRA": -0.08, "OFT": 0.08}
    for row_id, model in enumerate(models):
        for col, side in enumerate(SIDES):
            axis = axes[row_id, col]
            panel_values = []
            for method in ("LoRA", "OFT"):
                method_rows = [row for row in selected
                               if row["model"] == model and row["method"] == method]
                if not method_rows:
                    continue
                ys, lo, hi = [], [], []
                for value in categories:
                    observed = [row[f"{side}_rotation_rms_sine"] for row in method_rows
                                if row[category] == value
                                and row[f"{side}_rotation_rms_sine"] is not None]
                    if not observed:
                        break
                    ys.append(_mean(observed)); lo.append(min(observed)); hi.append(max(observed))
                if len(ys) != len(categories):
                    continue
                panel_values.extend(ys); panel_values.extend(lo); panel_values.extend(hi)
                xs = np.arange(len(categories)) + offsets[method]
                axis.errorbar(xs, ys, yerr=[np.asarray(ys) - np.asarray(lo),
                                            np.asarray(hi) - np.asarray(ys)],
                              marker="o", linewidth=1.7, capsize=2, color=colors[method],
                              label=method)
            _set_rotation_axis(axis, panel_values)
            axis.set_xticks(range(len(categories)), categories, rotation=25, ha="right")
            axis.set(title=f"{model}: {side.upper()} rotation",
                     ylabel="hierarchical RMS sine (adaptive symlog)")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def render_paper_report(roots, output, *, selected_ids=(), selected_metadata=None,
                        n_bands: int = 8, include_dose_response: bool = True) -> dict:
    """Create machine-readable tables and publication-oriented figures."""
    if n_bands < 2:
        raise ValueError("n_bands must be at least two")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    completed = _deduplicate_completed(roots)
    if not completed:
        raise FileNotFoundError("No completed v2 singular-rotation checkpoints")
    selected_metadata = dict(selected_metadata or {})
    for checkpoint_id in selected_ids:
        selected_metadata.setdefault(checkpoint_id, {})
    checkpoints, modules, depths, bands = [], [], [], []
    for directory, metadata in completed.values():
        checkpoint, module_rows, depth_rows, band_rows = _load_checkpoint(
            directory, metadata, selected_metadata, n_bands)
        checkpoints.append(checkpoint); modules.extend(module_rows)
        depths.extend(depth_rows); bands.extend(band_rows)
    checkpoints.sort(key=lambda row: (row["model"], row["method"], str(row["budget"]),
                                      row["learning_rate"], row["seed"]))
    selected = [row for row in checkpoints if row["selected_best_lr"]]
    metrics = ["relative_update_rms", "u_rotation_rms_sine", "v_rotation_rms_sine",
               "u_rotation_equivalent_degrees", "v_rotation_equivalent_degrees",
               "rotation_asymmetry", "endpoint_identifiable_coverage",
               "perturbation_certified_coverage", "singular_value_shift_median"]
    if any(row.get("selected_mean_dev_accuracy") is not None for row in selected):
        metrics.insert(0, "selected_mean_dev_accuracy")
    main_summary = _seed_summary(selected, ("model", "method", "budget", "capacity",
                                               "learning_rate"), metrics)
    module_summary = _seed_summary(
        [row for row in modules if row["selected_best_lr"]],
        ("model", "method", "budget", "capacity", "learning_rate", "projection_type"),
        ("u_rotation_rms_sine", "v_rotation_rms_sine",
         "endpoint_identifiable_coverage", "perturbation_certified_coverage"))
    depth_summary = _seed_summary(
        [row for row in depths if row["selected_best_lr"]],
        ("model", "method", "budget", "capacity", "learning_rate", "depth_group"),
        ("u_rotation_rms_sine", "v_rotation_rms_sine",
         "endpoint_identifiable_coverage", "perturbation_certified_coverage"))
    band_summary = _seed_summary(
        [row for row in bands if row["selected_best_lr"]],
        ("model", "method", "budget", "capacity", "learning_rate", "band",
         "normalized_band_center"),
        ("u_rotation_rms_sine", "v_rotation_rms_sine",
         "endpoint_identifiable_coverage", "perturbation_certified_coverage"))
    for name, rows in (("checkpoint_aggregate.csv", checkpoints),
                       ("main_selected_summary.csv", main_summary),
                       ("main_aggregate_summary.csv", main_summary),
                       ("bandwise_checkpoint.csv", bands),
                       ("main_bandwise_summary.csv", band_summary),
                       ("appendix_module_summary.csv", module_summary),
                       ("appendix_depth_summary.csv", depth_summary)):
        _write_csv(output / name, rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _plot_bandwise(bands, output / "main_bandwise_rotation.png", np, plt)
    if include_dose_response:
        _plot_full_lr(checkpoints, output / "main_full_lr_rotation_vs_update.png", np, plt)
    _plot_appendix(modules, "projection_type", output / "appendix_module_rotation.png", np, plt)
    _plot_appendix(depths, "depth_group", output / "appendix_depth_rotation.png", np, plt)
    manifest = {
        "schema_version": "1.0.0", "n_checkpoints": len(checkpoints),
        "n_selected_checkpoints": len(selected), "n_bands": n_bands,
        "include_dose_response": include_dose_response,
        "source_roots": [str(Path(root).resolve()) for root in roots],
        "aggregation": {
            "blocks_within_matrix": "singular-dimension-weighted RMS sine",
            "layers_within_projection": "equal-layer RMS",
            "projection_types_within_checkpoint": "equal-projection RMS",
            "replicate": "checkpoint seed",
            "nonidentifiable_bands": "excluded from rotation and retained in coverage",
            "band_measurement": "direct broad-subspace projector metric with base-gap-adjusted boundaries",
            "eligibility": "endpoint identifiability and stronger perturbation certification reported separately",
        },
    }
    temporary = output / "report_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "report_manifest.json")
    return manifest

"""Atomic v2 output and block-first plotting for singular rotations."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _jsonl(path: Path, rows) -> None:
    _atomic_text(path, "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def _open_jsonl(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def _iter_jsonl(path: Path):
    with _open_jsonl(path) as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return clean[:96] + "-" + hashlib.sha256(value.encode()).hexdigest()[:10]


def _positive_limits(values, np, *, lower_quantile=0.01, upper_quantile=0.995,
                     minimum=1e-8) -> tuple[float, float]:
    """Robust positive plot limits with explicit padding.

    Limits are data-adaptive, but callers must pass every series that shares an
    axis so visual comparisons within a panel retain a common scale.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if not finite.size:
        return minimum, minimum * 10
    low = max(minimum, float(np.quantile(finite, lower_quantile)))
    high = max(low * 1.05, float(np.quantile(finite, upper_quantile)))
    return low / 1.25, high * 1.25


def _symmetric_limit(values, np, *, quantile=0.995, minimum=1e-8) -> float:
    """Return a robust, symmetric color limit for signed measurements."""
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return minimum
    return max(minimum, float(np.quantile(finite, quantile)) * 1.05)


def _save_figure(fig, png_path: Path, *, dpi: int) -> None:
    """Write matched raster and vector versions of a plot."""
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(png_path.with_suffix(".pdf"))


def _leading_index_series(rows, side: str, np):
    """Aggregate a leading-block diagnostic at each *actual* singular index.

    Per-index angles depend on the fixed SVD gauge when singular values are
    unresolved. That affects their interpretation, not their availability:
    plotting only identifiable LoRA rows gave LoRA and OFT different x axes.
    Keep every measured row for both methods and leave claim eligibility to
    the block metrics and the identifiability fields in the raw output.
    """
    by_index = {}
    for row in rows:
        index = int(row["singular_index"])
        by_index.setdefault(index, []).append(float(row[f"{side}_rotation_deg"]))
    indices = sorted(by_index)
    rotations = [float(np.median(by_index[index])) for index in indices]
    return indices, rotations


class CheckpointOutput:
    """Stream one checkpoint and commit it only after every matrix succeeds."""

    def __init__(self, root, metadata: dict):
        self.root = Path(root)
        self.directory = self.root / _slug(metadata["checkpoint_id"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.layers, self.conventions, self.heatmaps = [], [], []
        self.n_indices = self.n_blocks = 0
        self._index_final = self.directory / "leading_indices.jsonl.gz"
        self._block_final = self.directory / "spectral_blocks.jsonl.gz"
        self._index_tmp = self._index_final.with_suffix(self._index_final.suffix + ".tmp")
        self._block_tmp = self._block_final.with_suffix(self._block_final.suffix + ".tmp")
        self._index_output = gzip.open(self._index_tmp, "wt")
        self._block_output = gzip.open(self._block_tmp, "wt")
        _atomic_text(self.directory / "metadata.json", json.dumps(
            {**metadata, "status": "running"}, indent=2, sort_keys=True,
            allow_nan=False) + "\n")

    def add(self, result: dict, convention=None) -> None:
        self.layers.append(result["layer"])
        for row in result["indices"]:
            self._index_output.write(json.dumps(row, allow_nan=False) + "\n")
            self.n_indices += 1
        for row in result["blocks"]:
            self._block_output.write(json.dumps(row, allow_nan=False) + "\n")
            self.n_blocks += 1
        if convention is not None:
            self.conventions.append({"matrix_name": result["layer"]["matrix_name"],
                                     **convention})
        if result["pairwise"] is not None:
            import numpy as np
            relative = Path("heatmaps") / f"{_slug(result['layer']['matrix_name'])}.npz"
            path = self.directory / relative
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, **{
                key: value.numpy() for key, value in result["pairwise"].items()})
            self.heatmaps.append({
                "matrix_name": result["layer"]["matrix_name"],
                "projection_type": result["layer"]["projection_type"],
                "depth": result["layer"]["depth"],
                "example_positions": result["layer"].get("example_positions", []),
                "path": str(relative),
            })

    def finish(self) -> Path:
        if not self.layers:
            raise RuntimeError("No adapted matrices were measured")
        self._index_output.close()
        self._block_output.close()
        self._index_tmp.replace(self._index_final)
        self._block_tmp.replace(self._block_final)
        _jsonl(self.directory / "layers.jsonl", self.layers)
        _jsonl(self.directory / "oft_convention.jsonl", self.conventions)
        self.metadata.update({
            "status": "complete", "n_matrices": len(self.layers),
            "n_leading_index_rows": self.n_indices,
            "n_block_rows": self.n_blocks,
            "raw_files": {
                "leading_indices": self._index_final.name,
                "spectral_blocks": self._block_final.name,
                "layers": "layers.jsonl", "oft_convention": "oft_convention.jsonl",
            },
            "heatmaps": self.heatmaps,
        })
        _atomic_text(self.directory / "metadata.json", json.dumps(
            self.metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return self.directory


def render_plots(root) -> None:
    """Render block-primary results and leading-block supplementary figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(root)
    completed = [(p, json.loads(p.read_text())) for p in sorted(root.glob("*/metadata.json"))]
    completed = [(p, m) for p, m in completed if m.get("status") == "complete"
                 and str(m.get("analysis_version", "")).startswith("2.")]
    if not completed:
        raise FileNotFoundError(f"No completed v2 measurements under {root}")
    plot_root = root / "plots_v2"
    block_root = plot_root / "block_rotation"
    leading_root = plot_root / "leading_indices"
    heatmap_root = plot_root / "heatmaps"
    for path in (block_root, leading_root, heatmap_root):
        path.mkdir(parents=True, exist_ok=True)

    block_summary, leading_summary = [], []
    for metadata_path, metadata in completed:
        directory = metadata_path.parent
        label = metadata.get("label") or metadata["checkpoint_id"]
        raw = metadata["raw_files"]
        blocks = [row for row in _iter_jsonl(directory / raw["spectral_blocks"])
                  if row["block_kind"] == "spectral_analysis_block"
                  and row["resolution_role"] == "primary_cross_method"]
        indices = list(_iter_jsonl(directory / raw["leading_indices"]))
        n_singular_by_matrix = {
            row["matrix_name"]: row["n_singular"] for row in _read_layers(directory)}

        # Block-first figures: each line is a layer; the thick line is the
        # within-checkpoint median. This is descriptive, not a seed CI.
        for projection in sorted({row["projection_type"] for row in blocks}):
            projected = [row for row in blocks if row["projection_type"] == projection]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False,
                                     constrained_layout=True)
            for axis, side in zip(axes, ("u", "v")):
                by_layer = {}
                for row in projected:
                    x = ((row["start_index"] + row["stop_index"])
                         / (2 * n_singular_by_matrix[row["matrix_name"]]))
                    by_layer.setdefault(row["depth"], []).append((x, row[side]["rms_sine"]))
                for values in by_layer.values():
                    values.sort()
                    axis.plot(*zip(*values), color="0.75", linewidth=0.5, alpha=0.5)
                # Boundaries can move slightly by layer; bin on the target block ordinal.
                ordinals = {}
                for row in projected:
                    if row["claim_eligible"]:
                        ordinals.setdefault(row["block_id"], []).append(row[side]["rms_sine"])
                if ordinals:
                    axis.plot([(i + 0.5) / max(1, len(ordinals)) for i in sorted(ordinals)],
                              [np.median(ordinals[i]) for i in sorted(ordinals)],
                              linewidth=2, label="eligible-block median")
                plotted = [value for values in by_layer.values() for _, value in values]
                if any(value > 0 for value in plotted):
                    low, high = _positive_limits(plotted, np)
                    axis.set_yscale("log")
                    axis.set_ylim(low, high)
                axis.set(title=f"{side.upper()} spectral-subspace rotation",
                         xlabel="normalized spectral rank",
                         ylabel="RMS sine of principal angles (log scale)")
            axes[1].legend(fontsize=8)
            fig.suptitle(f"{label}\n{projection}; block metrics are the primary claim")
            _save_figure(
                fig,
                block_root / f"block-{_slug(metadata['checkpoint_id']+'-'+projection)}.png",
                dpi=180,
            )
            plt.close(fig)

            for side in ("u", "v"):
                eligible = [row for row in projected if row["claim_eligible"]]
                vals = [row[side]["rms_sine"] for row in eligible]
                block_summary.append({
                    "checkpoint_id": metadata["checkpoint_id"], "method": metadata["method"],
                    "base_model": metadata.get("base_model"), "seed": metadata.get("seed"),
                    "capacity": metadata.get("capacity"), "projection_type": projection,
                    "side": side, "n_blocks": len(projected),
                    "n_claim_eligible_blocks": len(vals),
                    "claim_eligible_fraction": len(vals) / len(projected),
                    "median_rms_sine": float(np.median(vals)) if vals else None,
                    "mean_rms_sine": float(np.mean(vals)) if vals else None,
                    "stability_status_counts": {
                        status: sum(row["stability_status"] == status for row in projected)
                        for status in sorted({row["stability_status"] for row in projected})},
                })

        # Supplementary fixed-gauge curves contain the entire measured leading
        # spectral block. LoRA and OFT deliberately use the same actual-index
        # axis; identifiability filters belong to primary block-level claims,
        # not to the presentation of this diagnostic.
        for projection in sorted({row["projection_type"] for row in indices}):
            projected = [row for row in indices if row["projection_type"] == projection]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False,
                                     constrained_layout=True)
            for axis, side in zip(axes, ("u", "v")):
                xs, ys = _leading_index_series(projected, side, np)
                axis.plot(xs, ys, linewidth=1.4)
                if ys:
                    high = max(float(np.quantile(ys, 0.995)) * 1.15, 1e-4)
                    axis.set_ylim(0.0, high)
                    if len(xs) == 1:
                        axis.set_xlim(xs[0] - 0.5, xs[0] + 0.5)
                    else:
                        axis.set_xlim(xs[0], xs[-1])
                axis.set(title=f"{side.upper()} fixed-gauge leading-block rotation",
                         xlabel="actual singular index (0-based)", ylabel="angle (degrees)")
            fig.suptitle(
                f"{label}\n{projection}; per-index diagnostic, block metrics are primary")
            _save_figure(
                fig,
                leading_root / f"leading-{_slug(metadata['checkpoint_id']+'-'+projection)}.png",
                dpi=180,
            )
            plt.close(fig)
            leading_summary.append({
                "checkpoint_id": metadata["checkpoint_id"],
                "method": metadata["method"],
                "base_model": metadata.get("base_model"),
                "seed": metadata.get("seed"),
                "capacity": metadata.get("capacity"),
                "projection_type": projection,
                "n_rows": len(projected),
                "n_distinct_indices": len({int(r["singular_index"]) for r in projected}),
                "actual_index_min": min(int(r["singular_index"]) for r in projected),
                "actual_index_max": max(int(r["singular_index"]) for r in projected),
                "index_rendering": "all_rows_at_actual_zero_based_index",
                "interpretation": "fixed_gauge_supplementary",
                "identifiable_fraction": (
                    sum(r["angle_identifiable"] for r in projected) / len(projected)),
            })

        # Exact selected-layer views. Each U/V pair shares data-adaptive limits;
        # the numerical limits are shown on the color bars and are never reused
        # across unrelated checkpoints. Log energy reveals off-diagonal mass.
        for heatmap in metadata.get("heatmaps", []):
            data = np.load(directory / heatmap["path"])
            for window in ("top", "middle", "tail"):
                fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
                images = []
                energies = [data[f"{side}_{window}_energy"] for side in ("u", "v")]
                positive = np.concatenate([value[value > 0] for value in energies])
                floor = (float(np.quantile(positive, 0.01)) if positive.size
                         else np.finfo(np.float32).tiny)
                log_energies = [np.log10(np.clip(value, floor, None)) for value in energies]
                energy_low = min(float(np.quantile(value, 0.005)) for value in log_energies)
                energy_high = max(float(value.max()) for value in log_energies)
                residuals = [data[f"{side}_{window}_residual"] for side in ("u", "v")]
                residual_limit = _symmetric_limit(np.concatenate(
                    [value.ravel() for value in residuals]), np)
                for col, side in enumerate(("u", "v")):
                    images.append(axes[0, col].imshow(log_energies[col],
                        vmin=energy_low, vmax=energy_high, cmap="viridis", origin="lower",
                        interpolation="nearest"))
                    axes[0, col].set_title(f"{side.upper()} log10 squared alignment")
                    images.append(axes[1, col].imshow(data[f"{side}_{window}_residual"],
                        vmin=-residual_limit, vmax=residual_limit, cmap="coolwarm",
                        origin="lower", interpolation="nearest"))
                    axes[1, col].set_title(f"{side.upper()} signed residual from identity")
                fig.colorbar(images[0], ax=axes[0, :], label="log10 squared alignment", shrink=0.8)
                fig.colorbar(images[-1], ax=axes[1, :],
                             label=f"signed residual (limit {residual_limit:.2g})", shrink=0.8)
                position = "/".join(heatmap["example_positions"])
                fig.suptitle(f"{label}\n{heatmap['projection_type']} {position} — {window} spectrum")
                stem = metadata["checkpoint_id"] + "-" + heatmap["matrix_name"] + "-" + window
                _save_figure(
                    fig, heatmap_root / f"alignment-{_slug(stem)}.png", dpi=180
                )
                plt.close(fig)

            if "mechanism_top_normalized" in data:
                fig, axis = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
                mechanism = data["mechanism_top_normalized"]
                limit = _symmetric_limit(mechanism, np)
                image = axis.imshow(mechanism, vmin=-limit, vmax=limit,
                                    cmap="coolwarm", origin="lower", interpolation="nearest")
                kind = ("OFT skew-generator coupling" if metadata["method"].endswith("oft")
                        else "LoRA update in base singular coordinates")
                axis.set(title=kind, xlabel="base right-singular index",
                         ylabel="base left-singular index")
                fig.colorbar(image, ax=axis,
                             label=f"Frobenius-normalized coupling (limit {limit:.2g})")
                position = "/".join(heatmap["example_positions"])
                fig.suptitle(f"{label}\n{heatmap['projection_type']} {position}")
                stem = metadata["checkpoint_id"] + "-" + heatmap["matrix_name"] + "-mechanism"
                _save_figure(
                    fig, heatmap_root / f"mechanism-{_slug(stem)}.png", dpi=180
                )
                plt.close(fig)

            if "lora_first_order_u_mixing_top" in data:
                fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
                prediction_image = None
                normalized = []
                for side in ("u", "v"):
                    values = data[f"lora_first_order_{side}_mixing_top"]
                    norm = max(float(np.linalg.norm(values)), np.finfo(np.float32).tiny)
                    normalized.append(values / norm)
                limit = _symmetric_limit(np.concatenate(
                    [value.ravel() for value in normalized]), np)
                for axis, side, values in zip(axes, ("u", "v"), normalized):
                    prediction_image = axis.imshow(values, vmin=-limit, vmax=limit,
                        cmap="coolwarm", origin="lower", interpolation="nearest")
                    axis.set(title=f"predicted {side.upper()} mixing",
                             xlabel="source index", ylabel="destination index")
                fig.colorbar(prediction_image, ax=axes,
                             label=f"Frobenius-normalized coefficient (limit {limit:.2g})")
                position = "/".join(heatmap["example_positions"])
                fig.suptitle(f"{label}\n{heatmap['projection_type']} {position} — first order")
                stem = metadata["checkpoint_id"] + "-" + heatmap["matrix_name"] + "-first-order"
                _save_figure(
                    fig, heatmap_root / f"first-order-{_slug(stem)}.png", dpi=180
                )
                plt.close(fig)

            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
            transfer_image = None
            transfers = [data[f"{side}_sampled_band_transfer"] for side in ("u", "v")]
            positive = np.concatenate([value[value > 0] for value in transfers])
            floor = (float(np.quantile(positive, 0.01)) if positive.size
                     else np.finfo(np.float32).tiny)
            log_transfers = [np.log10(np.clip(value, floor, None)) for value in transfers]
            transfer_low = min(float(value.min()) for value in log_transfers)
            transfer_high = max(float(value.max()) for value in log_transfers)
            for axis, side, values in zip(axes, ("u", "v"), log_transfers):
                transfer_image = axis.imshow(values,
                    vmin=transfer_low, vmax=transfer_high, cmap="viridis", origin="lower",
                    interpolation="nearest")
                axis.set_xticks(range(3), ("top", "middle", "tail"))
                axis.set_yticks(range(3), ("top", "middle", "tail"))
                axis.set(title=f"{side.upper()} sampled-band transfer",
                         xlabel="downstream band", ylabel="base band")
            fig.colorbar(transfer_image, ax=axes, label="log10 normalized overlap energy")
            position = "/".join(heatmap["example_positions"])
            fig.suptitle(f"{label}\n{heatmap['projection_type']} {position}")
            stem = metadata["checkpoint_id"] + "-" + heatmap["matrix_name"] + "-band-transfer"
            _save_figure(
                fig, heatmap_root / f"band-transfer-{_slug(stem)}.png", dpi=180
            )
            plt.close(fig)

    _jsonl(root / "block_projection_summary.jsonl", block_summary)
    _jsonl(root / "leading_identifiability_summary.jsonl", leading_summary)


def _read_layers(directory: Path) -> list[dict]:
    return list(_iter_jsonl(directory / "layers.jsonl"))

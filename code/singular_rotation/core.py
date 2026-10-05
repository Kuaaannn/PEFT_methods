"""Stable, model-independent singular-vector rotation measurements.

The primary unit is a spectral subspace, not an individual singular vector.
Individual angles are emitted only for the leading spectral block and carry an
explicit identifiability label. For right-sided OFT, the fixed base SVD is
transported analytically, so a second SVD cannot create a spurious left rotation.
"""
from __future__ import annotations

from typing import Iterable

import torch


ANALYSIS_VERSION = "2.0.0"


def _svd(matrix: torch.Tensor, driver: str):
    kwargs = {"driver": driver} if matrix.is_cuda else {}
    return torch.linalg.svd(matrix.to(torch.float32), full_matrices=False, **kwargs)


def compute_base_svd(matrix: torch.Tensor, driver: str = "gesvd") -> dict:
    """Create a reusable, fixed-gauge base SVD artifact."""
    U, singular_values, Vh = _svd(matrix, driver)
    reconstruction = (U * singular_values.unsqueeze(0)) @ Vh
    residual = (torch.linalg.vector_norm(reconstruction - matrix.float()) /
                torch.linalg.vector_norm(matrix.float()).clamp_min(1e-30))
    return {"U": U, "singular_values": singular_values, "Vh": Vh,
            "shape": tuple(matrix.shape), "relative_reconstruction_residual": float(residual)}


def project_oft_rotation(rotation: torch.Tensor, block_size: int) -> tuple[torch.Tensor, dict]:
    """Project a block-diagonal OFT transform to its orthogonal polar factor."""
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("OFT rotation must be square")
    dimension = rotation.shape[0]
    if block_size < 1 or dimension % block_size:
        raise ValueError("OFT coordinate block size must divide the rotation dimension")
    source = rotation.float()
    blocks = torch.stack([
        source[start:start + block_size, start:start + block_size]
        for start in range(0, dimension, block_size)])
    left, _, right_h = torch.linalg.svd(blocks, full_matrices=False)
    projected_blocks = left @ right_h
    projected = torch.block_diag(*projected_blocks.unbind())
    source_norm = torch.linalg.vector_norm(source).clamp_min(1e-30)
    block_eye = torch.eye(block_size, device=source.device, dtype=torch.float32)
    defect = projected_blocks.transpose(-2, -1) @ projected_blocks - block_eye
    chordal = torch.linalg.matrix_norm(projected_blocks - block_eye, ord="fro", dim=(-2, -1)) \
        / (2.0 ** 0.5)
    diagonal_energy = blocks.square().sum()
    off_block_norm = torch.sqrt((source.square().sum() - diagonal_energy).clamp_min(0.0))
    residual = torch.sqrt((blocks - projected_blocks).square().sum() + off_block_norm.square())
    return projected, {
        "method": "independent_polar_svd_of_each_oft_coordinate_block",
        "coordinate_block_size": block_size,
        "n_coordinate_blocks": dimension // block_size,
        "relative_projection_residual": float(residual / source_norm),
        "off_block_relative_norm": float(off_block_norm / source_norm),
        "projected_orthogonality_fro": float(torch.linalg.vector_norm(defect)),
        "projected_orthogonality_max_abs": float(defect.abs().max()),
        "coordinate_block_chordal_mean": float(chordal.mean()),
        "coordinate_block_chordal_median": float(chordal.median()),
        "coordinate_block_chordal_max": float(chordal.max()),
    }


def apply_block_rotation_to_vectors(rotation: torch.Tensor, vectors: torch.Tensor,
                                    block_size: int) -> torch.Tensor:
    """Compute ``rotation @ vectors`` without a dense matrix multiplication."""
    if rotation.ndim == 3:
        blocks = rotation
        count = blocks.shape[0]
        if blocks.shape[1:] != (block_size, block_size):
            raise ValueError("batched OFT blocks have the wrong shape")
    else:
        if rotation.shape[0] % block_size or rotation.shape[0] != vectors.shape[0]:
            raise ValueError("block rotation and vector dimensions disagree")
        count = rotation.shape[0] // block_size
        blocks = torch.stack([rotation[i:i + block_size, i:i + block_size]
                              for i in range(0, rotation.shape[0], block_size)])
    if count * block_size != vectors.shape[0]:
        raise ValueError("OFT block count and vector dimensions disagree")
    shaped = vectors.reshape(count, block_size, vectors.shape[1])
    return torch.bmm(blocks, shaped).reshape_as(vectors)


def apply_block_rotation_to_weight_right(rotation: torch.Tensor, weight: torch.Tensor,
                                         block_size: int) -> torch.Tensor:
    """Compute ``weight @ rotation.T`` using OFT's coordinate blocks."""
    if rotation.shape[0] % block_size or rotation.shape[0] != weight.shape[1]:
        raise ValueError("block rotation and weight input dimensions disagree")
    count = rotation.shape[0] // block_size
    blocks = torch.stack([rotation[i:i + block_size, i:i + block_size]
                          for i in range(0, rotation.shape[0], block_size)])
    shaped = weight.reshape(weight.shape[0], count, block_size).permute(1, 0, 2)
    return torch.bmm(shaped, blocks.transpose(-2, -1)).permute(1, 0, 2).reshape_as(weight)


def _angle_degrees_abs(cosine: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.acos(cosine.abs().clamp(0.0, 1.0)))


def _angle_degrees_signed(cosine: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def _near_repeated_blocks(s0: torch.Tensor, s1: torch.Tensor, rtol: float,
                          atol: float) -> list[tuple[int, int]]:
    """Return maximal clusters whose internal edges are numerically unresolved."""
    k = s0.numel()
    if k == 0:
        return []
    scale0 = torch.maximum(s0[:-1].abs(), s0[1:].abs())
    scale1 = torch.maximum(s1[:-1].abs(), s1[1:].abs())
    joined = ((s0[:-1] - s0[1:]).abs() <= atol + rtol * scale0) | (
        (s1[:-1] - s1[1:]).abs() <= atol + rtol * scale1)
    split_after = torch.nonzero(~joined, as_tuple=False).flatten().add(1).cpu().tolist()
    starts = [0, *split_after, k]
    return list(zip(starts[:-1], starts[1:]))


def _relative_gaps(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return values.new_empty(0)
    scale = torch.maximum(values[:-1].abs(), values[1:].abs()).clamp_min(
        torch.finfo(values.dtype).tiny)
    return (values[:-1] - values[1:]).abs() / scale


def _spectral_blocks(values: torch.Tensor, target_width: int,
                     search_fraction: float = 0.25) -> list[tuple[int, int]]:
    """Make fixed-resolution blocks, moving boundaries to nearby large gaps.

    ``target_width`` is in singular-index space. It has no mathematical
    correspondence to an OFT feature-coordinate block, even when the two
    integers happen to match.
    """
    k = values.numel()
    if target_width < 1:
        raise ValueError("spectral_block_size must be positive")
    if k <= target_width:
        return [(0, k)]
    gaps = _relative_gaps(values)
    radius = max(1, int(round(target_width * search_fraction)))
    boundaries = [0]
    nominal = target_width
    while nominal < k:
        low = max(boundaries[-1] + 1, nominal - radius)
        high = min(k - 1, nominal + radius)
        if low > high:
            break
        local = gaps[low - 1:high]
        boundary = low + int(torch.argmax(local).item())
        boundaries.append(boundary)
        nominal = max(nominal + target_width, boundary + target_width)
    boundaries.append(k)
    return [(start, stop) for start, stop in zip(boundaries[:-1], boundaries[1:])
            if stop > start]


def _reporting_bands(values: torch.Tensor, count: int,
                     search_fraction: float = 0.2) -> list[tuple[int, int]]:
    """Create exactly ``count`` broad bands with boundaries pinned to base gaps."""
    k = values.numel()
    if count < 1 or count > k:
        raise ValueError("reporting band count must be between one and the spectrum size")
    gaps = _relative_gaps(values)
    nominal_width = k / count
    radius = max(1, int(round(nominal_width * search_fraction)))
    boundaries = [0]
    for band in range(1, count):
        nominal = int(round(band * nominal_width))
        low = max(boundaries[-1] + 1, nominal - radius)
        high = min(k - (count - band), nominal + radius)
        local = gaps[low - 1:high]
        boundaries.append(low + int(torch.argmax(local).item()))
    boundaries.append(k)
    return list(zip(boundaries[:-1], boundaries[1:]))


def _orthonormal_basis(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(matrix, mode="reduced").Q


def _principal_block(reference: torch.Tensor, trained: torch.Tensor) -> dict:
    reference = _orthonormal_basis(reference)
    trained = _orthonormal_basis(trained)
    cosines = torch.linalg.svdvals(reference.T @ trained).clamp(0.0, 1.0)
    angles = _angle_degrees_abs(cosines)
    sine2 = (1.0 - cosines.square()).clamp_min(0.0)
    return {
        "principal_cosines": [float(x) for x in cosines.cpu()],
        "principal_angles_deg": [float(x) for x in angles.cpu()],
        "mean_angle_deg": float(angles.mean()),
        "max_angle_deg": float(angles.max()),
        "rms_sine": float(torch.sqrt(sine2.mean())),
        "projector_distance": float(torch.sqrt(sine2.sum())),
        "normalized_retention": float(cosines.square().mean()),
    }


def _compact_subspace(reference: torch.Tensor, trained: torch.Tensor) -> dict:
    """RMS principal-angle statistic without materializing angle vectors.

    The columns originate from SVDs or an orthogonal analytic transport and are
    already orthonormal. This form is exact for RMS sine and avoids a large SVD
    and large JSON arrays for broad reporting bands.
    """
    overlap = reference.T @ trained
    size = reference.shape[1]
    retention = (overlap.square().sum() / size).clamp(0.0, 1.0)
    sine2 = (1.0 - retention).clamp_min(0.0)
    return {
        "rms_sine": float(torch.sqrt(sine2)),
        "projector_distance": float(torch.sqrt(size * sine2)),
        "normalized_retention": float(retention),
    }


def _boundary_gap(values: torch.Tensor, start: int, stop: int) -> tuple[float | None,
                                                                        float | None]:
    gaps = _relative_gaps(values)
    left = float(gaps[start - 1]) if start > 0 else None
    right = float(gaps[stop - 1]) if stop < values.numel() else None
    return left, right


def _boundary_absolute_gap(values: torch.Tensor, start: int, stop: int) -> float | None:
    gaps = []
    if start > 0:
        gaps.append(float((values[start - 1] - values[start]).abs()))
    if stop < values.numel():
        gaps.append(float((values[stop - 1] - values[stop]).abs()))
    return min(gaps) if gaps else None


def _local_absolute_gap(values: torch.Tensor, index: int) -> float | None:
    gaps = []
    if index > 0:
        gaps.append(float((values[index - 1] - values[index]).abs()))
    if index + 1 < values.numel():
        gaps.append(float((values[index] - values[index + 1]).abs()))
    return min(gaps) if gaps else None


def _spectral_norm_estimate(matrix: torch.Tensor, iterations: int = 16) -> float:
    """Deterministic power estimate; the Frobenius upper bound is also stored."""
    if not bool(torch.count_nonzero(matrix)):
        return 0.0
    n = matrix.shape[1]
    x = torch.sin(torch.arange(1, n + 1, device=matrix.device, dtype=torch.float32))
    x = x / torch.linalg.vector_norm(x)
    work = matrix.float()
    for _ in range(iterations):
        y = work @ x
        ynorm = torch.linalg.vector_norm(y)
        if float(ynorm) == 0.0:
            return 0.0
        x = work.T @ (y / ynorm)
        x = x / torch.linalg.vector_norm(x).clamp_min(torch.finfo(x.dtype).tiny)
    return float(torch.linalg.vector_norm(work @ x))


def choose_example_positions(entries: Iterable[dict]) -> dict[str, list[str]]:
    """Map deterministic example matrices to early/middle/late labels."""
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        grouped.setdefault(entry["projection_type"], []).append(entry)
    selected: dict[str, list[str]] = {}
    for values in grouped.values():
        ordered = sorted(values, key=lambda x: (x["depth"], x["name"]))
        positions = (("early", 0), ("middle", (len(ordered) - 1) // 2),
                     ("late", len(ordered) - 1))
        for label, index in positions:
            selected.setdefault(ordered[index]["name"], []).append(label)
    return selected


def choose_examples(entries: Iterable[dict]) -> set[str]:
    return set(choose_example_positions(entries))


def _window_slices(k: int, width: int) -> dict[str, tuple[int, int]]:
    width = min(width, k)
    middle_start = max(0, (k - width) // 2)
    return {"top": (0, width), "middle": (middle_start, middle_start + width),
            "tail": (k - width, k)}


def _pairwise_windows(U0, V0, U, V, width: int, mechanism=None) -> dict:
    result = {}
    windows = _window_slices(U0.shape[1], width)
    for label, (start, stop) in windows.items():
        u = U0[:, start:stop].T @ U[:, start:stop]
        v = V0[:, start:stop].T @ V[:, start:stop]
        result[f"u_{label}_signed"] = u.cpu()
        result[f"v_{label}_signed"] = v.cpu()
        result[f"u_{label}_energy"] = u.square().cpu()
        result[f"v_{label}_energy"] = v.square().cpu()
        result[f"u_{label}_residual"] = (u - torch.eye(
            u.shape[0], device=u.device, dtype=u.dtype)).cpu()
        result[f"v_{label}_residual"] = (v - torch.eye(
            v.shape[0], device=v.device, dtype=v.dtype)).cpu()
        result[f"{label}_start_index"] = torch.tensor(start)
    for side, reference, trained in (("u", U0, U), ("v", V0, V)):
        transfer = torch.empty((3, 3), device=reference.device, dtype=torch.float32)
        for row, (_, (a0, a1)) in enumerate(windows.items()):
            for col, (_, (b0, b1)) in enumerate(windows.items()):
                overlap = reference[:, a0:a1].T @ trained[:, b0:b1]
                transfer[row, col] = overlap.square().sum() / max(1, b1 - b0)
        result[f"{side}_sampled_band_transfer"] = transfer.cpu()
    if mechanism is not None:
        p = min(width, U0.shape[1])
        matrix = mechanism(U0[:, :p], V0[:, :p])
        norm = torch.linalg.vector_norm(matrix).clamp_min(torch.finfo(matrix.dtype).tiny)
        result["mechanism_top"] = matrix.cpu()
        result["mechanism_top_normalized"] = (matrix / norm).cpu()
        result["mechanism_top_frobenius_norm"] = norm.cpu()
    return result


def analyze_weight_pair(W0: torch.Tensor, W: torch.Tensor, *, name: str,
                        projection_type: str, depth: int, repeated_rtol: float = 1e-3,
                        repeated_atol: float = 0.0, heatmap_components: int = 128,
                        retain_pairwise: bool = False, driver: str = "gesvd",
                        rotation: torch.Tensor | None = None,
                        oft_generator: torch.Tensor | None = None,
                        oft_coordinate_block_size: int | None = None,
                        spectral_block_size: int = 128,
                        sensitivity_block_size: int | None = None,
                        reporting_band_count: int = 8,
                        top_block_size: int = 128,
                        base_svd: dict | None = None,
                        right_basis_transport=None, transport_method: str = "oft") -> dict:
    """Analyze a merged matrix relative to a fixed base SVD.

    With ``rotation``, primary OFT measurements transport the fixed right basis
    as ``rotation @ V0``. The SVD of ``W`` is only a realization diagnostic.
    ``right_basis_transport(V0)`` supplies the same one-sided construction for
    compact transforms such as HRA, without materializing a dense rotation.
    """
    if W0.ndim != 2 or W.shape != W0.shape:
        raise ValueError(f"{name}: expected equal two-dimensional weight shapes")
    if repeated_rtol < 0 or repeated_atol < 0:
        raise ValueError("Repeated-value tolerances must be nonnegative")
    if heatmap_components < 1 or top_block_size < 1:
        raise ValueError("component counts must be positive")
    if not bool(torch.isfinite(W0).all()) or not bool(torch.isfinite(W).all()):
        raise ValueError(f"{name}: weights must be finite")

    if base_svd is None:
        base_svd = compute_base_svd(W0, driver)
    if tuple(base_svd["shape"]) != tuple(W0.shape):
        raise ValueError(f"{name}: fixed base SVD shape does not match the weight")
    U0 = base_svd["U"].to(device=W0.device, dtype=torch.float32)
    s0 = base_svd["singular_values"].to(device=W0.device, dtype=torch.float32)
    Vh0 = base_svd["Vh"].to(device=W0.device, dtype=torch.float32)
    U_endpoint, s_endpoint, Vh_endpoint = _svd(W, driver)
    V0, V_endpoint = Vh0.T, Vh_endpoint.T
    delta = W.float() - W0.float()
    update_spectral_estimate = _spectral_norm_estimate(delta)
    update_fro = float(torch.linalg.vector_norm(delta))

    if rotation is not None and right_basis_transport is not None:
        raise ValueError("Specify either an OFT rotation or a compact right-basis transport")
    # Historical name retained to leave the OFT numerical branch unchanged.
    is_oft = rotation is not None or right_basis_transport is not None
    if is_oft:
        if right_basis_transport is not None:
            primary_V = right_basis_transport(V0)
            if primary_V.shape != V0.shape or not bool(torch.isfinite(primary_V).all()):
                raise ValueError(f"{name}: invalid transported right basis")
        elif rotation.ndim != 2 or rotation.shape != (W0.shape[1], W0.shape[1]):
            raise ValueError(f"{name}: OFT rotation does not act on the input dimension")
        elif oft_coordinate_block_size is None:
            primary_V = rotation.float() @ V0
        else:
            primary_V = apply_block_rotation_to_vectors(
                rotation.float(), V0, oft_coordinate_block_size)
        primary_U = U0
        primary_kind = f"fixed_base_svd_analytically_transported_by_{transport_method}"
    else:
        primary_U, primary_V = U_endpoint, V_endpoint
        primary_kind = "fixed_base_svd_to_aligned_endpoint_svd"

    u_diag_raw = torch.sum(U0 * primary_U, dim=0)
    v_diag_raw = torch.sum(V0 * primary_V, dim=0)
    if is_oft:
        signs = torch.ones_like(u_diag_raw)
    else:
        signs = torch.sign(u_diag_raw + v_diag_raw)
        fallback = torch.where(u_diag_raw.abs() >= v_diag_raw.abs(),
                               torch.sign(u_diag_raw), torch.sign(v_diag_raw))
        signs = torch.where(signs == 0, fallback, signs)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    primary_U, primary_V = primary_U * signs, primary_V * signs
    u_signed = torch.sum(U0 * primary_U, dim=0)
    v_signed = torch.sum(V0 * primary_V, dim=0)
    if is_oft:
        u_angles = _angle_degrees_signed(u_signed)
        v_angles = _angle_degrees_signed(v_signed / torch.linalg.vector_norm(
            primary_V, dim=0).clamp_min(torch.finfo(torch.float32).tiny))
    else:
        u_angles, v_angles = _angle_degrees_abs(u_signed), _angle_degrees_abs(v_signed)

    repeated = _near_repeated_blocks(s0, s_endpoint, repeated_rtol, repeated_atol)
    membership = {index: (-1, 1) for index in range(s0.numel())}
    block_rows = []
    for start, stop in repeated:
        if stop - start == 1:
            continue
        block_id = len(block_rows)
        for index in range(start, stop):
            membership[index] = (block_id, stop - start)
        block_rows.append({
            "matrix_name": name, "projection_type": projection_type, "depth": depth,
            "block_kind": "numerically_unresolved_cluster", "block_id": block_id,
            "start_index": start, "stop_index": stop, "size": stop - start,
            # This row defines identifiability only. Primary subspace metrics are
            # calculated on bounded spectral-analysis blocks below, avoiding a
            # second very large SVD when most of a dense tail is unresolved.
            "u": None, "v": None,
        })

    primary_spectral = _spectral_blocks(s0, spectral_block_size)
    resolutions = [(spectral_block_size, "primary_cross_method")]
    if sensitivity_block_size is not None and sensitivity_block_size != spectral_block_size:
        resolutions.append((sensitivity_block_size, "matched_oft_width_sensitivity"))
    for resolution, resolution_role in resolutions:
      spectral = _spectral_blocks(s0, resolution)
      for spectral_id, (start, stop) in enumerate(spectral):
        left_gap, right_gap = _boundary_gap(s0, start, stop)
        trained_left_gap, trained_right_gap = _boundary_gap(s_endpoint, start, stop)
        boundary = min(x for x in (left_gap, right_gap) if x is not None) \
            if left_gap is not None or right_gap is not None else None
        trained_boundary = min(x for x in (trained_left_gap, trained_right_gap)
                               if x is not None) \
            if trained_left_gap is not None or trained_right_gap is not None else None
        absolute_boundary = _boundary_absolute_gap(s0, start, stop)
        stability_ratio = (absolute_boundary / update_spectral_estimate
                           if absolute_boundary is not None
                           and update_spectral_estimate > 0 else None)
        if is_oft:
            stability_status = "exact_analytic_transport"
        elif stability_ratio is not None and stability_ratio >= 2.0:
            stability_status = "perturbation_certified_correspondence"
        elif ((boundary is None or boundary > repeated_rtol)
              and (trained_boundary is None or trained_boundary > repeated_rtol)):
            stability_status = "endpoint_subspaces_resolved_path_not_certified"
        else:
            stability_status = "numerically_unresolved_boundary"
        block_rows.append({
            "matrix_name": name, "projection_type": projection_type, "depth": depth,
            "block_kind": "spectral_analysis_block", "block_id": spectral_id,
            "start_index": start, "stop_index": stop, "size": stop - start,
            "target_width": resolution, "resolution_role": resolution_role,
            "oft_coordinate_block_size": oft_coordinate_block_size,
            "same_numeric_width_as_oft": (
                oft_coordinate_block_size is not None
                and resolution == oft_coordinate_block_size),
            "left_boundary_relative_gap": left_gap,
            "right_boundary_relative_gap": right_gap,
            "trained_left_boundary_relative_gap": trained_left_gap,
            "trained_right_boundary_relative_gap": trained_right_gap,
            "external_gap_over_update_spectral_estimate": stability_ratio,
            "stability_status": stability_status,
            "claim_eligible": stability_status != "numerically_unresolved_boundary",
            "u": _principal_block(U0[:, start:stop], primary_U[:, start:stop]),
            "v": _principal_block(V0[:, start:stop], primary_V[:, start:stop]),
            "base_spectral_energy_fraction": float(
                s0[start:stop].square().sum() / s0.square().sum()),
        })

    # Paper-facing bands are measured directly as broad subspaces. Combining
    # narrow-block rotations would wrongly count within-band transfer as
    # rotation out of the broad band.
    actual_reporting_band_count = min(reporting_band_count, s0.numel())
    for band_id, (start, stop) in enumerate(_reporting_bands(
            s0, actual_reporting_band_count)):
        left_gap, right_gap = _boundary_gap(s0, start, stop)
        trained_left_gap, trained_right_gap = _boundary_gap(s_endpoint, start, stop)
        boundary = min(x for x in (left_gap, right_gap) if x is not None) \
            if left_gap is not None or right_gap is not None else None
        trained_boundary = min(x for x in (trained_left_gap, trained_right_gap)
                               if x is not None) \
            if trained_left_gap is not None or trained_right_gap is not None else None
        absolute_boundary = _boundary_absolute_gap(s0, start, stop)
        stability_ratio = (absolute_boundary / update_spectral_estimate
                           if absolute_boundary is not None
                           and update_spectral_estimate > 0 else None)
        if is_oft:
            status = "exact_analytic_transport"
        elif stability_ratio is not None and stability_ratio >= 2.0:
            status = "perturbation_certified_correspondence"
        elif ((boundary is None or boundary > repeated_rtol)
              and (trained_boundary is None or trained_boundary > repeated_rtol)):
            status = "endpoint_subspaces_resolved_path_not_certified"
        else:
            status = "numerically_unresolved_boundary"
        block_rows.append({
            "matrix_name": name, "projection_type": projection_type, "depth": depth,
            "block_kind": "spectral_reporting_band", "band_id": band_id,
            "start_index": start, "stop_index": stop, "size": stop - start,
            "target_band_count": actual_reporting_band_count,
            "resolution_role": "primary_paper_band",
            "left_boundary_relative_gap": left_gap,
            "right_boundary_relative_gap": right_gap,
            "trained_left_boundary_relative_gap": trained_left_gap,
            "trained_right_boundary_relative_gap": trained_right_gap,
            "external_gap_over_update_spectral_estimate": stability_ratio,
            "stability_status": status,
            "endpoint_identifiable": status != "numerically_unresolved_boundary",
            "perturbation_certified": status in (
                "exact_analytic_transport", "perturbation_certified_correspondence"),
            "u": _compact_subspace(U0[:, start:stop], primary_U[:, start:stop]),
            "v": _compact_subspace(V0[:, start:stop], primary_V[:, start:stop]),
            "base_spectral_energy_fraction": float(
                s0[start:stop].square().sum() / s0.square().sum()),
        })

    top_stop = min(primary_spectral[0][1], top_block_size)
    scale = max(float(s0[0]), torch.finfo(torch.float32).tiny)
    index_rows = []
    total_energy = s0.square().sum()
    for i in range(top_stop):
        block_id, block_size = membership[i]
        local_gap = _local_absolute_gap(s0, i)
        gap_ratio = (local_gap / update_spectral_estimate
                     if local_gap is not None and update_spectral_estimate > 0 else None)
        if is_oft:
            status = "pinned_gauge" if block_size > 1 else "analytic_transport"
            identifiable = block_size == 1
        else:
            if block_size > 1:
                status, identifiable = "repeated_block", False
            elif update_spectral_estimate == 0 or (gap_ratio is not None and gap_ratio >= 2.0):
                status, identifiable = "perturbation_certified", True
            else:
                status, identifiable = "endpoint_only_not_correspondence_certified", False
        index_rows.append({
            "matrix_name": name, "projection_type": projection_type, "depth": depth,
            "singular_index": i, "base_singular_value": float(s0[i]),
            "trained_singular_value": float(s_endpoint[i]),
            "singular_value_shift_over_s0_max": float((s_endpoint[i] - s0[i]) / scale),
            "joint_sign_flip": int(signs[i]),
            "u_cosine_signed_aligned": float(u_signed[i]),
            "v_cosine_signed_aligned": float(v_signed[i]),
            "u_cosine_abs": float(u_signed[i].abs()),
            "v_cosine_abs": float(v_signed[i].abs()),
            "u_rotation_deg": float(u_angles[i]), "v_rotation_deg": float(v_angles[i]),
            "block_id": block_id, "block_size": block_size,
            "angle_identifiable": identifiable,
            "identifiability_status": status,
            "local_gap_over_update_spectral_estimate": gap_ratio,
            "analysis_scope": "leading_spectral_block",
            "base_energy_fraction": float(s0[i].square() / total_energy),
            "weighted_u_sine2": float(s0[i].square() * torch.sin(
                torch.deg2rad(u_angles[i])).square()),
            "weighted_v_sine2": float(s0[i].square() * torch.sin(
                torch.deg2rad(v_angles[i])).square()),
        })

    endpoint_diagnostic = {
        "mean_u_diagonal_abs": float(torch.sum(U0 * U_endpoint, dim=0).abs().mean()),
        "mean_v_diagonal_abs": float(torch.sum(V0 * V_endpoint, dim=0).abs().mean()),
        "max_singular_value_shift_over_s0_max": float(
            ((s_endpoint - s0).abs() / scale).max()),
    }
    pairwise = None
    if retain_pairwise:
        mechanism = None
        if is_oft and oft_generator is not None:
            mechanism = lambda _u, v: v.T @ apply_block_rotation_to_vectors(
                oft_generator.float(), v, oft_coordinate_block_size)
        elif not is_oft:
            mechanism = lambda u, v: u.T @ (delta @ v)
        pairwise = _pairwise_windows(U0, V0, primary_U, primary_V,
                                     heatmap_components, mechanism)
        pairwise["base_singular_values"] = s0.cpu()
        pairwise["trained_singular_values"] = s_endpoint.cpu()
        if not is_oft and "mechanism_top" in pairwise:
            coupling = pairwise["mechanism_top"]
            p = coupling.shape[0]
            sigma = s0[:p].cpu()
            sigma_i = sigma.unsqueeze(0)
            sigma_j = sigma.unsqueeze(1)
            denominator = sigma_i.square() - sigma_j.square()
            scale2 = sigma[0].square().clamp_min(torch.finfo(sigma.dtype).tiny)
            valid = denominator.abs() > repeated_rtol * scale2
            valid.fill_diagonal_(False)
            left_prediction = torch.zeros_like(coupling)
            right_prediction = torch.zeros_like(coupling)
            left_numerator = sigma_i * coupling + sigma_j * coupling.T
            right_numerator = sigma_j * coupling + sigma_i * coupling.T
            left_prediction[valid] = left_numerator[valid] / denominator[valid]
            right_prediction[valid] = right_numerator[valid] / denominator[valid]
            pairwise["lora_first_order_u_mixing_top"] = left_prediction
            pairwise["lora_first_order_v_mixing_top"] = right_prediction
            pairwise["lora_first_order_valid_top"] = valid

    return {
        "layer": {
            "matrix_name": name, "projection_type": projection_type, "depth": depth,
            "m": W0.shape[0], "n": W0.shape[1], "n_singular": s0.numel(),
            "primary_analysis": primary_kind,
            "spectral_block_size": spectral_block_size,
            "sensitivity_block_size": sensitivity_block_size,
            "n_primary_spectral_blocks": len(primary_spectral),
            "n_reporting_bands": actual_reporting_band_count,
            "top_block_stop_index": top_stop,
            "n_nonidentifiable_top_indices": sum(
                not row["angle_identifiable"] for row in index_rows),
            "update_spectral_norm_estimate": update_spectral_estimate,
            "update_frobenius_norm_upper_bound": update_fro,
            "relative_weight_change": float(
                torch.linalg.vector_norm(delta) /
                torch.linalg.vector_norm(W0.float()).clamp_min(1e-30)),
            "endpoint_svd_diagnostic": endpoint_diagnostic,
            "oft_coordinate_block_size": oft_coordinate_block_size,
        },
        "indices": index_rows, "blocks": block_rows, "pairwise": pairwise,
    }


def verify_oft_convention(W0: torch.Tensor, W: torch.Tensor, rotation: torch.Tensor,
                          expected: str = "right_transpose", tolerance: float = 5e-3,
                          block_size: int | None = None) -> dict:
    """Verify PEFT's realized merge convention and rotation orthogonality."""
    R = rotation
    candidates = {}
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("OFT rotation must be square")
    if R.shape[0] == W0.shape[1]:
        candidates["right_transpose"] = (R @ W0.to(R.dtype).T).T.to(W.dtype)
        candidates["right"] = (R.T @ W0.to(R.dtype).T).T.to(W.dtype)
    if R.shape[0] == W0.shape[0]:
        candidates["left"] = (R @ W0.to(R.dtype)).to(W.dtype)
        candidates["left_transpose"] = (R.T @ W0.to(R.dtype)).to(W.dtype)
    if expected not in candidates:
        raise ValueError(f"Expected OFT convention {expected!r} is dimensionally invalid")
    update_norm = torch.linalg.vector_norm((W - W0).float()).clamp_min(1e-30)
    weight_norm = torch.linalg.vector_norm(W.float()).clamp_min(1e-30)
    residuals = {}
    for label, candidate in candidates.items():
        error = torch.linalg.vector_norm((candidate - W).float())
        residuals[label] = {"relative_to_weight": float(error / weight_norm),
                            "relative_to_update": float(error / update_norm)}
    best = min(residuals, key=lambda key: residuals[key]["relative_to_weight"])
    if block_size is not None:
        if R.shape[0] % block_size:
            raise ValueError("OFT coordinate block size must divide rotation dimension")
        blocks = torch.stack([R.float()[i:i + block_size, i:i + block_size]
                              for i in range(0, R.shape[0], block_size)])
        eye = torch.eye(block_size, device=R.device, dtype=torch.float32)
        defect = blocks.transpose(-2, -1) @ blocks - eye
        diagonal_energy = blocks.square().sum()
        off_block_relative = float(torch.sqrt(
            (R.float().square().sum() - diagonal_energy).clamp_min(0.0)) /
            torch.linalg.vector_norm(R.float()).clamp_min(1e-30))
    else:
        eye = torch.eye(R.shape[0], device=R.device, dtype=torch.float32)
        defect = R.float().T @ R.float() - eye
        off_block_relative = None
    return {
        "expected_convention": expected, "best_convention": best,
        "verified": best == expected and residuals[expected]["relative_to_weight"] <= tolerance,
        "tolerance": tolerance, "residuals": residuals,
        "rotation_orthogonality_fro": float(torch.linalg.vector_norm(defect)),
        "rotation_orthogonality_max_abs": float(defect.abs().max()),
        "rotation_off_block_relative_norm": off_block_relative,
        "formula": "row-vector forward xR; merged W*=W0 R^T",
    }

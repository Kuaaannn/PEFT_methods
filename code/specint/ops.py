"""Canonical matrix operations (PROTOCOL.md section 3).

Orientation is ``y = W h`` with ``W`` of shape (m, n). The unit is the whole adapted weight
matrix; a fused projection is one matrix. Nothing here knows what a layer, a model or a
checkpoint is.

Every operator returns an :class:`Edit`, never a bare tensor, because a requested edit can be
*infeasible* - the protocol forbids silent clipping, absolute-value folding, and log-ratio
extrapolation through a null tail. An infeasible request is recorded with a reason, and the
caller reduces the common range for the paired comparison rather than quietly repairing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# --------------------------------------------------------------------------------------
# factorization
# --------------------------------------------------------------------------------------

FACTOR_DTYPE = torch.float32
REDUCE_DTYPE = torch.float64


def _fro2(x: torch.Tensor) -> float:
    """Squared Frobenius norm accumulated in float64 (contract: reduction_dtype)."""
    return float((x.to(REDUCE_DTYPE) ** 2).sum())


def _fro(x: torch.Tensor) -> float:
    return _fro2(x) ** 0.5


@dataclass
class Factors:
    """The cached canonical pair and thin SVD for one matrix.

    ``W_star = U diag(s_star) Vh``, ``s0 = svdvals(W0)``, ``Delta = W_star - W0``.
    Singular values are descending. Spectral edits hold ``U`` and ``Vh`` fixed and pair the two
    spectra by sorted endpoint index; this does not assert that singular vectors keep their
    identities during training, and edited coefficients are never re-sorted independently of
    their vectors.
    """

    name: str
    W0: torch.Tensor
    W_star: torch.Tensor
    U: torch.Tensor
    s_star: torch.Tensor
    Vh: torch.Tensor
    s0: torch.Tensor
    # One immutable base SVD supplies both the spectrum and causal orientation gauge.
    # s0_basis is retained as a descriptive alias for the orientation operators.
    U0: Optional[torch.Tensor] = None
    s0_basis: Optional[torch.Tensor] = None
    Vh0: Optional[torch.Tensor] = None
    solver: str = "gesvd"
    rebuild_floor: Optional[float] = None      # ||rebuild(U,s_star,Vh) - W_star||_F / ||W0||_F
    base_rebuild_floor: Optional[float] = None # ||rebuild(U0,s0_basis,Vh0) - W0||_F / ||W0||_F
    degenerate: bool = False                   # near-repeated singular values in s_star
    degenerate_replacement_spread: float = 0.0 # max spread of s0 within a tied group of s_star
    _cache: dict = field(default_factory=dict, repr=False)

    # -- derived quantities, computed once -------------------------------------------------
    @property
    def Delta(self) -> torch.Tensor:
        if "Delta" not in self._cache:
            self._cache["Delta"] = self.W_star - self.W0
        return self._cache["Delta"]

    @property
    def W_R(self) -> torch.Tensor:
        """W_R = U diag(s0) Vh - the trained directions carrying the pretrained spectrum."""
        if "W_R" not in self._cache:
            self._cache["W_R"] = rebuild(self.U, self.s0, self.Vh)
        return self._cache["W_R"]

    @property
    def E_R(self) -> torch.Tensor:
        """E_R = W_R - W_star."""
        if "E_R" not in self._cache:
            self._cache["E_R"] = self.W_R - self.W_star
        return self._cache["E_R"]

    @property
    def delta_norm(self) -> float:
        return self._num("delta_norm", lambda: _fro(self.Delta))

    @property
    def W0_norm(self) -> float:
        return self._num("W0_norm", lambda: _fro(self.W0))

    @property
    def restoration_edit_norm(self) -> float:
        return self._num("restoration_edit_norm", lambda: _fro(self.E_R))

    @property
    def restoration_base_distance(self) -> float:
        return self._num("restoration_base_distance", lambda: _fro(self.W_R - self.W0))

    def scalar_cache(self) -> dict:
        """Immutable scalars survive streaming; derived full matrices do not."""
        return {name: getattr(self, name) for name in (
            "W0_norm", "delta_norm", "restoration_edit_norm", "restoration_base_distance")}

    def _num(self, k, fn):
        if k not in self._cache:
            self._cache[k] = fn()
        return self._cache[k]

    def to(self, device, non_blocking: bool = False) -> "Factors":
        """A copy of these factors on ``device``, with derived tensors dropped.

        The factors for a large network do not fit beside the model on one accelerator: for this
        backbone the widest matrix alone carries ~1.7 GB of U, Vh, Delta, W_R and E_R. Callers
        therefore keep factors in host memory and stream one matrix at a time. Cached derived
        tensors are not carried across, because recomputing them on the target device is cheaper
        than copying them and keeps the two copies from disagreeing.
        """
        if self.W0.device == torch.device(device):
            return self
        moved = Factors(
            name=self.name,
            W0=self.W0.to(device, non_blocking=non_blocking),
            W_star=self.W_star.to(device, non_blocking=non_blocking),
            U=self.U.to(device, non_blocking=non_blocking),
            s_star=self.s_star.to(device, non_blocking=non_blocking),
            Vh=self.Vh.to(device, non_blocking=non_blocking),
            s0=self.s0.to(device, non_blocking=non_blocking),
            U0=(self.U0.to(device, non_blocking=non_blocking)
                if self.U0 is not None else None),
            s0_basis=(self.s0_basis.to(device, non_blocking=non_blocking)
                      if self.s0_basis is not None else None),
            Vh0=(self.Vh0.to(device, non_blocking=non_blocking)
                 if self.Vh0 is not None else None),
            solver=self.solver,
            rebuild_floor=self.rebuild_floor,
            base_rebuild_floor=self.base_rebuild_floor,
            degenerate=self.degenerate,
            degenerate_replacement_spread=self.degenerate_replacement_spread,
        )
        # Scalars are device-independent and expensive to recompute; carry them over.
        for k in ("delta_norm", "W0_norm", "restoration_edit_norm", "restoration_base_distance"):
            if k in self._cache:
                moved._cache[k] = self._cache[k]
        return moved


def rebuild(U: torch.Tensor, s: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
    """U diag(s) Vh, without materializing diag(s)."""
    return (U * s.unsqueeze(0)) @ Vh


def singular_values(W: torch.Tensor, driver: str = "gesvd") -> torch.Tensor:
    """Use the same CUDA solver for base and realized spectra as for factors.

    The implicit CUDA Jacobi driver can introduce drift larger than the spectral
    intervention on full model matrices. Never silently mix the two solvers.
    """
    return torch.linalg.svdvals(W, **({"driver": driver} if W.is_cuda else {}))


def factor(W0: torch.Tensor, W_star: torch.Tensor, name: str = "",
           driver: str = "gesvd", degenerate_rtol: float = 1e-6) -> Factors:
    """Thin SVD of the trained matrix plus the base spectrum, in FP32 or higher.

    ``driver`` is passed to ``torch.linalg.svd`` on CUDA. The contract's default is ``gesvd``:
    the CUDA Jacobi default can be less accurate and its error is comparable to bf16
    storage, which is indistinguishable from a small edit.
    """
    W0 = W0.to(FACTOR_DTYPE)
    W_star = W_star.to(FACTOR_DTYPE)
    kw = {"full_matrices": False}
    if W_star.is_cuda and driver is not None:
        kw["driver"] = driver
    U, s_star, Vh = torch.linalg.svd(W_star, **kw)
    U0, s0, Vh0 = torch.linalg.svd(W0, **kw)

    f = Factors(name=name, W0=W0, W_star=W_star, U=U, s_star=s_star, Vh=Vh, s0=s0,
                U0=U0, s0_basis=s0, Vh0=Vh0, solver=driver)
    f.rebuild_floor = _fro(rebuild(U, s_star, Vh) - W_star) / max(_fro(W0), 1e-30)
    f.base_rebuild_floor = _fro(rebuild(U0, s0, Vh0) - W0) / max(f.W0_norm, 1e-30)
    # Repeated singular values make individual vectors non-unique (PROTOCOL section 3). The
    # flag that matters is a tied *learned* block whose replacement values differ materially:
    # attributing an effect to an individual direction is then basis-dependent.
    if s_star.numel() > 1 and float(s_star.max()) > 0:
        tol = degenerate_rtol * float(s_star.max())
        tied = (s_star[:-1] - s_star[1:]).abs() <= tol
        f.degenerate = bool(tied.any())
        if f.degenerate:
            spread, start = 0.0, 0
            for i in range(s_star.numel()):
                last = i == s_star.numel() - 1
                if last or not bool(tied[i]):
                    if i > start:
                        blk = s0[start:i + 1]
                        denom = max(float(blk.abs().max()), 1e-30)
                        spread = max(spread, float(blk.max() - blk.min()) / denom)
                    start = i + 1
            f.degenerate_replacement_spread = spread
    return f


def _require_base_svd(f: Factors, operator: str):
    """Return the fixed base SVD or an explicit infeasibility record.

    Old serialized 21-cell factor files do not contain these fields.  They must be rebuilt under
    the new library identity rather than silently recomputing a potentially different base gauge
    once per cell.
    """
    if f.U0 is None or f.s0_basis is None or f.Vh0 is None:
        return None, _infeasible(
            operator, {},
            "fixed base SVD is absent; rebuild factors with SPECINT >= 1.1.0",
        )
    return (f.U0, f.s0_basis, f.Vh0), None


def _relative_gaps(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return values.new_empty(0)
    scale = torch.maximum(values[:-1].abs(), values[1:].abs()).clamp_min(
        torch.finfo(values.dtype).tiny)
    return (values[:-1] - values[1:]).abs() / scale


def orientation_alignment_blocks(f: Factors, gap_rtol: float = 1e-4) -> list[tuple[int, int]]:
    """Maximal blocks that must be aligned jointly because either endpoint is nearly repeated.

    A boundary is identifiable only when both the base and trained adjacent relative gaps exceed
    ``gap_rtol``.  Generic isolated modes therefore reduce to ordinary joint sign alignment.
    """
    if gap_rtol <= 0:
        raise ValueError("gap_rtol must be positive")
    base, bad = _require_base_svd(f, "orientation_alignment")
    if bad is not None:
        raise ValueError(bad.reason)
    _, s0_basis, _ = base
    k = f.s_star.numel()
    if k == 0:
        return []
    joined = (_relative_gaps(s0_basis) <= gap_rtol) | (
        _relative_gaps(f.s_star) <= gap_rtol)
    split_after = torch.nonzero(~joined, as_tuple=False).flatten().add(1).cpu().tolist()
    boundaries = [0, *split_after, k]
    return list(zip(boundaries[:-1], boundaries[1:]))


def aligned_trained_bases(f: Factors, gap_rtol: float = 1e-4):
    """Put trained left/right singular frames in one base-anchored, joint block gauge.

    For block ``B`` one orthogonal Procrustes factor is fitted to the *sum* of the left and right
    overlaps and applied to both trained frames.  Applying the same factor is essential: it fixes
    SVD signs and repeated-block gauge jointly rather than inventing independent U/V signs.
    """
    base, bad = _require_base_svd(f, "orientation_alignment")
    if bad is not None:
        raise ValueError(bad.reason)
    U0, _, Vh0 = base
    V, V0 = f.Vh.T, Vh0.T
    U_aligned = f.U.clone()
    V_aligned = V.clone()
    blocks = orientation_alignment_blocks(f, gap_rtol)
    repeated_blocks = [(start, stop) for start, stop in blocks if stop - start > 1]
    singleton_indices = torch.as_tensor(
        [start for start, stop in blocks if stop - start == 1],
        device=f.U.device, dtype=torch.long)
    max_residual = 0.0
    if singleton_indices.numel():
        score = ((f.U[:, singleton_indices] * U0[:, singleton_indices]).sum(dim=0)
                 + (V[:, singleton_indices] * V0[:, singleton_indices]).sum(dim=0))
        signs = torch.where(score < 0, -torch.ones_like(score), torch.ones_like(score))
        U_aligned[:, singleton_indices] = (U_aligned[:, singleton_indices]
                                            * signs.unsqueeze(0))
        V_aligned[:, singleton_indices] = (V_aligned[:, singleton_indices]
                                            * signs.unsqueeze(0))
    for start, stop in repeated_blocks:
        cross = (f.U[:, start:stop].T @ U0[:, start:stop]
                 + V[:, start:stop].T @ V0[:, start:stop])
        kwargs = ({"driver": f.solver} if cross.is_cuda and f.solver is not None else {})
        left, _, right_h = torch.linalg.svd(cross, full_matrices=False, **kwargs)
        q = left @ right_h
        U_aligned[:, start:stop] = f.U[:, start:stop] @ q
        V_aligned[:, start:stop] = V[:, start:stop] @ q
        eye = torch.eye(stop - start, device=q.device, dtype=q.dtype)
        max_residual = max(max_residual, float((q.T @ q - eye).abs().max()))
    return U_aligned, V_aligned.T, {
        "gauge": "joint_block_procrustes_to_fixed_base_svd",
        "gap_rtol": gap_rtol,
        "n_alignment_blocks": len(blocks),
        "n_singleton_alignment_blocks": int(singleton_indices.numel()),
        "n_repeated_alignment_blocks": len(repeated_blocks),
        "max_alignment_block_size": max((stop - start for start, stop in blocks), default=0),
        "repeated_alignment_blocks": [[start, stop] for start, stop in repeated_blocks],
        "max_alignment_orthogonality_residual": max_residual,
    }


def rotation_band_slices(f: Factors, search_fraction: float = 0.2) -> dict[str, tuple[int, int]]:
    """Three broad causal bands with boundaries pinned to nearby fixed-base spectral gaps."""
    if not 0 <= search_fraction < 0.5:
        raise ValueError("search_fraction must be in [0, 0.5)")
    base, bad = _require_base_svd(f, "rotation_band_restore")
    if bad is not None:
        raise ValueError(bad.reason)
    _, values, _ = base
    k, count = values.numel(), 3
    if k < count:
        raise ValueError("at least three singular values are required for three causal bands")
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
    return {name: span for name, span in zip(BANDS, zip(boundaries[:-1], boundaries[1:]))}


# --------------------------------------------------------------------------------------
# edit results
# --------------------------------------------------------------------------------------

OK, INFEASIBLE = "ok", "infeasible"


@dataclass
class Edit:
    """One constructed variant of one matrix, or a recorded refusal to construct it."""

    operator: str
    params: dict
    W: Optional[torch.Tensor]
    status: str = OK
    reason: Optional[str] = None
    info: dict = field(default_factory=dict)
    # Algebraic planned spectrum, not a measurement of the cast/installed weight.
    spectrum: Optional[torch.Tensor] = None

    @property
    def feasible(self) -> bool:
        return self.W is not None


def _infeasible(op, params, reason, **info) -> Edit:
    return Edit(operator=op, params=params, W=None, status=INFEASIBLE, reason=reason, info=info)


def _require_nonneg(s: torch.Tensor, op: str, params: dict) -> Optional[Edit]:
    """The protocol's hard constraint: edited spectral coefficients must be nonnegative."""
    neg = int((s < 0).sum())
    if neg:
        worst = float(s.min())
        return _infeasible(op, params,
                           f"{neg} edited singular values would be negative (min {worst:.3e}); "
                           "clipping and absolute-value folding are forbidden",
                           n_negative=neg, min_value=worst)
    return None


# --------------------------------------------------------------------------------------
# reference and norm-control operators
# --------------------------------------------------------------------------------------

def op_base(f: Factors) -> Edit:
    return Edit("base", {}, f.W0, spectrum=f.s0)


def op_trained(f: Factors) -> Edit:
    return Edit("trained", {}, f.W_star, spectrum=f.s_star)


def op_reconstruct(f: Factors) -> Edit:
    """U diag(s_star) Vh. The caller must push this through the same save/reload path as every
    other edit, so that the measured reconstruction floor includes storage casting."""
    return Edit("reconstruct", {}, rebuild(f.U, f.s_star, f.Vh),
                info={"rebuild_floor": f.rebuild_floor}, spectrum=f.s_star)


def op_restore(f: Factors) -> Edit:
    """W_R = U diag(s0) Vh."""
    return Edit("restore", {}, f.W_R,
                info={"er_norm_fraction": f.restoration_edit_norm / f.delta_norm if f.delta_norm else None},
                spectrum=f.s0)


def op_match_base(f: Factors) -> Edit:
    """W0 + g Delta with g = ||W_R - W0||_F / ||Delta||_F, per matrix.

    g is never capped at one: matching the restored point may require amplification.
    """
    if f.delta_norm == 0:
        return _infeasible("match_base", {}, "||Delta||_F = 0; the scaling is undefined")
    g = f.restoration_base_distance / f.delta_norm
    return Edit("match_base", {"g": g}, f.W0 + g * f.Delta, info={"g": g})


def op_match_edit(f: Factors, sign: int) -> Edit:
    """W_star +/- c Delta with c = ||E_R||_F / ||Delta||_F, per matrix."""
    op = "match_edit_plus" if sign > 0 else "match_edit_minus"
    if f.delta_norm == 0:
        return _infeasible(op, {}, "||Delta||_F = 0; the scaling is undefined")
    c = f.restoration_edit_norm / f.delta_norm
    return Edit(op, {"c": c}, f.W_star + sign * c * f.Delta, info={"c": c})


def op_update_scale(f: Factors, q: float) -> Edit:
    """W0 + q Delta. Acts on merged weights for both LoRA and OFT."""
    return Edit("update_scale", {"q": q}, f.W0 + q * f.Delta)


# --------------------------------------------------------------------------------------
# causal spectrum/orientation operators
# --------------------------------------------------------------------------------------

def op_spectrum_only(f: Factors) -> Edit:
    """Keep the trained singular values in the fixed pretrained singular frames."""
    base, bad = _require_base_svd(f, "spectrum_only")
    if bad is not None:
        return bad
    U0, _, Vh0 = base
    return Edit("spectrum_only", {}, rebuild(U0, f.s_star, Vh0), spectrum=f.s_star, info={
        "fixed_component": "base_left_and_right_singular_frames",
        "learned_component": "trained_singular_values",
        "base_rebuild_floor": f.base_rebuild_floor,
    })


def op_left_orientation_only(f: Factors, gap_rtol: float = 1e-4) -> Edit:
    """Apply only the trained left-frame motion to the pretrained SVD."""
    base, bad = _require_base_svd(f, "left_orientation_only")
    if bad is not None:
        return bad
    _, s0_basis, Vh0 = base
    U_aligned, _, alignment = aligned_trained_bases(f, gap_rtol)
    return Edit("left_orientation_only", {"gap_rtol": gap_rtol},
                rebuild(U_aligned, s0_basis, Vh0), spectrum=s0_basis, info={
                    **alignment,
                    "fixed_component": "base_singular_values_and_right_frame",
                    "learned_component": "trained_left_frame",
                    "base_rebuild_floor": f.base_rebuild_floor,
                })


def op_right_orientation_only(f: Factors, gap_rtol: float = 1e-4) -> Edit:
    """Apply only the trained right-frame motion to the pretrained SVD."""
    base, bad = _require_base_svd(f, "right_orientation_only")
    if bad is not None:
        return bad
    U0, s0_basis, _ = base
    _, Vh_aligned, alignment = aligned_trained_bases(f, gap_rtol)
    return Edit("right_orientation_only", {"gap_rtol": gap_rtol},
                rebuild(U0, s0_basis, Vh_aligned), spectrum=s0_basis, info={
                    **alignment,
                    "fixed_component": "base_singular_values_and_left_frame",
                    "learned_component": "trained_right_frame",
                    "base_rebuild_floor": f.base_rebuild_floor,
                })


def op_rotation_band_restore(f: Factors, band: str) -> Edit:
    """Restore one spectral third's orientation contribution to its pretrained value.

    The reference is ``W_R = U_* diag(s0) V_*^T``.  For band ``B`` this constructs

    ``W_R - U_*B diag(s0_B) V_*B^T + U_0B diag(s0_basis_B) V_0B^T``.

    Substituting complete rank-one/block contributions avoids concatenating base and trained
    singular frames, which would generally cease to be orthogonal.  The resulting matrix is an
    intentional causal ablation; unlike ``restore``, it is not asserted to be isospectral.
    """
    params = {"band": band}
    if band not in BANDS:
        return _infeasible("rotation_band_restore", params,
                           f"unknown band {band!r}; expected one of {BANDS}")
    base, bad = _require_base_svd(f, "rotation_band_restore")
    if bad is not None:
        bad.params = params
        return bad
    U0, s0_basis, Vh0 = base
    try:
        start, stop = rotation_band_slices(f)[band]
    except ValueError as exc:
        return _infeasible("rotation_band_restore", params, str(exc))
    indices = torch.arange(start, stop, device=f.W0.device)
    if not stop > start:
        return _infeasible("rotation_band_restore", params,
                           f"band {band!r} is empty")
    trained_component = rebuild(f.U[:, indices], f.s0[indices], f.Vh[indices, :])
    base_component = rebuild(U0[:, indices], s0_basis[indices], Vh0[indices, :])
    W = f.W_R - trained_component + base_component
    return Edit("rotation_band_restore", params, W, info={
        "reference": "restore",
        "construction": "replace_complete_rank_one_contributions",
        "band": band,
        "partition": "three broad bands; boundaries moved to nearby fixed-base spectral gaps",
        "index_start": start,
        "index_stop": stop,
        "index_count": stop - start,
        "spectrum_preserving": False,
        "trained_band_component_norm": _fro(trained_component),
        "base_band_component_norm": _fro(base_component),
        "planned_band_substitution_norm": _fro(base_component - trained_component),
        "base_rebuild_floor": f.base_rebuild_floor,
    })


# --------------------------------------------------------------------------------------
# complementary orientation-contribution selections (append-only extension)
# --------------------------------------------------------------------------------------

def _rotation_subset_edit(f: Factors, operator: str, params: dict,
                          start: int, stop: int, *, preserve: bool) -> Edit:
    """Keep/remove a contiguous contribution set; never used by legacy operators.

    Subsets are paired by sorted endpoint index, not by a claim that individual
    singular vectors retain their identity. No new alignment is introduced.
    """
    base, bad = _require_base_svd(f, operator)
    if bad is not None:
        bad.params = params
        return bad
    U0, s0_basis, Vh0 = base
    d = f.s_star.numel()
    if not 0 <= start < stop <= d:
        return _infeasible(operator, params, f"invalid index interval [{start}, {stop}) for d={d}")
    boundaries = []
    for boundary in (start, stop):
        if 0 < boundary < d:
            base_gap = float(_relative_gaps(s0_basis[boundary - 1:boundary + 1])[0])
            trained_gap = float(_relative_gaps(f.s_star[boundary - 1:boundary + 1])[0])
            boundaries.append({"index": boundary, "base_relative_gap": base_gap,
                               "trained_relative_gap": trained_gap,
                               "splits_nearly_repeated_block": min(base_gap, trained_gap) <= 1e-4})
    full = start == 0 and stop == d
    info = {
        "reference": "restore", "construction": "replace_complete_rank_one_contributions",
        "background": "base" if preserve else "restore",
        "selected_action": "preserve" if preserve else "restore",
        "index_start": start, "index_stop": stop, "index_count": stop - start,
        "matrix_singular_dimension": d,
        "preserved_index_count": stop - start if preserve else d - (stop - start),
        "restored_index_count": d - (stop - start) if preserve else stop - start,
        "boundary_gap_rtol": 1e-4, "boundary_gaps": boundaries,
        "boundary_identifiable": all(not b["splits_nearly_repeated_block"] for b in boundaries),
        "spectrum_preserving": full, "base_rebuild_floor": f.base_rebuild_floor,
    }
    if full:
        # Avoid introducing a reconstruction residual at an exact endpoint.
        info["exact_endpoint"] = "restore" if preserve else "base"
        return Edit(operator, params, f.W_R if preserve else f.W0,
                    spectrum=f.s0, info=info)
    trained_component = rebuild(f.U[:, start:stop], f.s0[start:stop], f.Vh[start:stop, :])
    base_component = rebuild(U0[:, start:stop], s0_basis[start:stop], Vh0[start:stop, :])
    difference = trained_component - base_component
    W = f.W0 + difference if preserve else f.W_R - difference
    info.update(trained_selected_component_norm=_fro(trained_component),
                base_selected_component_norm=_fro(base_component),
                planned_selected_substitution_norm=_fro(difference))
    return Edit(operator, params, W, info=info)


def op_rotation_band_preserve(f: Factors, band: str) -> Edit:
    """Keep one band's orientation contribution; restore the other two to base."""
    operator, params = "rotation_band_preserve", {"band": band}
    if band not in BANDS:
        return _infeasible(operator, params, f"unknown band {band!r}; expected one of {BANDS}")
    base, bad = _require_base_svd(f, operator)
    if bad is not None:
        bad.params = params
        return bad
    try:
        start, stop = rotation_band_slices(f)[band]
    except ValueError as exc:
        return _infeasible(operator, params, str(exc))
    edit = _rotation_subset_edit(f, operator, params, start, stop, preserve=True)
    edit.info.update(band=band,
                     partition="three broad bands; boundaries moved to nearby fixed-base spectral gaps")
    return edit


def _rotation_topk_edit(f: Factors, k: int, *, preserve: bool) -> Edit:
    operator = "rotation_topk_only" if preserve else "rotation_topk_restore"
    params = {"k": k}
    if type(k) is not int or k < 1:
        return _infeasible(operator, params, "k must be a positive integer (not bool)")
    d = f.s_star.numel()
    effective_k = min(k, d)
    edit = _rotation_subset_edit(f, operator, params, 0, effective_k, preserve=preserve)
    edit.info.update(requested_k=k, effective_k=effective_k,
                     k_policy="min(requested_k, matrix_singular_dimension)",
                     k_saturated=k > d, partition="descending sorted endpoint indices")
    return edit


def op_rotation_topk_only(f: Factors, k: int) -> Edit:
    """W0 plus the first k orientation contributions, not a rank-k weight truncation."""
    return _rotation_topk_edit(f, k, preserve=True)


def op_rotation_topk_restore(f: Factors, k: int) -> Edit:
    """W_R minus the first k orientation contributions; keep the learned remainder."""
    return _rotation_topk_edit(f, k, preserve=False)


# --------------------------------------------------------------------------------------
# spectral operators
# --------------------------------------------------------------------------------------

def op_spectral_path(f: Factors, lam: float) -> Edit:
    """s = s_star + lam (s0 - s_star). lam = 0 is trained, lam = 1 is restored."""
    params = {"lam": lam}
    s = f.s_star + lam * (f.s0 - f.s_star)
    bad = _require_nonneg(s, "spectral_path", params)
    if bad:
        return bad
    return Edit("spectral_path", params, rebuild(f.U, s, f.Vh), spectrum=s)


def spectral_edit(f: Factors, a: torch.Tensor, z: torch.Tensor, operator: str,
                  params: dict) -> Edit:
    """s = s_star + a * z, the common form of every sign-structured spectral operator.

    Because U and Vh are held fixed, ``||U diag(a*z) Vh||_F = ||a||_2`` exactly, independent of
    the signs z. Equal coordinatewise absolute amplitudes therefore give equal matrix-edit norms
    and equal immediate output-edit norms on any common input - the protocol's strong control.
    """
    s = f.s_star + a * z
    bad = _require_nonneg(s, operator, params)
    if bad:
        bad.info["requested_edit_norm"] = float(torch.linalg.vector_norm(a.to(REDUCE_DTYPE)))
        return bad
    e = Edit(operator, params, rebuild(f.U, s, f.Vh), spectrum=s)
    e.info["requested_edit_norm"] = float(torch.linalg.vector_norm(a.to(REDUCE_DTYPE)))
    return e


def amplitude_restoration(f: Factors, t: float) -> torch.Tensor:
    """a = t |s0 - s_star|; t = 1 is the observed restoration amplitude (Q3 start)."""
    return t * (f.s0 - f.s_star).abs()


def amplitude_relative(f: Factors, sigma: float) -> torch.Tensor:
    """a = sigma s_star."""
    return sigma * f.s_star


BANDS = ("top", "mid", "bottom")


def band_indices(k: int) -> dict:
    """Deterministic index thirds via numpy.array_split, as the contract fixes."""
    parts = np.array_split(np.arange(k), 3)
    return {name: torch.as_tensor(p.copy()) for name, p in zip(BANDS, parts)}


def band_masks(k: int, device=None) -> dict:
    idx = band_indices(k)
    out = {}
    for name, ix in idx.items():
        m = torch.zeros(k, dtype=torch.bool, device=device)
        if ix.numel():
            m[ix.to(device)] = True
        out[name] = m
    return out


def amplitude_band(f: Factors, band: str, d: float):
    """a_i = d s_star,i / ||s_star,B||_2 on band B, zero elsewhere.

    ``||a||_2 = d`` for every band, so the requested matrix-edit norm is identical across bands -
    the protocol's equal-edit-energy requirement. Returns ``(a, note)``; a zero-energy band
    cannot receive a relative edit and yields ``None``.
    """
    k = f.s_star.numel()
    mask = band_masks(k, device=f.s_star.device)[band]
    band_norm = float(torch.linalg.vector_norm(f.s_star[mask].to(REDUCE_DTYPE)))
    if band_norm == 0.0:
        return None, f"band '{band}' has zero spectral energy; a relative edit is undefined"
    a = torch.zeros_like(f.s_star)
    a[mask] = d * f.s_star[mask] / band_norm
    return a, None


def max_paired_scale(f: Factors, a_unit: torch.Tensor) -> float:
    """Largest c with both ``s_star + c a_unit`` and ``s_star - c a_unit`` nonnegative.

    Used to report the largest *common* feasible range for an antithetic pair instead of
    silently skipping the band or sign that fails first.
    """
    au = a_unit.abs().to(REDUCE_DTYPE)
    live = au > 0
    if not bool(live.any()):
        return float("inf")
    return float((f.s_star.to(REDUCE_DTYPE)[live] / au[live]).min())


def op_spectral_sign(f: Factors, t: float, z: torch.Tensor) -> Edit:
    return spectral_edit(f, amplitude_restoration(f, t), z, "spectral_sign", {"t": t})


def op_relative_sign(f: Factors, sigma: float, z: torch.Tensor) -> Edit:
    return spectral_edit(f, amplitude_relative(f, sigma), z, "relative_sign", {"sigma": sigma})


def op_spectral_band(f: Factors, band: str, d: float, z: torch.Tensor, coherent: bool) -> Edit:
    params = {"band": band, "d": d, "coherent": coherent}
    a, note = amplitude_band(f, band, d)
    if a is None:
        return _infeasible("spectral_band", params, note)
    return spectral_edit(f, a, z, "spectral_band", params)


def op_spectral_shape(f: Factors, v: torch.Tensor) -> Edit:
    """s' = ||s_star||_2 (s_star + v) / ||s_star + v||_2, requiring s_star + v positive.

    Shape change at fixed spectral 2-norm. Normalization is nonlinear, so the realized distance
    must be measured afresh and the two normalized signs are *not* an exact antithetic pair.
    """
    params = {"v_norm": float(torch.linalg.vector_norm(v.to(REDUCE_DTYPE)))}
    raw = f.s_star + v
    if bool((raw <= 0).any()):
        return _infeasible("spectral_shape", params,
                           "s_star + v is not strictly positive; shape normalization is undefined")
    s = float(torch.linalg.vector_norm(f.s_star.to(REDUCE_DTYPE))) * raw / float(
        torch.linalg.vector_norm(raw.to(REDUCE_DTYPE)))
    s = s.to(f.s_star.dtype)
    return Edit("spectral_shape", params, rebuild(f.U, s, f.Vh), spectrum=s)


def op_global_gain(f: Factors, target_edit_norm: float, sign: int = 1) -> Edit:
    """s = g s_star with |g - 1| ||s_star||_2 = target, i.e. a pure scale change matched in size
    to a shape edit. This is the comparison the protocol asks for alongside spectral_shape."""
    ns = float(torch.linalg.vector_norm(f.s_star.to(REDUCE_DTYPE)))
    params = {"target_edit_norm": target_edit_norm, "sign": sign}
    if ns == 0:
        return _infeasible("global_gain", params, "||s_star||_2 = 0")
    g = 1.0 + sign * target_edit_norm / ns
    if g < 0:
        return _infeasible("global_gain", params, f"gain {g:.3e} would make the spectrum negative")
    e = Edit("global_gain", {**params, "g": g}, rebuild(f.U, g * f.s_star, f.Vh),
             spectrum=g * f.s_star)
    e.info["g"] = g
    return e

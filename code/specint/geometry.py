"""Weight geometry and the high-dimensional nulls (PROTOCOL.md section 5).

All ratios are defined per matrix; network aggregation is the square root of summed numerator
energies over summed denominator energies, never the mean of per-matrix normalized values.
An undefined ratio is ``None`` with a reason, never 0 and never inf.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .ops import REDUCE_DTYPE, Factors, _fro, _fro2, rebuild, singular_values
from .rng import generator


# --------------------------------------------------------------------------------------
# per-matrix geometry
# --------------------------------------------------------------------------------------

@dataclass
class MatrixGeometry:
    name: str
    b_m: Optional[float]                 # ||W - W0||_F / ||W0||_F
    e_m: Optional[float]                 # ||W - W_star||_F / ||W0||_F
    s_m: Optional[float]                 # ||svdvals(W) - s0||_2 / ||s0||_2
    er_norm_fraction: Optional[float]    # ||E_R||_F / ||Delta||_F
    er_energy_fraction: Optional[float]  # the square of the above
    W0_norm: float
    delta_norm: float
    edit_norm: float                     # ||W - W_star||_F, unnormalized
    base_distance: float                 # ||W - W0||_F, unnormalized
    reason: Optional[str] = None
    spectrum_source: str = "svd_of_installed_weight"

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _ratio(num: float, den: float):
    if den == 0:
        return None
    return num / den


def matrix_geometry(f: Factors, W: torch.Tensor, spectrum: bool = True, *,
                    known_spectrum: Optional[torch.Tensor] = None,
                    spectrum_source: str = "svd_of_installed_weight") -> MatrixGeometry:
    """Geometry of one edited matrix against its own base and trained references.

    ``spectrum=False`` skips the ``s_m`` SVD, which is the expensive term; callers that only
    need distances should pass False and record ``s_m`` as null with a reason.
    """
    base_distance = _fro(W - f.W0)
    edit_norm = _fro(W - f.W_star)
    s_m, reason = None, None
    if spectrum:
        s0_norm = float(torch.linalg.vector_norm(f.s0.to(REDUCE_DTYPE)))
        if s0_norm == 0:
            reason = "||s0||_2 = 0; s_m undefined"
        else:
            sv = (singular_values(W.to(torch.float32), driver=f.solver)
                  if known_spectrum is None else known_spectrum.sort(descending=True).values)
            s_m = float(torch.linalg.vector_norm((sv - f.s0).to(REDUCE_DTYPE))) / s0_norm
    else:
        reason = "s_m not requested"
    er = _ratio(f.restoration_edit_norm, f.delta_norm)
    return MatrixGeometry(
        name=f.name,
        b_m=_ratio(base_distance, f.W0_norm),
        e_m=_ratio(edit_norm, f.W0_norm),
        s_m=s_m,
        er_norm_fraction=er,
        er_energy_fraction=(er ** 2 if er is not None else None),
        W0_norm=f.W0_norm,
        delta_norm=f.delta_norm,
        edit_norm=edit_norm,
        base_distance=base_distance,
        reason=reason,
        spectrum_source=spectrum_source if spectrum else "not_requested",
    )


def edit_geometry(f: Factors, edit, realized: torch.Tensor):
    """Measure installed weights; analytic spectra apply only to the planned matrix.

    Endpoint spectra can be reused only when the installed tensor is exactly that endpoint.
    General affine/hybrid edits still need a planned SVD. The measured spectrum of every
    other stored edit is computed explicitly, including BF16 quantization effects.
    """
    known, source = None, "svd_of_installed_weight"
    if edit.operator in ("base", "trained"):
        endpoint, values = ((f.W0, f.s0) if edit.operator == "base"
                            else (f.W_star, f.s_star))
        if torch.equal(realized, endpoint):
            known, source = values, "cached_exact_endpoint"
    actual = matrix_geometry(f, realized, known_spectrum=known, spectrum_source=source)
    if torch.equal(realized, edit.W):
        planned = actual
    else:
        planned = matrix_geometry(
            f, edit.W, known_spectrum=edit.spectrum,
            spectrum_source=("analytic_planned_coefficients" if edit.spectrum is not None
                             else "svd_of_planned_weight"))
    return actual, planned


def restoration_valid(f: Factors, realized: torch.Tensor, geometry: MatrixGeometry,
                      storage_cast_relative_error: float) -> tuple[bool, float]:
    """Shared LLM/FLUX restoration gate, including the actual storage cast."""
    tolerance = max(5e-5, 5 * (f.rebuild_floor or 0.0),
                    1.1 * storage_cast_relative_error)
    passed = bool(torch.isfinite(realized).all()) and (
        geometry.s_m is not None and geometry.s_m <= tolerance)
    if f.W0_norm == 0:
        passed = bool(torch.count_nonzero(realized) == 0)
    return passed, tolerance


# --------------------------------------------------------------------------------------
# network aggregation
# --------------------------------------------------------------------------------------

class EnergyAggregator:
    """sqrt(sum numerator energies / sum denominator energies) over the declared matrix scope."""

    def __init__(self):
        self._num = {}
        self._den = {}
        self.n = 0

    def add(self, key: str, numerator_energy: float, denominator_energy: float) -> None:
        self._num[key] = self._num.get(key, 0.0) + numerator_energy
        self._den[key] = self._den.get(key, 0.0) + denominator_energy

    def add_geometry(self, g: MatrixGeometry) -> None:
        self.add("b", g.base_distance ** 2, g.W0_norm ** 2)
        self.add("e", g.edit_norm ** 2, g.W0_norm ** 2)
        self.add("er", (g.er_norm_fraction or 0.0) ** 2 * g.delta_norm ** 2, g.delta_norm ** 2)
        self.n += 1

    def value(self, key: str):
        den = self._den.get(key, 0.0)
        if den == 0:
            return None
        return (self._num[key] / den) ** 0.5

    def as_dict(self) -> dict:
        out = {f"{k}_network": self.value(k) for k in self._den}
        out["n_matrices"] = self.n
        for k in self._den:
            if self._den[k] == 0:
                out[f"{k}_network_reason"] = "zero denominator energy over the declared scope"
        return out


# --------------------------------------------------------------------------------------
# matched-control verification (PROTOCOL section 6)
# --------------------------------------------------------------------------------------

def matching_error(realized: float, requested: float):
    """Relative matching error of a norm control; None when the request is degenerate."""
    if requested == 0:
        return None
    return abs(realized - requested) / requested


def check_match(realized: float, requested: float, tol: float = 0.01) -> tuple:
    """(status, error). ``unmatched`` when the control missed its target by more than tol."""
    err = matching_error(realized, requested)
    if err is None:
        return "infeasible", None
    return ("ok" if err <= tol else "unmatched"), err


def below_floor(requested_edit_norm: float, floor_norm: float) -> bool:
    """A requested change smaller than the measured reconstruction floor is unresolved, even
    when its relative match looks good."""
    return requested_edit_norm <= floor_norm


# --------------------------------------------------------------------------------------
# function-space probe (PROTOCOL section 5)
# --------------------------------------------------------------------------------------

def output_edit_energy(E: torch.Tensor, H: torch.Tensor) -> float:
    """sum_h ||E h||_2^2 for captured common inputs H of shape (n_samples, n).

    Separates a geometrically small edit from a functionally irrelevant one. Inputs must be
    captured from the *reference* model (base for base->trained, untouched trained for
    trained->edit), never from the edited network itself.
    """
    return float((E.to(REDUCE_DTYPE) @ H.to(REDUCE_DTYPE).T).pow(2).sum())


# --------------------------------------------------------------------------------------
# high-dimensional nulls (PROTOCOL section 5)
# --------------------------------------------------------------------------------------

def random_update(f: Factors, kind: str, rank: Optional[int], draw: int,
                  *ids: object) -> torch.Tensor:
    """A seeded random update matched per matrix to ||Delta||_F.

    ``kind`` is ``"rank_r"`` (a product of two Gaussian factors, algebraic rank ``rank``) or
    ``"dense"``. Matching is on the Frobenius norm of the update, not of the result.
    """
    m, n = f.W0.shape
    dev, dt = f.W0.device, f.W0.dtype
    g = generator(*ids, f.name, kind, rank, draw, device="cpu")
    if kind == "rank_r":
        if not rank:
            raise ValueError("rank_r null requires a rank")
        A = torch.randn(m, rank, generator=g).to(dev, dt)
        B = torch.randn(rank, n, generator=g).to(dev, dt)
        E = A @ B
    elif kind == "dense":
        E = torch.randn(m, n, generator=g).to(dev, dt)
    else:
        raise ValueError(f"unknown null kind {kind!r}")
    en = _fro(E)
    if en == 0:
        return E
    return E * (f.delta_norm / en)


def randomized_subspace_update(f: Factors, draw: int, *ids: object) -> torch.Tensor:
    """The stronger optional null: keep Delta's singular values, randomize its subspaces."""
    U, s, Vh = torch.linalg.svd(f.Delta.to(torch.float32), full_matrices=False,
                               **({"driver": f.solver} if f.W0.is_cuda else {}))
    m, n = f.W0.shape
    k = s.numel()
    g = generator(*ids, f.name, "subspace", draw, device="cpu")
    Qu, _ = torch.linalg.qr(torch.randn(m, k, generator=g).to(f.W0.device, torch.float32))
    Qv, _ = torch.linalg.qr(torch.randn(n, k, generator=g).to(f.W0.device, torch.float32))
    return rebuild(Qu, s, Qv.T).to(f.W0.dtype)


def base_basis_energy(f: Factors, E: torch.Tensor) -> dict:
    """Energy of an update in the base singular bases, including the part outside the thin spans.

    For rectangular matrices the diagonal/off-diagonal split inside the thin bases is incomplete:
    the residual outside them must be reported too, or a rotational preference can be claimed
    from an artefact of the truncation.
    """
    # A diagnostic compares the learned update with many random draws.  Reusing
    # this immutable base basis avoids repeating the largest SVD for every draw.
    if f.U0 is not None and f.Vh0 is not None:
        U0, Vh0 = f.U0, f.Vh0
    elif "base_svd_bases" not in f._cache:
        U0, _, Vh0 = torch.linalg.svd(f.W0.to(torch.float32), full_matrices=False,
                                     **({"driver": f.solver} if f.W0.is_cuda else {}))
        f._cache["base_svd_bases"] = (U0, Vh0)
    else:
        U0, Vh0 = f._cache["base_svd_bases"]
    C = U0.T @ E.to(torch.float32) @ Vh0.T                   # coefficients in the thin bases
    total = _fro2(E)
    inside = _fro2(C)
    diag = float((torch.diagonal(C).to(REDUCE_DTYPE) ** 2).sum())
    return {
        "energy_total": total,
        "energy_in_thin_bases": inside,
        "energy_outside_thin_bases": max(total - inside, 0.0),
        "energy_diagonal": diag,
        "energy_offdiagonal_inside": max(inside - diag, 0.0),
    }


def spectrum_shift(f: Factors, W: torch.Tensor) -> float:
    """||svdvals(W) - s0||_2 / ||s0||_2 - the quantity compared against the random nulls."""
    s0n = float(torch.linalg.vector_norm(f.s0.to(REDUCE_DTYPE)))
    if s0n == 0:
        return float("nan")
    sv = singular_values(W.to(torch.float32), driver=f.solver)
    return float(torch.linalg.vector_norm((sv - f.s0).to(REDUCE_DTYPE))) / s0n

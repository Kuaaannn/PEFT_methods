"""During-training measurement. Three tiers, separated by cost.

No optimizer step ever blocks on an SVD. The tiers are:

  Tier 0  every step        global scalars, < 1% overhead
  Tier 1  every N steps     per-matrix, NO SVD of any d_out x d_in matrix, ~2-3%

Tier 1 is cheap because of two exact identities that avoid ever materialising the merged
update. For a LoRA-family adapter with dW = s B A, B (d_out, r), A (r, d_in):

    ||dW||_F^2      = s^2 * tr[ (B^T B)(A A^T) ]
    sigma_i(dW)     = s * sqrt( lambda_i[ (B^T B)(A A^T) ] )

Both Gram matrices are r x r, so the FULL nonzero spectrum of the update -- and hence its
operator norm, stable rank, entropy rank, and energy ranks r_0.90/0.95/0.99 -- costs
O(r^2 (d_in + d_out)) and is EXACT. `B @ A` is never formed.

For OFT, dW = W_0 (R^T - I) with R block-diagonal, so the same quantities come from one
blocked matmul, and the orthogonality residual ||R^T R - I||_F is a batched (n_b, b, b)
bmm on tensors the forward pass already built.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch


# --------------------------------------------------------------------------------------
# Spectrum summaries shared by every tier
# --------------------------------------------------------------------------------------

def spectrum_summary(sv: torch.Tensor, prefix: str = "") -> dict[str, float]:
    """Norm/rank descriptors from a vector of singular values (descending, non-negative).

    Effective-rank conventions are pinned here so two quantities called "effective rank"
    are never compared across different definitions:
      stable rank        ||W||_F^2 / ||W||_2^2
      entropy rank       exp(-sum p_i log p_i)          with p_i = sigma_i^2 / ||W||_F^2
      participation rank 1 / sum p_i^2
      energy rank r_tau  min r such that cumulative energy fraction >= tau
    """
    sv = sv.detach().to(torch.float64).flatten()
    sv = sv[sv > 0]
    if sv.numel() == 0:
        return {}
    fro2 = float((sv ** 2).sum())
    op = float(sv.max())
    p = (sv ** 2) / fro2
    ent = float(-(p * p.clamp_min(1e-300).log()).sum())
    cum = torch.cumsum(p, 0)
    out = {
        f"{prefix}fro": math.sqrt(fro2),
        f"{prefix}op": op,
        f"{prefix}nuclear": float(sv.sum()),
        f"{prefix}stable_rank": fro2 / (op ** 2),
        f"{prefix}entropy_rank": math.exp(ent),
        f"{prefix}participation_rank": float(1.0 / (p ** 2).sum()),
        f"{prefix}n_sv": int(sv.numel()),
    }
    for tau in (0.90, 0.95, 0.99):
        out[f"{prefix}r_{int(tau * 100)}"] = int(torch.searchsorted(cum, tau).item()) + 1
    return out


def lowrank_update_spectrum(B: torch.Tensor, A: torch.Tensor,
                            scaling: float) -> torch.Tensor:
    """Exact nonzero singular values of dW = scaling * B @ A, without forming B @ A.

    The nonzero eigenvalues of (BA)^T (BA) equal those of (B^T B)(A A^T), an r x r
    product. Cost O(r^2 (d_in + d_out)) instead of O(d_in d_out min(d_in, d_out)).
    """
    # The Gram matrices are r x r, so float64 costs nothing and keeps the small
    # singular values accurate; squaring through a Gram loses half the precision.
    Bf = B.detach().to(torch.float64)
    Af = A.detach().to(torch.float64)
    GA = Af @ Af.T                            # (r, r) symmetric PSD
    GB = Bf.T @ Bf                            # (r, r) symmetric PSD
    # nonzero eig(B GA B^T) == eig(GA^{1/2} GB GA^{1/2}), symmetric PSD and r x r.
    # Symmetrising this way stays valid when B = 0 at initialization.
    evA, VA = torch.linalg.eigh(GA)
    GA_half = (VA * evA.clamp_min(0).sqrt()) @ VA.T
    ev = torch.linalg.eigvalsh(GA_half @ GB @ GA_half).clamp_min(0)
    return (scaling * ev.sqrt()).sort(descending=True).values


def lowrank_update_fro(B: torch.Tensor, A: torch.Tensor, scaling: float) -> float:
    """||scaling * B A||_F, via the same r x r trick."""
    Bf = B.detach().to(torch.float64)
    Af = A.detach().to(torch.float64)
    return float(scaling * torch.sqrt(torch.clamp(
        ((Bf.T @ Bf) * (Af @ Af.T).T).sum(), min=0.0)))


# --------------------------------------------------------------------------------------
# Tier 0 -- every optimizer step
# --------------------------------------------------------------------------------------

@dataclass
class StepTelemetry:
    """One row of per-step training metrics."""

    run_id: str
    seed: int
    step: int
    epoch_frac: float
    examples_seen: int
    tokens_seen: int
    lr: float
    cumulative_lr: float
    loss: float
    grad_norm_pre_clip: float
    grad_norm_post_clip: float
    param_norm: float
    update_norm: float
    update_param_cosine: float
    n_nonfinite: int
    skipped: bool
    step_time_s: float
    tokens_per_s: float
    mem_allocated: int
    mem_reserved: int

    def as_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


class StepRecorder:
    """Accumulates Tier-0 rows. Holds a flat copy of trainable params to form updates.

    The previous-parameter snapshot is the only real cost: one extra copy of the
    trainable set. For adapters that is tens of MB; for full fine-tuning of a 1.5B model
    it is ~6 GB in fp32, so `track_update` can be turned off and the update norm taken
    from the optimizer's own step instead.
    """

    def __init__(self, run_id: str, seed: int, params: Iterable[torch.nn.Parameter],
                 track_update: bool = True):
        self.run_id = run_id
        self.seed = seed
        self.params = [p for p in params if p.requires_grad]
        self.track_update = track_update
        self._prev = self._flat().clone() if track_update else None
        self.cumulative_lr = 0.0
        self.rows: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def _flat(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1).float() for p in self.params])

    def record(self, *, step: int, epoch_frac: float, examples_seen: int,
               tokens_seen: int, lr: float, loss: float,
               grad_norm_pre: float, grad_norm_post: float,
               tokens_this_step: int, skipped: bool = False) -> StepTelemetry:
        now = time.perf_counter()
        dt = now - self._t0
        self._t0 = now
        self.cumulative_lr += lr

        cur = self._flat()
        param_norm = float(cur.norm())
        if self.track_update and self._prev is not None:
            upd = cur - self._prev
            update_norm = float(upd.norm())
            denom = update_norm * param_norm
            cos = float((upd @ cur) / denom) if denom > 0 else 0.0
            self._prev = cur
        else:
            update_norm, cos = float("nan"), float("nan")

        n_nonfinite = sum(int((~torch.isfinite(p)).sum()) for p in self.params)
        t = StepTelemetry(
            run_id=self.run_id, seed=self.seed, step=step, epoch_frac=epoch_frac,
            examples_seen=examples_seen, tokens_seen=tokens_seen, lr=lr,
            cumulative_lr=self.cumulative_lr, loss=loss,
            grad_norm_pre_clip=grad_norm_pre, grad_norm_post_clip=grad_norm_post,
            param_norm=param_norm, update_norm=update_norm, update_param_cosine=cos,
            n_nonfinite=n_nonfinite, skipped=skipped, step_time_s=dt,
            tokens_per_s=tokens_this_step / dt if dt > 0 else 0.0,
            mem_allocated=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            mem_reserved=torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        )
        self.rows.append(t.as_row())
        return t


# --------------------------------------------------------------------------------------
# Tier 1 -- per-matrix, no large SVD
# --------------------------------------------------------------------------------------

@torch.no_grad()
def lora_matrix_telemetry(module, base_fro: float) -> dict[str, float]:
    """Per-matrix Tier-1 record for a LoRA / rsLoRA / DoRA layer.

    `module` is a peft LoraLayer. Reads the active adapter's A, B and scaling.
    """
    name = module.active_adapters[0]
    A = module.lora_A[name].weight            # (r, d_in)
    B = module.lora_B[name].weight            # (d_out, r)
    s = float(module.scaling[name])
    sv = lowrank_update_spectrum(B, A, s)

    out: dict[str, float] = {
        "a_fro": float(A.detach().float().norm()),
        "b_fro": float(B.detach().float().norm()),
        "scaling": s,
        "delta_fro": lowrank_update_fro(B, A, s),
    }
    out.update(spectrum_summary(sv, prefix="delta_"))
    out["delta_rel_fro"] = out["delta_fro"] / base_fro if base_fro > 0 else float("nan")

    # Factor conditioning for the adapter matrices.
    for tag, M in (("a", A), ("b", B)):
        Mf = M.detach().float()
        g = Mf @ Mf.T if Mf.shape[0] <= Mf.shape[1] else Mf.T @ Mf
        ev = torch.linalg.eigvalsh(g).clamp_min(0).sqrt()
        pos = ev[ev > 0]
        out[f"{tag}_sv_max"] = float(ev.max())
        out[f"{tag}_sv_min"] = float(pos.min()) if pos.numel() else 0.0
        out[f"{tag}_cond"] = (out[f"{tag}_sv_max"] / out[f"{tag}_sv_min"]
                              if out[f"{tag}_sv_min"] > 0 else float("inf"))

    # DoRA's magnitude path, kept separate from the direction path.
    mag = getattr(module, "lora_magnitude_vector", None)
    if mag is not None and name in mag:
        m = mag[name].weight.detach().float()
        out["dora_mag_mean"] = float(m.mean())
        out["dora_mag_std"] = float(m.std())
        out["dora_mag_min"] = float(m.min())
        out["dora_mag_max"] = float(m.max())
    return out


# Above this normalised residual, R is no longer orthogonal to the precision the
# project's central claim depends on ("OFT preserves every singular value exactly").
#
# The gate is not decorative under the `oft` arm. Truncated Cayley-Neumann is exact
# only for small generators, and NS5's residual is strongly angle-dependent
# (scripts/measure_oft_variants.py):
#
#     angle    exact      NS5       NS7       NS10
#     3e-03   1.31e-06  1.01e-06  2.38e-07  2.38e-07
#     1e-02   1.25e-06  1.29e-04  1.55e-06  2.98e-07
#     3e-02   1.43e-06  1.04e-02  1.08e-03  2.98e-07
#
# So NS5 holds to roughly a 5e-3 rotation angle and fails above it. Since angle grows
# with learning rate, a high-LR cell can leave the orthogonal manifold silently. That is
# a result to record, not an error to suppress -- the run continues and is flagged.
ORTHO_GATE = 1e-5


@torch.no_grad()
def oft_matrix_telemetry(module, W0: torch.Tensor, base_fro: float,
                         cached_R: torch.Tensor | None = None) -> dict[str, float]:
    """Per-matrix Tier-1 record for an OFT layer.

    Records the orthogonality residual used by the numerical gate: under the
    Cayley-Neumann default this is nonzero and grows with the generator, which would
    silently break the "OFT preserves every singular value" premise.
    """
    name = module.active_adapters[0]
    rot = module.oft_R[name]
    R = cached_R
    if R is None:
        R = rot._cayley_batch(rot.weight, rot.block_size,
                              rot.use_cayley_neumann, rot.num_cayley_neumann_terms)
    R = R.detach().float()                                  # (n_blocks, b, b)
    nb, b, _ = R.shape
    eye = torch.eye(b, device=R.device, dtype=R.dtype).expand(nb, b, b)
    ortho_res = torch.linalg.matrix_norm(R.transpose(-1, -2) @ R - eye)   # (n_blocks,)

    theta = rot.weight.detach().float()

    # Rotation magnitude from the TRACE, not from eigenvalues.
    #
    # For orthogonal R the eigenvalues are exp(+-i phi_j), so
    #     ||R - I||_F^2 = 2b - 2 tr(R)                       (exact, verified to 2e-14)
    #     RMS angle      = sqrt(2 (b - tr(R)) / b)           (3.8e-5 rel at 0.02 rad,
    #                                                         0.8% at 0.3 rad)
    # Both are O(b) from the diagonal. The obvious implementation --
    # `torch.linalg.eigvals(R.to(torch.complex64))` -- is O(b^3) COMPLEX, and profiling
    # showed it cost 1179 of 1297 ms for a single down_proj: 424 min of telemetry per
    # run against ~50 min of training. Grid 1's OFT arms spent 86% of their wall clock
    # measuring rotation angles.
    #
    # The exact angle DISTRIBUTION (mean, max, quantiles) is recovered at Tier 2, where
    # it runs once per checkpoint instead of every 50 steps.
    tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)          # (n_blocks,)
    dist2 = (2 * b - 2 * tr).clamp_min(0)
    angle_rms = torch.sqrt((2 * (b - tr) / b).clamp_min(0))

    # Normalised by sqrt(b) so the gate is comparable across block sizes: for a (b, b)
    # block, ||.||_F accumulates b^2 entries while the gate is a per-entry tolerance.
    ortho_norm = float(ortho_res.max()) / (b ** 0.5)

    out = {
        "block_size": float(b),
        "n_blocks": float(nb),
        "ortho_residual_max": float(ortho_res.max()),
        "ortho_residual_mean": float(ortho_res.mean()),
        "ortho_residual_norm": ortho_norm,
        "ortho_gate_exceeded": float(ortho_norm > ORTHO_GATE),
        "generator_fro": float(theta.norm()),
        "generator_max_abs": float(theta.abs().max()),
        "angle_rms": float(angle_rms.mean()),
        "angle_rms_max_block": float(angle_rms.max()),
        "dist_from_identity": float(torch.sqrt(dist2.sum())),
    }

    # dW = W_0 (R^T - I): one blocked matmul, no dense (d_in, d_in) rotation is formed.
    #
    # The transpose is load-bearing. PEFT merges as W <- W_0 R^T (right-OFT), so using
    # `bmm(Wb, R)` here would silently report an update norm ~1.8x the true one and
    # corrupt every OFT geometry row while training looked healthy. Pinned against the
    # real merge by tests/test_v0_orientation.py::test_oft_telemetry_delta_matches_merge.
    W0f = W0.detach().float()
    d_out, d_in = W0f.shape
    Wb = W0f.reshape(d_out, nb, b).permute(1, 0, 2)          # (nb, d_out, b)
    rotated = torch.bmm(Wb, R.transpose(-1, -2))             # W_0 R^T, blockwise
    dW = (rotated.permute(1, 0, 2).reshape(d_out, d_in) - W0f)
    out["delta_fro"] = float(dW.norm())
    out["delta_rel_fro"] = out["delta_fro"] / base_fro if base_fro > 0 else float("nan")
    return out


@dataclass
class MatrixRecorder:
    """Collects Tier-1 rows on a fixed cadence."""

    run_id: str
    seed: int
    every: int = 50
    rows: list[dict[str, Any]] = field(default_factory=list)

    def due(self, step: int) -> bool:
        return step % self.every == 0

    @torch.no_grad()
    def record(self, model, step: int, base_fro: dict[str, float],
               method: str) -> int:
        """Walk the adapted layers once and emit one row per matrix."""
        from peft.tuners.lora.layer import LoraLayer
        try:
            from peft.tuners.oft.layer import OFTLayer
        except ImportError:                                   # pragma: no cover
            OFTLayer = ()

        n = 0
        for name, mod in model.named_modules():
            key = name.replace("base_model.model.", "")
            if isinstance(mod, LoraLayer) and mod.active_adapters:
                rec = lora_matrix_telemetry(mod, base_fro.get(key + ".weight", 0.0))
            elif OFTLayer and isinstance(mod, OFTLayer) and mod.active_adapters:
                W0 = mod.get_base_layer().weight
                rec = oft_matrix_telemetry(mod, W0, base_fro.get(key + ".weight", 0.0))
            else:
                continue
            rec.update(run_id=self.run_id, seed=self.seed, step=step,
                       matrix=key, method=method)
            self.rows.append(rec)
            n += 1
        return n


@torch.no_grad()
def full_ft_matrix_telemetry(model, W0_cache: dict[str, torch.Tensor],
                             spec_names: list[str]) -> list[dict[str, Any]]:
    """Tier-1 for full fine-tuning: no adapter to read, so compare against cached W_0."""
    params = dict(model.named_parameters())
    rows = []
    for key in spec_names:
        W = params[key].detach().float()
        W0 = W0_cache[key].to(W.device, torch.float32)
        dW = W - W0
        w_fro, d_fro = float(W.norm()), float(dW.norm())
        w0_fro = float(W0.norm())
        rows.append({
            "matrix": key,
            "w_fro": w_fro,
            "w0_fro": w0_fro,
            "delta_fro": d_fro,
            "delta_rel_fro": d_fro / w0_fro if w0_fro > 0 else float("nan"),
            "w_w0_cosine": float((W * W0).sum() / (w_fro * w0_fro))
            if w_fro * w0_fro > 0 else float("nan"),
        })
    return rows

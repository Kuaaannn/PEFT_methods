"""One-sided PEFT mechanism diagnostics, NEVER substitutions for causal weights.

HRA follows the LLM joint108 compact-WY diagnostic: W*=W0 Q, V=Q.T V0.
OFT uses the actual saved raw transform: W*=W0 R.T, V=R V0. No polar
projection or independent SVD gauge correction is applied to evaluated weights.
"""
from __future__ import annotations

import torch

from specint.ops import _fro, rebuild


def _summary(f, Vh, replay, *, method, convention, orthogonality, rank_or_block):
    scale = max(f.W0_norm, 1e-30)
    error = _fro(replay.float() - f.W_star) / scale
    cosine = (f.Vh0 * Vh).sum(dim=1) / (f.Vh0.norm(dim=1) * Vh.norm(dim=1)).clamp_min(1e-30)
    angles = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
    row = {
        "matrix_name": f.name, "method": method, "convention": convention,
        "primary_analysis": "fixed_base_svd_analytic_transport",
        "analytic_u_relative_motion": 0.0, "u_transport": "exactly U0; no second SVD",
        "analytic_v_rotation_deg_mean": float(angles.mean()),
        "analytic_v_rotation_deg_max": float(angles.max()),
        "analytic_orthogonality_fro": orthogonality,
        "standard_merge_replay_relative_error": error,
        "standard_merge_replay_verified": error <= 1e-6,
        "analytic_vs_bf16_endpoint_relative_error": _fro(rebuild(f.U0, f.s0, Vh) - f.W_star) / scale,
        "merged_svd_u_mean_abs_cosine_diagnostic": float((f.U0 * f.U).sum(0).abs().clamp(0, 1).mean()),
        "merged_svd_u_angles_are_mechanism": False,
        "evaluation_weights": "unchanged_standard_merged_bfloat16",
        "intervention_factors": "unchanged_actual_merged_weight_SVD",
        **rank_or_block,
    }
    if not row["standard_merge_replay_verified"]:
        raise RuntimeError(f"One-sided merge convention replay failed: {row}")
    return row


@torch.inference_mode()
def transport_diagnostic(f, layer, method):
    if not f.W0.is_cuda or method not in ("hra", "oft"):
        raise ValueError("One-sided diagnostics require CUDA and HRA/OFT")
    active = layer.active_adapters
    if len(active) != 1:
        raise ValueError("Exactly one active adapter required")
    adapter = active[0]
    with torch.autocast(device_type="cuda", enabled=False):
        if method == "hra":
            from peft.tuners.hra.layer import _cwy_factors, _right_multiply_hra
            if layer.hra_apply_GS[adapter]:
                raise ValueError("This frozen batch uses non-GS HRA")
            vectors = layer.hra_u[adapter].detach()
            if not bool(torch.isfinite(vectors).all()) or bool((vectors.float().norm(dim=0) == 0).any()):
                raise ValueError("Nonfinite/zero Householder vector")
            Vh = _right_multiply_hra(f.Vh0.float(), vectors.float(), False, reverse=False, cast_input=False)
            replay = _right_multiply_hra(f.W0.to(torch.bfloat16), vectors, False).to(torch.bfloat16)
            u, t = _cwy_factors(vectors.float())
            gram, t = u.double().T @ u.double(), t.double()
            middle = -t - t.T + t.T @ gram @ t
            mg = middle @ gram
            defect = float((mg * mg.T).sum().clamp_min(0).sqrt())
            return _summary(f, Vh, replay, method=method, convention="W*=W0 Q; V=Q.T V0",
                            orthogonality=defect, rank_or_block={"rank": vectors.shape[1]})
        rotation = layer.get_delta_weight(adapter).detach()
        block_size = layer.oft_R[adapter].block_size
        # Reproduce PEFT's order and dtype exactly, independently of layer.merge().
        replay = (rotation @ f.W0.to(torch.bfloat16).to(rotation.dtype).T).T.to(torch.bfloat16)
        blocks = torch.stack([rotation[i:i + block_size, i:i + block_size].float()
                              for i in range(0, rotation.shape[0], block_size)])
        Vh = torch.bmm(blocks, f.Vh0.T.reshape(len(blocks), block_size, -1)).reshape_as(f.Vh0.T).T
        eye = torch.eye(block_size, device=blocks.device)
        defect = float(torch.linalg.vector_norm(blocks.transpose(-2, -1) @ blocks - eye))
        return _summary(f, Vh, replay, method=method, convention="W*=W0 R.T; V=R V0",
                        orthogonality=defect, rank_or_block={"oft_block_size": block_size,
                        "orthogonal_projection_applied": False})

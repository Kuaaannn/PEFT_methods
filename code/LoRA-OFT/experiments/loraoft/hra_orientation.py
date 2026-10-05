"""HRA's one-sided mechanism, kept separate from the BF16 evaluation endpoint.

For the saved non-GS HRA layer, W = W0 Q and Q = H(u1)...H(ur).
Fix W0 = U0 S0 V0.T once: U = U0, V.T = V0.T Q. No second SVD
defines the primary orientation. Compact WY avoids a dense input-size Q.

These are analysis-only diagnostics. They never replace a trained checkpoint,
SPECINT's merged-weight factors, or the weights used by standard evaluation.
"""
from __future__ import annotations


def transported_bases(base_svd, opt_u):
    import torch
    from peft.tuners.hra.layer import _right_multiply_hra
    U0, Vh0 = base_svd["U"], base_svd["Vh"]
    if (not U0.is_cuda or not Vh0.is_cuda or not opt_u.is_cuda
            or opt_u.ndim != 2 or opt_u.shape[0] != Vh0.shape[1]):
        raise ValueError("HRA transport requires compatible CUDA base factors and adapter vectors")
    if not bool(torch.isfinite(opt_u).all()) or bool((opt_u.float().norm(dim=0) == 0).any()):
        raise ValueError("Nonfinite/zero HRA Householder vector")
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        # Merge is W0 @ Q (reverse=False). Thus V = Q.T @ V0, NOT Q @ V0.
        Vh = _right_multiply_hra(Vh0.float(), opt_u.float(), False,
                                 reverse=False, cast_input=False)
    return U0, Vh


def transport_diagnostic(f, opt_u):
    """Small durable per-matrix record; full factors stay in ephemeral storage."""
    import torch
    from peft.tuners.hra.layer import _cwy_factors, _right_multiply_hra
    from specint.ops import _fro, rebuild
    base_svd = {"U": f.U0, "Vh": f.Vh0}
    U, Vh = transported_bases(base_svd, opt_u)
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        W = rebuild(U, f.s0_basis, Vh)
        # Independently replay the STANDARD merge precision/order, not the
        # higher-precision analytic endpoint. No change to evaluation weights.
        replay = _right_multiply_hra(f.W0.to(torch.bfloat16), opt_u, False)
        replay = replay.to(torch.bfloat16).float()
        merge_error = _fro(replay - f.W_star) / max(f.W0_norm, 1e-30)
        u, t = _cwy_factors(opt_u.float())
        gram = u.double().T @ u.double()
        t = t.double()
        middle = -t - t.T + t.T @ gram @ t
        mg = middle @ gram
        defect = float((mg * mg.T).sum().clamp_min(0).sqrt())
        cos_v = (f.Vh0 * Vh).sum(dim=1) / (
            f.Vh0.norm(dim=1) * Vh.norm(dim=1)).clamp_min(1e-30)
        angles_v = torch.rad2deg(torch.acos(cos_v.clamp(-1, 1)))
        # Same fixed U frame; any apparent output rotation from a fresh SVD is
        # not the learned Householder action. Keep that distinct, not zeroed.
        raw_u_cos = (f.U0 * f.U).sum(dim=0).abs().clamp(0, 1)
        result = {
            "matrix_name": f.name, "convention": "W_star = W0 @ Q",
            "Q": "ordered_householder_product_H1_to_Hr",
            "primary_analysis": "fixed_base_svd_analytically_transported_by_hra",
            "u_transport": "U0 (unchanged output frame)",
            "v_transport": "Q.T @ V0", "rank": opt_u.shape[1],
            "analytic_u_relative_motion": _fro(U - f.U0) / max(_fro(f.U0), 1e-30),
            "analytic_v_rotation_deg_mean": float(angles_v.mean()),
            "analytic_v_rotation_deg_max": float(angles_v.max()),
            "analytic_orthogonality_fro": defect,
            "standard_merge_replay_relative_error": merge_error,
            "standard_merge_replay_verified": merge_error <= 1e-6,
            "analytic_vs_bf16_endpoint_relative_error": _fro(W - f.W_star) / max(f.W0_norm, 1e-30),
            "merged_svd_u_mean_abs_cosine_diagnostic": float(raw_u_cos.mean()),
            "merged_svd_u_angles_are_mechanism": False,
            "evaluation_weights": "unchanged_standard_merged_bfloat16",
            "intervention_factors": "unchanged_actual_merged_weight_SVD",
        }
        if not result["standard_merge_replay_verified"]:
            raise RuntimeError(f"HRA right-side merge replay failed: {result}")
        return result

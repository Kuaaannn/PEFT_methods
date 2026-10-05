"""OFT Eq. (1) hyperspherical energy of ROW neurons in y = W x.

Standalone, opt-in extension: no SVD, model loading, PEFT, or legacy contract
changes. All numerical work requires CUDA. Ordered-pair energy is twice the
upper-triangle sum. FP32 (or audit FP64) GEMMs, FP64 kernels/reductions; never
project a transform or silently regularize the inverse-distance singularity.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path

import torch

VERSION = "he-riesz1-row-exact-v1"


def library_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@contextmanager
def _precision():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _normalize(W, dtype):
    # Float64 row norms also avoid under/overflow on very small/large rows.
    norms = torch.linalg.vector_norm(W.double(), dim=1)
    diagnostics = {"nonfinite_rows": int((~torch.isfinite(norms)).sum()),
                   "zero_rows": int((norms == 0).sum())}
    if any(diagnostics.values()):
        return None, diagnostics
    normalized = (W.double() / norms[:, None]).to(dtype)
    return normalized, diagnostics


def _direct_distances(W, rows, cols, *, chunk=256):
    """Exceptional near-coincident pairs: use direct normalized FP64 distance."""
    result = torch.empty(rows.numel(), device=W.device, dtype=torch.float64)
    for start in range(0, rows.numel(), chunk):
        a = W[rows[start:start + chunk]].double()
        b = W[cols[start:start + chunk]].double()
        a = a / torch.linalg.vector_norm(a, dim=1)[:, None]
        b = b / torch.linalg.vector_norm(b, dim=1)[:, None]
        result[start:start + chunk] = ((a - b) ** 2).sum(1)
    return result


def _distance_tile(N, W, i, j, block_size, repair_below):
    cosine = N[i:i + block_size] @ N[j:j + block_size].T
    distance2 = 2.0 - 2.0 * cosine.double()
    if i == j:
        # Nonpairs are masked before any inverse distance or minimum reduction.
        valid = torch.ones_like(distance2, dtype=torch.bool).triu(1)
    else:
        valid = torch.ones_like(distance2, dtype=torch.bool)
    suspect = valid & (distance2 < repair_below)
    # One scalar check per tile; the exceptional FP64 gather stays bounded.
    repaired = 0
    if bool(suspect.any()):
        indices = suspect.nonzero()
        repaired = indices.shape[0]
        d = _direct_distances(W, indices[:, 0] + i, indices[:, 1] + j)
        distance2[indices[:, 0], indices[:, 1]] = d
        cosine[indices[:, 0], indices[:, 1]] = (1.0 - d / 2.0).to(cosine.dtype)
    distance2.masked_fill_(~valid, float("inf"))
    # cos < -1 is only rounding. Its true minimum is -1 (squared chord 4).
    distance2.clamp_max_(4.0).masked_fill_(~valid, float("inf"))
    return cosine, distance2, valid, repaired


def _finite(x):
    import math
    value = float(x)
    return value if math.isfinite(value) else None


@torch.inference_mode()
def compare_hyperspherical_energy(W0, W, *, block_size=1024, matmul_dtype="float32",
                                  repair_below=1e-5, sample_count=4096, sample_seed=0):
    """Exact all-pairs statistics plus optional IID pairs for plotting/auditing.

    Returned `samples` are small CUDA tensors (serialization is the caller's
    responsibility). Samples are NOT used to estimate any primary metric.
    Invalid/divergent endpoints return explicit statuses and JSON-safe nulls.
    """
    if not W0.is_cuda or not W.is_cuda or W0.device != W.device:
        raise ValueError("HE requires two weights on the same CUDA device; no CPU fallback")
    if W0.ndim != 2 or W.shape != W0.shape or not W0.is_floating_point() or not W.is_floating_point():
        raise ValueError("Expected equally shaped real floating point [out, in] matrices")
    if block_size < 1 or sample_count < 0 or not 0 < repair_below < 1:
        raise ValueError("Invalid block size, sample count, or repair threshold")
    if matmul_dtype not in ("float32", "float64"):
        raise ValueError("matmul_dtype must be float32 or float64")
    n, d = W.shape
    count = n * (n - 1) // 2
    result = {"version": VERSION, "shape": [n, d], "neuron_axis": "rows",
              "ordered_pair_count": 2 * count, "unordered_pair_count": count,
              "all_pairs": True, "matmul_dtype": matmul_dtype, "reduction_dtype": "float64",
              "tf32": False, "block_size": block_size, "repair_below": repair_below,
              "base_storage_dtype": str(W0.dtype), "adapted_storage_dtype": str(W.dtype),
              "status": "ok", "base": None, "adapted": None,
              "relative_he_change": None, "absolute_relative_he_change": None,
              "pair_cosine_rms_change": None, "pair_kernel_relative_l1_change": None,
              "max_absolute_cosine_change": None, "samples": {}}
    if n < 2 or d < 1:
        result["status"] = "undefined_insufficient_neurons_or_features"
        return result
    dtype = getattr(torch, matmul_dtype)
    same = W is W0
    with _precision():
        N0, diag0 = _normalize(W0, dtype)
        N, diag = (N0, diag0) if same else _normalize(W, dtype)
        result["row_diagnostics"] = {"base": diag0, "adapted": diag}
        if N0 is None or N is None:
            result["status"] = "undefined_zero_or_nonfinite_rows"
            return result
        # [sum k0, sum k, sum delta k, sum |delta k|, sum delta cosine^2,
        #  number of coincident base pairs, number of coincident adapted pairs]
        sums = torch.zeros(7, dtype=torch.float64, device=W.device)
        minimum = torch.full((2,), float("inf"), dtype=torch.float64, device=W.device)
        max_change = torch.zeros((), dtype=torch.float64, device=W.device)
        repairs = [0, 0]
        for i in range(0, n, block_size):
            for j in range(i, n, block_size):
                c0, d0, valid, repaired0 = _distance_tile(N0, W0, i, j, block_size, repair_below)
                c, dt, _, repaired = ((c0, d0, valid, repaired0) if same else
                                      _distance_tile(N, W, i, j, block_size, repair_below))
                repairs[0] += repaired0
                repairs[1] += repaired
                k0, kt = d0.rsqrt(), dt.rsqrt()
                delta = kt - k0
                dc = (c.double() - c0.double()).masked_fill(~valid, 0)
                sums += torch.stack((k0.sum(), kt.sum(), delta.sum(), delta.abs().sum(),
                                     dc.square().sum(), (d0 == 0).sum(), (dt == 0).sum()))
                minimum = torch.minimum(minimum, torch.stack((d0.min(), dt.min())))
                max_change = torch.maximum(max_change, dc.abs().max())
        values = sums.tolist()  # Small final scalar serialization, not CPU analysis.
        mins = minimum.sqrt().tolist()
        for index, key in enumerate(("base", "adapted")):
            result[key] = {"he": _finite(2 * values[index]),
                           "mean_pair_energy": _finite(values[index] / count),
                           "minimum_pair_distance": mins[index],
                           "coincident_pair_count": int(values[5 + index]),
                           "fp64_repaired_pair_count": repairs[index]}
        result["pair_cosine_rms_change"] = _finite((sums[4] / count).sqrt())
        result["max_absolute_cosine_change"] = _finite(max_change)
        if values[5] or values[6]:
            result["status"] = "divergent_coincident_neurons"
        else:
            result["relative_he_change"] = values[2] / values[0]
            result["absolute_relative_he_change"] = abs(values[2] / values[0])
            result["pair_kernel_relative_l1_change"] = values[3] / values[0]
        if sample_count:
            generator = torch.Generator(device=W.device).manual_seed(sample_seed)
            rows = torch.randint(n, (sample_count,), generator=generator, device=W.device)
            cols = torch.randint(n - 1, (sample_count,), generator=generator, device=W.device)
            cols += (cols >= rows)
            # Uniform unordered pairs with replacement, same across methods/seeds.
            rows, cols = torch.minimum(rows, cols), torch.maximum(rows, cols)
            d0 = _direct_distances(W0, rows, cols)
            dt = d0 if same else _direct_distances(W, rows, cols)
            result["samples"] = {"row_i": rows, "row_j": cols,
                                 "base_cosine": (1 - d0 / 2).float(),
                                 "adapted_cosine": (1 - dt / 2).float()}
            result["sample_seed"] = sample_seed
            result["sample_count"] = sample_count
            result["sample_protocol"] = "uniform_unordered_pairs_with_replacement_fp64_direct"
        return result


def hyperspherical_energy(W, **kwargs):
    """Single-endpoint ordered-pair HE; see comparison for validity diagnostics."""
    result = compare_hyperspherical_energy(W, W, **kwargs)
    return {"status": result["status"], "energy": result["base"],
            "row_diagnostics": result.get("row_diagnostics")}

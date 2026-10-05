"""Capacity map: exact trainable-parameter arithmetic for every method.

This module exists so that parameter counts are computed from the model's actual
shapes *before* any performance is observed. The arithmetic path has no torch
dependency and does not require a GPU.

The central non-obvious fact it encodes:

    A uniform OFT block size `b` must divide `d_in` of EVERY adapted matrix.

For Qwen2.5-1.5B all-linear that means `b | gcd(1536, 8960) = 256`, so `b <= 256` and
`P_OFT <= 64.9M`. LoRA rank 64 (73.9M) already exceeds the entire uniform-OFT range and
rank 256 (295M) has no counterpart at all. Matching therefore runs in the other
direction: LoRA rank is a free integer with a ~1.15M granule, so we choose the rank that
matches each legal OFT block size. That closes four anchors to within 2.5%.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from math import gcd
from functools import reduce
from typing import Iterable

# Preregistered matching tolerance. Outside this, a pair is not called "matched"; it is
# reported as two separate points on the capacity curve.
MATCH_TOLERANCE = 0.10


@dataclass(frozen=True)
class MatrixSpec:
    """One adapted linear matrix, in the common convention y = W x, W is (d_out, d_in)."""

    name: str          # framework-native key suffix, e.g. "q_proj"
    d_out: int
    d_in: int
    n_layers: int = 1  # how many identical copies exist across the model

    @property
    def base_params(self) -> int:
        return self.d_out * self.d_in * self.n_layers

    def lora_params(self, r: int) -> int:
        return r * (self.d_in + self.d_out) * self.n_layers

    def dora_extra_params(self) -> int:
        """DoRA's magnitude vector: one scalar per output column of the merged weight."""
        return self.d_out * self.n_layers

    def hra_params(self, r: int) -> int:
        """HRA stores ``r`` Householder vectors in the input space."""
        if r <= 0 or r % 2:
            raise ValueError("HRA rank must be a positive even integer")
        return self.d_in * r * self.n_layers

    def oft_params(self, block_size: int) -> int:
        """Non-shared block-Cayley OFT: d_in/b blocks, each skew-symmetric b x b.

        Each block contributes b(b-1)/2 free parameters, so a matrix holds
        (d_in / b) * b(b-1)/2 = d_in (b - 1) / 2.
        """
        if self.d_in % block_size:
            raise ValueError(
                f"block size {block_size} does not divide d_in={self.d_in} for {self.name}"
            )
        return self.d_in * (block_size - 1) // 2 * self.n_layers

    def legal_block_sizes(self, max_block: int | None = None) -> list[int]:
        limit = max_block or self.d_in
        return [b for b in range(2, limit + 1) if self.d_in % b == 0]


# --------------------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------------------

def qwen2_all_linear(hidden: int, intermediate: int, n_layers: int, n_heads: int,
                     n_kv_heads: int) -> list[MatrixSpec]:
    """All transformer linear matrices of a Qwen2-family decoder.

    Embeddings, norms and the LM head are excluded: they are frozen in the primary
    comparison.
    """
    head_dim = hidden // n_heads
    kv_dim = head_dim * n_kv_heads
    return [
        MatrixSpec("q_proj", hidden, hidden, n_layers),
        MatrixSpec("k_proj", kv_dim, hidden, n_layers),
        MatrixSpec("v_proj", kv_dim, hidden, n_layers),
        MatrixSpec("o_proj", hidden, hidden, n_layers),
        MatrixSpec("gate_proj", intermediate, hidden, n_layers),
        MatrixSpec("up_proj", intermediate, hidden, n_layers),
        MatrixSpec("down_proj", hidden, intermediate, n_layers),
    ]


# Both 1.5B Qwen2.5 models share these shapes; verified against the published configs.
QWEN25_1_5B = qwen2_all_linear(hidden=1536, intermediate=8960, n_layers=28,
                               n_heads=12, n_kv_heads=2)

# PEFT's method_comparison default targets q_proj and v_proj only. Kept so the
# harness-validation cell can be built from the same code path.
LLAMA32_3B_ALL_LINEAR = qwen2_all_linear(hidden=3072, intermediate=8192, n_layers=28,
                                         n_heads=24, n_kv_heads=8)
LLAMA32_3B_QV = [m for m in LLAMA32_3B_ALL_LINEAR if m.name in ("q_proj", "v_proj")]

# meta-llama/Meta-Llama-3.1-8B. The seven adapted matrix shapes are verified against
# the published config: hidden=4096, intermediate=14336, 32 layers, 32 attention heads,
# and 8 key/value heads. Embeddings, norms, and the LM head remain frozen.
LLAMA31_8B_ALL_LINEAR = qwen2_all_linear(hidden=4096, intermediate=14336, n_layers=32,
                                         n_heads=32, n_kv_heads=8)


# --------------------------------------------------------------------------------------
# Aggregate arithmetic
# --------------------------------------------------------------------------------------

def lora_params(specs: Iterable[MatrixSpec], r: int) -> int:
    return sum(m.lora_params(r) for m in specs)


def dora_params(specs: Iterable[MatrixSpec], r: int) -> int:
    specs = list(specs)
    return lora_params(specs, r) + sum(m.dora_extra_params() for m in specs)


def hra_params(specs: Iterable[MatrixSpec], r: int) -> int:
    return sum(m.hra_params(r) for m in specs)


def oft_params(specs: Iterable[MatrixSpec], block_size: int) -> int:
    return sum(m.oft_params(block_size) for m in specs)


def lora_granule(specs: Iterable[MatrixSpec]) -> int:
    """Parameters added per unit of LoRA rank -- the resolution of the matching knob."""
    return lora_params(specs, 1)


def max_uniform_block_size(specs: Iterable[MatrixSpec]) -> int:
    """Largest block size dividing d_in of every adapted matrix."""
    return reduce(gcd, (m.d_in for m in specs))


def legal_uniform_block_sizes(specs: Iterable[MatrixSpec]) -> list[int]:
    """Every block size legal for ALL adapted matrices simultaneously.

    These are exactly the divisors >= 2 of gcd(d_in), which is why the ladder is so
    coarse: for Qwen2.5-1.5B it is {2, 4, 8, ..., 256}.
    """
    g = max_uniform_block_size(specs)
    return [b for b in range(2, g + 1) if g % b == 0]


def matched_rank_for_block(specs: Iterable[MatrixSpec], block_size: int) -> tuple[int, float]:
    """The LoRA rank whose parameter count best matches this OFT block size.

    Returns (rank, signed relative mismatch) where mismatch = (P_lora - P_oft) / P_oft.
    Rank is clamped to >= 1, so tiny block sizes can be unmatchable; the caller must
    check the mismatch against MATCH_TOLERANCE rather than trusting the rank.
    """
    specs = list(specs)
    p_oft = oft_params(specs, block_size)
    granule = lora_granule(specs)
    r = max(1, round(p_oft / granule))
    # round() can land on the wrong side when the ratio sits near .5; check the neighbour.
    best = min((abs(cand * granule - p_oft), cand) for cand in (r, max(1, r - 1), r + 1))[1]
    p_lora = best * granule
    return best, (p_lora - p_oft) / p_oft


@dataclass
class CapacityPoint:
    method: str                 # "lora" | "rslora" | "dora" | "oft" | "full"
    capacity_kind: str          # "rank" | "block_size" | "none"
    capacity: int
    trainable_params: int
    optimizer_state_bytes: int  # AdamW: two fp32 moments per trainable parameter
    anchor: str | None          # name of the matched-budget anchor, if any
    mismatch_to_anchor: float | None
    bracket: str | None         # "matched" | "lower" | "upper" | None
    note: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def build_capacity_map(specs: Iterable[MatrixSpec],
                       block_sizes: Iterable[int] | None = None,
                       extra_ranks: Iterable[int] = (),
                       base_total_params: int | None = None) -> list[CapacityPoint]:
    """Freeze the capacity map for one model/placement.

    For each legal OFT block size, emit the OFT point and the parameter-matched LoRA and
    DoRA points. Ranks in `extra_ranks` are emitted as unmatched conventional reference
    points -- they exist so the familiar numbers stay visible without being described as
    matched.
    """
    specs = list(specs)
    if block_sizes is None:
        block_sizes = [b for b in legal_uniform_block_sizes(specs) if b >= 16]
    points: list[CapacityPoint] = []
    matched_ranks: set[int] = set()

    for b in sorted(block_sizes):
        p_oft = oft_params(specs, b)
        anchor = f"b{b}"
        points.append(CapacityPoint("oft", "block_size", b, p_oft, 8 * p_oft,
                                    anchor, 0.0, "matched"))
        r, mismatch = matched_rank_for_block(specs, b)
        within = abs(mismatch) <= MATCH_TOLERANCE
        matched_ranks.add(r)
        note = "" if within else (
            f"no LoRA rank within {MATCH_TOLERANCE:.0%}; capacity-curve point only"
        )
        for method, count in (("lora", lora_params(specs, r)),
                              ("rslora", lora_params(specs, r)),
                              ("dora", dora_params(specs, r))):
            m = (count - p_oft) / p_oft
            points.append(CapacityPoint(method, "rank", r, count, 8 * count, anchor, m,
                                        "matched" if within else None, note))

    for r in sorted(set(extra_ranks) - matched_ranks):
        for method, count in (("lora", lora_params(specs, r)),
                              ("dora", dora_params(specs, r))):
            points.append(CapacityPoint(
                method, "rank", r, count, 8 * count, None, None, None,
                "conventional reference point, NOT parameter-matched to any legal OFT block",
            ))

    if base_total_params is not None:
        points.append(CapacityPoint("full", "none", 0, base_total_params,
                                    8 * base_total_params, None, None, None,
                                    "mechanism/performance reference, not parameter-matched"))
    return points


def freeze_capacity_map(points: list[CapacityPoint], path: str) -> None:
    """Write the map to disk. This file is an input to the experiment, not an output.

    It must be written before any performance number is observed, and never edited
    afterwards, to prevent parameter matching from depending on observed results.
    """
    with open(path, "w") as f:
        json.dump({"tolerance": MATCH_TOLERANCE,
                   "points": [p.as_row() for p in points]}, f, indent=2)


def format_table(points: list[CapacityPoint]) -> str:
    head = f"{'method':8} {'kind':11} {'cap':>5} {'params(M)':>10} {'anchor':>7} {'mismatch':>9}  note"
    lines = [head, "-" * len(head)]
    for p in points:
        mm = "" if p.mismatch_to_anchor is None else f"{100 * p.mismatch_to_anchor:+8.2f}%"
        lines.append(
            f"{p.method:8} {p.capacity_kind:11} {p.capacity:5d} "
            f"{p.trainable_params / 1e6:10.3f} {p.anchor or '':>7} {mm:>9}  {p.note}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    specs = QWEN25_1_5B
    print(f"adapted base params : {sum(m.base_params for m in specs) / 1e9:.3f} B")
    print(f"LoRA granule (r=1)  : {lora_granule(specs) / 1e6:.3f} M")
    print(f"max uniform block   : {max_uniform_block_size(specs)}")
    print()
    print(format_table(build_capacity_map(specs, extra_ranks=(8, 64, 256))))

"""Frozen 21-cell SPECINT plan and its six-cell causal extension.

``PLAN_21`` mirrors ``LoRA-OFT/experiments/checkpoint_protocol/lora_21/plan_21.json``;
``PLAN_27`` mirrors ``checkpoint_protocol/causal_27/plan_27.json`` after the LLM core is prepended.
Model-specific code only supplies the matrix name used by the shared RNG. Every intervention
equation is called directly from the shared :mod:`specint` package.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from . import ops
from .rng import random_signs


@dataclass(frozen=True)
class Cell:
    operator: str
    params: dict
    namespace: str

    def build(self, factors, _entry=None):
        return build_edit(factors, self.operator, self.params, self.namespace)

    @property
    def rng_ids(self) -> tuple:
        if self.params.get("z") in ("random", "neg_random"):
            return self.namespace, "sign", self.params.get("draw", 0)
        return ()

    def opposite(self, factors, _entry=None):
        spec = antithetic_spec(self.operator, self.params)
        if spec is None:
            return None
        operator, params = spec
        return params, build_edit(factors, operator, params, self.namespace)


# Order and parameters match the LLM lora_21 protocol exactly: seven core cells followed by the
# fourteen frozen diagnostic cells. Keep this literal visible and reviewable; it is a protocol,
# not a tunable sweep.
PLAN_21 = (
    ("restore", {}),
    ("base", {}),
    ("trained", {}),
    ("reconstruct", {}),
    ("match_base", {}),
    ("match_edit_plus", {}),
    ("match_edit_minus", {}),
    ("update_scale", {"q": 0.5}),
    ("spectral_path", {"lam": 0.5}),
    ("spectral_sign", {"t": 1.0, "z": "restore"}),
    ("spectral_sign", {"t": 0.001, "z": "random", "draw": 0}),
    ("spectral_sign", {"t": 0.001, "z": "neg_random", "draw": 0}),
    ("relative_sign", {"sigma": 0.001, "z": "ones", "draw": 0}),
    ("relative_sign", {"sigma": 0.001, "z": "neg_ones", "draw": 0}),
    ("relative_sign", {"sigma": 0.001, "z": "random", "draw": 0}),
    ("relative_sign", {"sigma": 0.001, "z": "neg_random", "draw": 0}),
    ("spectral_shape", {"sigma": 0.001, "z": "random", "draw": 0}),
    ("global_gain", {"sigma": 0.001, "z": "random", "draw": 0}),
    ("spectral_band", {"band": "top", "d_rel": 0.001, "z": "ones"}),
    ("spectral_band", {"band": "mid", "d_rel": 0.001, "z": "ones"}),
    ("spectral_band", {"band": "bottom", "d_rel": 0.001, "z": "ones"}),
)

# Six endpoint-derived causal decompositions.  These extend, rather than alter, PLAN_21.
# ``rotation_band_restore`` uses three broad bands with boundaries pinned to base spectral gaps.
CAUSAL_6 = (
    ("spectrum_only", {}),
    ("left_orientation_only", {"gap_rtol": 1e-4}),
    ("right_orientation_only", {"gap_rtol": 1e-4}),
    ("rotation_band_restore", {"band": "top"}),
    ("rotation_band_restore", {"band": "mid"}),
    ("rotation_band_restore", {"band": "bottom"}),
)

PLAN_27 = PLAN_21 + CAUSAL_6

# Append-only source IDs 28..40. Never renumber the frozen inventories.
ORIENTATION_SELECTION_13 = (
    ("rotation_band_preserve", {"band": "top"}),
    ("rotation_band_preserve", {"band": "mid"}),
    ("rotation_band_preserve", {"band": "bottom"}),
    ("rotation_topk_only", {"k": 1}),
    ("rotation_topk_only", {"k": 4}),
    ("rotation_topk_only", {"k": 32}),
    ("rotation_topk_only", {"k": 128}),
    ("rotation_topk_only", {"k": 512}),
    ("rotation_topk_restore", {"k": 1}),
    ("rotation_topk_restore", {"k": 4}),
    ("rotation_topk_restore", {"k": 32}),
    ("rotation_topk_restore", {"k": 128}),
    ("rotation_topk_restore", {"k": 512}),
)
PLAN_40 = PLAN_27 + ORIENTATION_SELECTION_13
CAUSAL_9_SOURCE_IDS = (1, 3, 4, 22, 23, 24, 25, 26, 27)
CAUSAL_22_SOURCE_IDS = CAUSAL_9_SOURCE_IDS + tuple(range(28, 41))
PLAN_22 = tuple(PLAN_40[i - 1] for i in CAUSAL_22_SOURCE_IDS)


def randomness_id(bank_hash: str) -> str:
    """Use the same frozen-bank RNG namespace as the LLM checkpoint analysis."""
    return "frozen-bank/" + bank_hash


def _signs(f, mode: str, draw: int, namespace: str) -> torch.Tensor:
    if mode in ("restore", "neg_restore"):
        z = torch.sign(f.s0 - f.s_star)
        return -z if mode == "neg_restore" else z
    if mode in ("ones", "neg_ones"):
        z = torch.ones_like(f.s_star)
        return -z if mode == "neg_ones" else z
    if mode in ("random", "neg_random"):
        z = random_signs(
            f.s_star.numel(),
            namespace,
            f.name,
            "sign",
            draw,
            device=f.W0.device,
            dtype=f.W0.dtype,
        )
        return -z if mode == "neg_random" else z
    raise ValueError(f"unknown sign mode {mode!r}")


def build_edit(f, operator: str, params: dict, namespace: str):
    """Dispatch exactly as the LLM adapter does; algebra remains in shared SPECINT."""
    if operator in ("base", "trained", "reconstruct", "restore", "match_base"):
        return getattr(ops, f"op_{operator}")(f)
    if operator in ("match_edit_plus", "match_edit_minus"):
        return ops.op_match_edit(f, 1 if operator.endswith("plus") else -1)
    if operator == "update_scale":
        return ops.op_update_scale(f, **params)
    if operator == "spectral_path":
        return ops.op_spectral_path(f, **params)
    if operator == "spectrum_only":
        return ops.op_spectrum_only(f)
    if operator == "left_orientation_only":
        return ops.op_left_orientation_only(f, **params)
    if operator == "right_orientation_only":
        return ops.op_right_orientation_only(f, **params)
    if operator == "rotation_band_restore":
        return ops.op_rotation_band_restore(f, **params)
    if operator == "rotation_band_preserve":
        return ops.op_rotation_band_preserve(f, **params)
    if operator == "rotation_topk_only":
        return ops.op_rotation_topk_only(f, **params)
    if operator == "rotation_topk_restore":
        return ops.op_rotation_topk_restore(f, **params)

    mode = params.get("z", "random")
    z = _signs(f, mode, params.get("draw", 0), namespace)
    if operator == "spectral_sign":
        return ops.op_spectral_sign(f, params["t"], z)
    if operator == "relative_sign":
        return ops.op_relative_sign(f, params["sigma"], z)
    if operator == "spectral_band":
        d = params["d_rel"] * float(f.s_star.double().norm())
        return ops.op_spectral_band(f, params["band"], d, z, mode in ("ones", "neg_ones"))
    if operator in ("spectral_shape", "global_gain"):
        shape = ops.op_spectral_shape(f, params["sigma"] * f.s_star * z)
        if not shape.feasible or operator == "spectral_shape":
            return shape
        return ops.op_global_gain(f, ops._fro(shape.W - f.W_star), params.get("sign", 1))
    raise ValueError(f"unsupported operator {operator!r}")


def antithetic_spec(operator: str, params: dict):
    """Return the same opposite request used by the LLM geometry checks."""
    if operator == "match_edit_plus":
        return "match_edit_minus", {}
    if operator == "match_edit_minus":
        return "match_edit_plus", {}
    if operator in ("spectral_sign", "relative_sign", "spectral_band"):
        opposite = {
            "restore": "neg_restore",
            "neg_restore": "restore",
            "ones": "neg_ones",
            "neg_ones": "ones",
            "random": "neg_random",
            "neg_random": "random",
        }
        mode = params.get("z")
        if mode in opposite:
            return operator, {**params, "z": opposite[mode]}
    return None


def make_cell(operator: str, params: dict, bank_hash: str) -> Cell:
    """Create one cell using the shared frozen-bank RNG namespace."""
    return Cell(operator, dict(params), randomness_id(bank_hash))


def plan_21(bank_hash: str) -> list[Cell]:
    """Instantiate the exact frozen plan for one evaluation bank."""
    cells = [make_cell(operator, params, bank_hash) for operator, params in PLAN_21]
    if len(cells) != 21:
        raise AssertionError(f"frozen protocol must contain 21 cells, found {len(cells)}")
    return cells


def plan_27(bank_hash: str) -> list[Cell]:
    """The frozen 21 cells followed by the six causal decomposition cells."""
    cells = [make_cell(operator, params, bank_hash) for operator, params in PLAN_27]
    if len(cells) != 27 or cells[:21] != plan_21(bank_hash):
        raise AssertionError("causal protocol must preserve PLAN_21 and contain 27 cells")
    return cells


def plan_40(bank_hash: str) -> list[Cell]:
    """Frozen 27 cells plus thirteen complementary orientation selections."""
    cells = [make_cell(operator, params, bank_hash) for operator, params in PLAN_40]
    if len(cells) != 40 or cells[:27] != plan_27(bank_hash):
        raise AssertionError("expanded protocol must preserve PLAN_27 and contain 40 cells")
    return cells


def plan_22(bank_hash: str) -> list[Cell]:
    """Original joint36 causal9 followed by the thirteen new selections."""
    return [make_cell(operator, params, bank_hash) for operator, params in PLAN_22]

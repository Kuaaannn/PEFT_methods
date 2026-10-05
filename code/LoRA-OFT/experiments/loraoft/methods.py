"""Construct the shared method configurations for training and evaluation.

OFT uses PEFT's block-diagonal rotation with five-term Cayley-Neumann by
default, non-shared blocks, and no COFT constraint. The truncated series is
approximate: orthogonality residuals are measured during training.
HRA uses the bundled compact-WY implementation with identity initialization.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

MethodName = Literal[
    "full", "lora", "rslora", "dora", "pissa", "milora", "hra",
    "oft",
]

# All transformer linear matrices. Embeddings, norms and the LM head stay frozen in the
# primary comparison. Every method receives exactly this set in a given comparison.
ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTENTION_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_ONLY = ["gate_proj", "up_proj", "down_proj"]
# PEFT's method_comparison default resolves to this; used only by the harness-validation cell.
PEFT_DEFAULT_QV = ["q_proj", "v_proj"]

PLACEMENTS = {
    "all-linear": ALL_LINEAR,
    "attention-only": ATTENTION_ONLY,
    "mlp-only": MLP_ONLY,
    "peft-default-qv": PEFT_DEFAULT_QV,
}


@dataclass
class MethodSpec:
    """A fully resolved, immutable description of one method arm."""

    method: MethodName
    placement: str = "all-linear"
    # LoRA family
    r: int | None = None
    alpha: int | None = None
    dropout: float = 0.0
    # OFT family
    block_size: int | None = None
    use_cayley_neumann: bool = True
    num_cayley_neumann_terms: int = 5
    block_share: bool = False
    coft: bool = False
    # Keep adapter storage in FP32; torch.autocast controls forward arithmetic.
    autocast_adapter_dtype: bool = True

    def __post_init__(self) -> None:
        if self.method in ("lora", "rslora", "dora", "pissa", "milora", "hra"):
            if self.r is None:
                raise ValueError(f"{self.method} requires r")
            if self.method != "hra" and self.alpha is None:
                # alpha/r = 2, the convention PEFT-Arena and PEFT's own comparison use.
                self.alpha = 2 * self.r
            if self.method == "hra" and self.r % 2:
                raise ValueError("HRA requires an even rank for identity initialization")
            if self.block_size is not None:
                raise ValueError(f"{self.method} does not take block_size")
        elif self.method == "oft":
            if self.block_size is None:
                raise ValueError(f"{self.method} requires block_size")
            if self.r is not None:
                raise ValueError("OFT capacity is block_size; passing r would be ambiguous")
            if self.coft:
                raise ValueError("COFT radius constraint is excluded from the primary comparison")
            if self.block_share:
                raise ValueError("primary OFT is non-shared")
        elif self.method == "full":
            if any(v is not None for v in (self.r, self.block_size)):
                raise ValueError("full fine-tuning takes no capacity parameter")
        else:
            raise ValueError(f"unknown method {self.method!r}")

    @property
    def target_modules(self) -> list[str]:
        return list(PLACEMENTS[self.placement])

    @property
    def capacity_kind(self) -> str:
        if self.method == "full":
            return "none"
        return "rank" if self.r is not None else "block_size"

    @property
    def capacity(self) -> int:
        return self.r if self.r is not None else (self.block_size or 0)

    def label(self) -> str:
        """Stable, filesystem-safe identifier used in run IDs and result paths."""
        if self.method == "full":
            return "full"
        cap = f"r{self.r}" if self.r is not None else f"b{self.block_size}"
        return f"{self.method}-{cap}"

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def build_method(spec: MethodSpec):
    """Return (peft_config_or_None, resolved_kwargs).

    `None` means full fine-tuning: no adapter is attached and every target matrix trains
    directly. The caller is responsible for freezing embeddings, norms and the LM head.
    """
    from peft import HRAConfig, LoraConfig, OFTConfig

    if spec.method == "full":
        return None, {}

    if spec.method in ("lora", "rslora", "dora", "pissa", "milora"):
        kwargs = dict(
            r=spec.r,
            lora_alpha=spec.alpha,
            lora_dropout=spec.dropout,
            target_modules=spec.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            # PiSSA/MiLoRA are initialized after wrapping by the shared project-level
            # spectral initializer. Keeping this a stock LoRA config lets their
            # converted checkpoints reload through unmodified standard PEFT eval.
            init_lora_weights=True,
            use_rslora=spec.method == "rslora",
            use_dora=spec.method == "dora",
        )
        return LoraConfig(**kwargs), kwargs

    if spec.method == "hra":
        kwargs = dict(
            r=spec.r,
            apply_GS=False,
            target_modules=spec.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            init_weights=True,
        )
        return HRAConfig(**kwargs), kwargs

    kwargs = dict(
        oft_block_size=spec.block_size,
        r=0,                                # PEFT requires exactly one of r / oft_block_size
        module_dropout=spec.dropout,
        target_modules=spec.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        init_weights=True,                  # identity init
        block_share=False,
        coft=False,
        use_cayley_neumann=spec.use_cayley_neumann,
        num_cayley_neumann_terms=spec.num_cayley_neumann_terms,
    )
    return OFTConfig(**kwargs), kwargs


def freeze_non_target_parameters(model, spec: MethodSpec) -> tuple[int, int]:
    """For full fine-tuning, train only the target linear matrices.

    Adapter methods handle this themselves. Returns (n_trainable, n_total).
    """
    if spec.method != "full":
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return n_tr, sum(p.numel() for p in model.parameters())

    targets = tuple(spec.target_modules)
    n_tr = 0
    n_total = 0
    for name, param in model.named_parameters():
        n_total += param.numel()
        # Matches "...layers.3.self_attn.q_proj.weight" but never embeddings, norm
        # vectors, or the LM head.
        is_target = is_target_weight(canonical_name(name), targets)
        param.requires_grad_(is_target)
        if is_target:
            n_tr += param.numel()
    return n_tr, n_total


def canonical_name(param_name: str) -> str:
    """Strip PEFT's wrapper segments so adapter and full-FT runs share matrix keys."""
    return param_name.replace("base_model.model.", "").replace(".base_layer", "")


def resolve_param(params: dict, canonical: str):
    """Find a parameter by its canonical name, whatever wrapping PEFT applied.

    A PEFT-wrapped weight is `base_model.model.<path>.base_layer.weight` while the same
    weight unwrapped is `<path>.weight`, and the two prefixes are applied independently.
    Enumerating the combinations in one place stops each caller from inventing its own
    partial lookup -- a missed combination does not raise, it returns nothing, and the
    metric that depended on it silently becomes NaN.
    """
    candidates = (
        canonical,
        f"base_model.model.{canonical}",
        canonical.replace(".weight", ".base_layer.weight"),
        f"base_model.model.{canonical}".replace(".weight", ".base_layer.weight"),
    )
    for c in candidates:
        if c in params:
            return params[c]
    return None


def is_target_weight(canonical: str, targets) -> bool:
    """Does this canonical parameter name refer to one of the target linear matrices?

    Matches on a whole path SEGMENT, so `q_proj.weight` at the top level and
    `model.layers.0.self_attn.q_proj.weight` both match, while a substring collision
    cannot. Requiring a leading dot silently misses top-level modules.
    """
    return any(canonical == f"{t}.weight" or canonical.endswith(f".{t}.weight")
               for t in targets)


def target_matrix_names(model, spec: MethodSpec) -> list[str]:
    """Fully-qualified names of the base weights under analysis, in a stable order.

    Used by telemetry and geometry so that every run indexes matrices identically. Names
    are of the *base* weight even for adapter methods, because analysis operates on the
    merged W in the common convention.
    """
    targets = tuple(spec.target_modules)
    # Canonicalise BEFORE filtering. A PEFT-wrapped weight is named
    # `...q_proj.base_layer.weight`, which does not contain `.q_proj.weight`, so
    # filtering on the raw name silently returns an EMPTY list for every adapter method
    # -- and an empty list makes base norms empty, which makes every relative update norm
    # NaN, with no error anywhere along the way.
    names = {canonical_name(n) for n, _ in model.named_parameters()}
    matched = sorted(n for n in names if is_target_weight(n, targets))
    if not matched:
        raise ValueError(
            f"no target matrices matched {targets} in this model; "
            "placement or architecture naming is wrong"
        )
    return matched

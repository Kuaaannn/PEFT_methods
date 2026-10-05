"""Module manifest for the FLUX transformer (PROTOCOL.md section 1).

Fixes, for each selected linear operator: orientation, fused/sliced boundaries, role and layer
index. The default unit is the whole adapted weight matrix, and a fused projection counts as one
matrix. The manifest is hashed, and that hash keys every cached result: changing the matrix scope
must invalidate measurements rather than silently mix them.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Optional

# transformer_blocks.<i>.attn.to_q.weight  /  single_transformer_blocks.<i>.attn.to_qkv_mlp_proj.weight
_NAME = re.compile(r"^(single_transformer_blocks|transformer_blocks)\.(\d+)\.(.+)\.weight$")

# Projections that pack several logical operators into one tensor. The protocol keeps them whole.
_FUSED = {"attn.to_qkv_mlp_proj": ("q", "k", "v", "mlp")}


def clean_param_name(name: str) -> str:
    """Transformer parameter name as it exists on the *unwrapped* model.

    PEFT renames an adapted module's weight to ``<module>.base_layer.weight`` and prefixes the
    tree with ``base_model.model.``. Both are wrapper artefacts; the manifest keys on the plain
    name so that a manifest hash does not depend on whether an adapter happened to be attached.
    """
    return name.replace("base_model.model.", "").replace(".base_layer.", ".")


@dataclass
class ModuleEntry:
    name: str                  # parameter name on the transformer, ending in ".weight"
    block_type: str            # "transformer_blocks" | "single_transformer_blocks"
    layer: int
    role: str                  # e.g. "attn.to_q", "ff.linear_in"
    shape: tuple               # (m, n) with y = W h
    m: int
    n: int
    aspect: float              # m / n
    fused_parts: Optional[tuple] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModuleManifest:
    entries: list
    orientation: str = "y = W h, W is (m, n)"
    unit: str = "whole adapted weight matrix; a fused projection is one matrix"

    def __len__(self):
        return len(self.entries)

    def names(self):
        return [e.name for e in self.entries]

    def hash(self) -> str:
        payload = json.dumps(
            {"orientation": self.orientation, "unit": self.unit,
             "entries": [[e.name, e.m, e.n] for e in self.entries]},
            sort_keys=True).encode()
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def as_dict(self) -> dict:
        return {"orientation": self.orientation, "unit": self.unit, "hash": self.hash(),
                "n_matrices": len(self.entries), "entries": [e.as_dict() for e in self.entries]}

    def stratified(self, per_role: int = 1) -> list:
        """One entry per role, for the diagnostic subset. Deterministic: lowest layer index wins."""
        seen, out = {}, []
        for e in sorted(self.entries, key=lambda x: (x.role, x.block_type, x.layer)):
            k = e.role
            if seen.get(k, 0) < per_role:
                seen[k] = seen.get(k, 0) + 1
                out.append(e)
        return out

    def by_role(self) -> dict:
        d = {}
        for e in self.entries:
            d.setdefault(e.role, []).append(e)
        return d


def adapter_weight_names(adapter_keys) -> set[str]:
    """Decode saved PEFT keys, including HRA's ParameterDict (no .weight suffix)."""
    stems = set()
    for key in adapter_keys:
        for marker in (".lora_A", ".lora_B", ".oft_R", ".hra_u"):
            if marker in key:
                stem, suffix = key.split(marker, 1)
                if not suffix or suffix.startswith("."):
                    stems.add(stem)
                    break
    return {clean_param_name(stem + ".weight") for stem in stems}


def from_adapter_keys(adapter_keys, param_shapes: dict, *, strict=False) -> ModuleManifest:
    """Build the manifest from the adapter's own target list and the base model's shapes.

    ``adapter_keys`` are PEFT state-dict keys; ``param_shapes`` maps transformer parameter name
    to shape. Both are facts about the checkpoint, so the manifest cannot drift from what was
    actually adapted.
    """
    names = adapter_weight_names(adapter_keys)
    missing = names - set(param_shapes)
    if strict and (not names or missing):
        raise ValueError(f"Empty/unknown adapted matrix coverage: {sorted(missing)}")
    entries = []
    for name in sorted(names):
        if name not in param_shapes:
            continue
        shape = tuple(param_shapes[name])
        m_ = _NAME.match(name)
        if not m_:
            block_type, layer, role = "other", -1, name[: -len(".weight")]
        else:
            block_type, layer, role = m_.group(1), int(m_.group(2)), m_.group(3)
        entries.append(ModuleEntry(
            name=name, block_type=block_type, layer=layer, role=role,
            shape=shape, m=shape[0], n=shape[1], aspect=shape[0] / shape[1],
            fused_parts=_FUSED.get(role),
        ))
    entries.sort(key=lambda e: (e.block_type, e.layer, e.role))
    return ModuleManifest(entries=entries)

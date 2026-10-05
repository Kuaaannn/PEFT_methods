"""Standard-dtype adapter merging, checked against native inference.

The merge uses PEFT's actual implementation at the model's configured dtype. This is the same
representation evaluated by the LLM checkpoint protocol. Only after the exact merged tensor is
captured does shared SPECINT widen it for SVD and geometry.

No model evaluated by this repository is converted to FP32.
"""
from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from .modules import ModuleManifest, clean_param_name, from_adapter_keys


@dataclass
class MergeCheck:
    name: str
    rel_error: float                     # ||merged(h) - native(h)|| / ||native(h)||
    passed: bool
    tolerance: float
    rel_error_vs_update: float = 0.0     # the same discrepancy against ||native(h) - base(h)||


def _peft_layers(peft_model) -> dict:
    """Adapted PEFT layers keyed by the underlying transformer parameter name."""
    from peft.tuners.lora.layer import LoraLayer
    from peft.tuners.oft.layer import OFTLayer
    from peft.tuners.hra.layer import HRALayer
    out = {}
    for mod_name, mod in peft_model.named_modules():
        if isinstance(mod, (LoraLayer, OFTLayer, HRALayer)):
            out[clean_param_name(mod_name + ".weight")] = mod
    return out


def build_manifest(peft_model, adapter_state_dict, *, strict=False) -> ModuleManifest:
    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    shapes = {clean_param_name(n): tuple(p.shape) for n, p in base.named_parameters()}
    return from_adapter_keys(adapter_state_dict.keys(), shapes, strict=strict)


def merged_pair(layer, device="cuda", check: bool = True, tolerance: float = 0.01,
                n_probe: int = 8, generator: Optional[torch.Generator] = None):
    """Return ``(W0, W_star, MergeCheck)`` in the layer's standard storage dtype.

    The native and merged forwards use the same layer copy, dtype and inputs. Their comparison is
    diagnostic; the merged endpoint itself is the checkpoint-analysis representation, matching
    the LLM protocol's standard-evaluation policy.

    ``rel_error_vs_update`` is reported alongside: it is the discrepancy measured against the
    size of the update rather than the size of the output, which is what makes the check
    sensitive to a dropped scaling or a transposed rotation.
    """
    base_layer = layer.get_base_layer()
    dtype = base_layer.weight.dtype
    W0 = base_layer.weight.detach().to(device=device).clone()

    merged_layer = copy.deepcopy(layer).to(device=device)

    native = None
    h = None
    if check:
        n = W0.shape[1]
        g = generator or torch.Generator(device=device).manual_seed(0)
        h = torch.randn(n_probe, n, generator=g, device=device, dtype=dtype)
        with torch.no_grad():
            native = merged_layer(h).detach()

    merged_layer.merge()
    W_star = merged_layer.get_base_layer().weight.detach().clone()

    chk = None
    if check:
        merged = h @ W_star.T
        if base_layer.bias is not None:
            merged = merged + base_layer.bias.detach().to(device=device, dtype=dtype)
        diff = float((merged.float() - native.float()).norm())
        rel = diff / max(float(native.float().norm()), 1e-30)
        base_out = h @ W0.T
        if base_layer.bias is not None:
            base_out = base_out + base_layer.bias.detach().to(device=device, dtype=dtype)
        update_scale = float((native.float() - base_out.float()).norm())
        chk = MergeCheck(name=getattr(layer, "_pipeline_name", ""), rel_error=rel,
                         passed=rel <= tolerance, tolerance=tolerance,
                         rel_error_vs_update=diff / max(update_scale, 1e-30))

    del merged_layer
    return W0, W_star, chk


def iter_pairs(peft_model, manifest: ModuleManifest, device="cuda", check: bool = True,
               names=None, verbose=False, strict=False) -> Iterator:
    """Yield ``(entry, W0, W_star, MergeCheck)`` for each selected matrix, freeing as it goes."""
    layers = _peft_layers(peft_model)
    wanted = set(names) if names else None
    expected = set(manifest.names()) if wanted is None else wanted
    if strict and (not expected or expected - set(layers)
                   or expected - set(manifest.names())):
        raise ValueError(f"Adapted layer coverage mismatch: {sorted(expected - set(layers))}")
    for i, e in enumerate(manifest.entries):
        if wanted and e.name not in wanted:
            continue
        layer = layers.get(e.name)
        if layer is None:
            continue
        layer._pipeline_name = e.name
        W0, W_star, chk = merged_pair(layer, device=device, check=check)
        yield e, W0, W_star, chk
        del W0, W_star
        if i % 16 == 0:
            gc.collect()
            torch.cuda.empty_cache()

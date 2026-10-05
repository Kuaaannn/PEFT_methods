"""FLUX model loading (adapter layer).

The only place this repository names a model class or a checkpoint directory layout. Kept apart
from ``merge`` so that the merge path can be exercised on a stub module in a unit test.
"""
from __future__ import annotations

import json
import pathlib

import torch
from safetensors.torch import load_file

from .manifest import BASE_MODEL, HERE, Checkpoint


def load_transformer(dtype=torch.bfloat16, device="cuda"):
    from diffusers import Flux2Transformer2DModel
    tf = Flux2Transformer2DModel.from_pretrained(
        BASE_MODEL, subfolder="transformer", torch_dtype=dtype)
    return tf.to(device).eval()


def adapter_dir(ck: Checkpoint) -> pathlib.Path:
    return (HERE / ck.weights).parent


def adapter_state_dict(ck: Checkpoint) -> dict:
    return load_file(str(HERE / ck.weights))


def adapter_config(ck: Checkpoint) -> dict:
    if ck.adapter_config:
        return ck.adapter_config
    return json.loads((adapter_dir(ck) / "adapter_config.json").read_text())


def attach(
    ck: Checkpoint,
    transformer=None,
    dtype=torch.bfloat16,
    device="cuda",
    autocast_adapter_dtype: bool = True,
):
    """Base transformer with this checkpoint's adapter applied, unmerged."""
    from peft import PeftModel
    tf = transformer if transformer is not None else load_transformer(dtype=dtype, device=device)
    return PeftModel.from_pretrained(
        tf,
        str(adapter_dir(ck)),
        torch_device=str(device),
        autocast_adapter_dtype=autocast_adapter_dtype,
    )

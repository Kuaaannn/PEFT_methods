"""Lazy imports and strict allocation checks. Never selects a CPU fallback."""
from __future__ import annotations

import importlib
import os
import sys
import signal
from pathlib import Path

from .io import REPO, WORKSPACE, file_hash, read_json

_SIGNAL_PID = None


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)


def implementation_identity():
    import importlib.metadata
    bridge()
    import peft
    paths = [REPO / "experiments/loraoft" / name for name in ("methods.py", "spectral_lora.py", "capacity.py")]
    paths += [Path(peft.__file__).parent / name for name in
              ("tuners/hra/layer.py", "tuners/oft/layer.py", "tuners/lora/layer.py")]
    return {"files": {str(path): file_hash(path) for path in paths},
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "peft", "accelerate", "datasets", "safetensors")}}


def require_cuda():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Each task uses one independent GPU, not distributed training")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This profile requires bfloat16 support")
    global _SIGNAL_PID
    if _SIGNAL_PID != os.getpid():
        signal.signal(signal.SIGTERM, _terminate)
        _SIGNAL_PID = os.getpid()
    return torch


def bridge():
    """Expose existing pure helpers only inside a new-harness process."""
    for root in (REPO / "experiments", WORKSPACE):
        if str(root) not in sys.path:
            sys.path.append(str(root))
    return importlib.import_module("loraoft.methods")


def check_peft():
    import peft
    import peft.tuners.hra.layer as hra
    from peft import OFTConfig
    required = {"oft_block_size", "use_cayley_neumann", "num_cayley_neumann_terms"}
    if not required.issubset(OFTConfig.__dataclass_fields__) or not hasattr(hra, "_cwy_factors"):
        raise RuntimeError("Install the bundled PEFT runtime with standard OFT and compact-WY HRA")
    reference = REPO / "experiments/third_party/peft-runtime/peft"
    for relative in ("tuners/hra/layer.py", "tuners/oft/layer.py", "tuners/lora/layer.py"):
        actual = Path(peft.__file__).parent / relative
        if file_hash(actual) != file_hash(reference / relative):
            raise RuntimeError(f"PEFT implementation differs from protected math: {relative}")


def validate_snapshot(config):
    root = Path(config.snapshot)
    data = read_json(root / "config.json")
    expected = {"qwen": ("qwen2", 3584, 18944, 28), "llama": ("llama", 4096, 14336, 32)}[config.model_key]
    actual = tuple(data.get(key) for key in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers"))
    if actual != expected:
        raise ValueError(f"Wrong base model: {actual} != {expected}")
    if root.name != config.model_revision or root.parent.name != "snapshots":
        raise ValueError("Use the pinned local HF snapshots/<revision> directory")
    if root.parent.parent.name != "models--" + config.model_id.replace("/", "--"):
        raise ValueError("Snapshot is not from the declared BASE model repository (no Instruct/Coder substitution)")
    index = read_json(root / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    if any(not (root / shard).is_file() for shard in shards):
        raise FileNotFoundError("Incomplete local base snapshot")
    return {"model_id": config.model_id, "revision": config.model_revision,
            "config_sha256": file_hash(root / "config.json"),
            "index_sha256": file_hash(root / "model.safetensors.index.json"),
            "shards": [{"name": s, "bytes": (root / s).stat().st_size,
                        "resolved": str((root / s).resolve())} for s in shards]}


def tokenizer_for(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.snapshot, local_files_only=True, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def base_model(config):
    torch = require_cuda()
    validate_snapshot(config)
    from transformers import AutoModelForCausalLM
    # Put HRA/DoRA initialization on CUDA as well; never wrap a host-resident model.
    return AutoModelForCausalLM.from_pretrained(
        config.snapshot, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True)


def method_spec(config):
    methods = bridge()
    kwargs = {"block_size": config.block_size} if config.method == "oft" else {"r": config.rank}
    if config.method not in ("oft", "hra"):
        kwargs["alpha"] = config.alpha
    return methods.MethodSpec(config.method, **kwargs)


def attach_for_training(config, model):
    methods = bridge()
    check_peft()
    from peft import get_peft_model
    spec = method_spec(config)
    peft_config, _ = methods.build_method(spec)
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=True)
    spectral = None
    if config.method in ("pissa", "milora"):
        from loraoft.spectral_lora import SpectralLoRARuntime
        spectral = SpectralLoRARuntime(config.method, base_model_id=config.model_id,
                                      cache_path=Path(config.spectral_cache) if config.spectral_cache else None)
        spectral.initialize(model)
    n, total = methods.freeze_non_target_parameters(model, spec)
    if config.expected_trainable is not None and n != config.expected_trainable:
        raise ValueError(f"Trainable parameters {n} != {config.expected_trainable}")
    names = methods.target_matrix_names(model, spec)
    if len(names) != model.config.num_hidden_layers * 7:
        raise ValueError("Incomplete all-matrix target coverage")
    return model, spectral, {"trainable": n, "total": total, "target_matrices": names,
                             "method_spec": spec.as_row()}

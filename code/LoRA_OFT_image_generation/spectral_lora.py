"""PiSSA/MiLoRA initialization without modifying PEFT.

Both methods move a rank-r SVD component from the frozen base into ordinary
LoRA factors. PiSSA selects the principal component and MiLoRA selects the
minor component. Checkpoints are exported through PEFT's official mutated-init
conversion as standard rank-2r LoRA adapters relative to the original base.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


SpectralMethod = Literal["pissa", "milora"]
SCHEMA_VERSION = 1
ALGORITHM = "exact-svd-residual-lora-v1"
METHOD_CONFIG_NAME = "method_config.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_method_config(experiment: str | Path, peft_config=None) -> dict:
    """Load explicit method identity, or infer legacy native PEFT methods."""
    path = Path(experiment) / METHOD_CONFIG_NAME
    if path.is_file():
        config = json.loads(path.read_text())
        if config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported method config schema: {path}")
        method = config.get("method")
        if method not in {"lora", "dora", "pissa", "milora", "hra", "oft"}:
            raise ValueError(f"unsupported method {method!r} in {path}")
        return config

    peft_type = str(getattr(peft_config, "peft_type", "")).split(".")[-1].lower()
    if peft_type == "lora":
        method = "dora" if bool(getattr(peft_config, "use_dora", False)) else "lora"
    elif peft_type in {"hra", "oft"}:
        method = peft_type
    elif peft_config is None:
        method = "full"
    else:
        method = peft_type
    return {"schema_version": SCHEMA_VERSION, "method": method, "inferred": True}


def validate_method_config(config: dict, peft_config) -> None:
    """Fail before model loading when method identity and PEFT config disagree."""
    method = config["method"]
    peft_type = str(getattr(peft_config, "peft_type", "")).split(".")[-1].lower()
    if method in {"lora", "dora", "pissa", "milora"} and peft_type != "lora":
        raise ValueError(f"method {method} requires a LORA adapter_config, found {peft_type}")
    if method == "hra" and peft_type != "hra":
        raise ValueError(f"method hra requires an HRA adapter_config, found {peft_type}")
    if method == "oft" and peft_type != "oft":
        raise ValueError(f"method oft requires an OFT adapter_config, found {peft_type}")
    if method == "dora" and not bool(getattr(peft_config, "use_dora", False)):
        raise ValueError("DoRA method config requires use_dora=true")
    if method != "dora" and bool(getattr(peft_config, "use_dora", False)):
        raise ValueError(f"method {method} cannot use a DoRA adapter config")
    if method in {"pissa", "milora"}:
        if getattr(peft_config, "init_lora_weights", None) is not True:
            raise ValueError(
                "project-level PiSSA/MiLoRA requires init_lora_weights=true; "
                "the residual decomposition is applied exactly once by spectral_lora"
            )
        expected_rank = config.get("training_rank")
        if expected_rank is not None and int(expected_rank) != int(peft_config.r):
            raise ValueError(f"method training_rank={expected_rank} but adapter r={peft_config.r}")
        if bool(getattr(peft_config, "use_rslora", False)):
            raise ValueError("formal PiSSA/MiLoRA uses ordinary LoRA scaling, not rsLoRA")
        dropout = getattr(peft_config, "lora_dropout", 0.0)
        if isinstance(dropout, dict):
            nonzero = any(float(value) != 0.0 for value in dropout.values())
        else:
            nonzero = float(dropout) != 0.0
        if nonzero:
            raise ValueError("formal PiSSA/MiLoRA uses zero LoRA dropout")
    if method == "hra":
        if int(peft_config.r) % 2:
            raise ValueError("HRA rank must be even for identity initialization")
        if bool(getattr(peft_config, "apply_GS", False)):
            raise ValueError("formal HRA uses the paper/default apply_GS=false setting")


def _adapter_name(layer) -> str:
    active = list(layer.active_adapters)
    if active != ["default"]:
        raise ValueError(f"spectral LoRA requires the single default adapter, found {active}")
    return active[0]


def _layer_signature(model) -> str:
    rows = []
    for name, layer, adapter in _linear_lora_layers(model):
        rows.append({
            "name": name,
            "shape": list(layer.get_base_layer().weight.shape),
            "fan_in_fan_out": bool(layer.fan_in_fan_out),
            "rank": int(layer.r[adapter]),
            "scaling": float(layer.scaling[adapter]),
        })
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _linear_lora_layers(model):
    from peft.tuners.lora.layer import LoraLayer

    for name, layer in model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        adapter = _adapter_name(layer)
        if adapter not in layer.lora_A or adapter not in layer.lora_B:
            raise TypeError(f"{name}: spectral initialization supports linear LoRA layers only")
        if layer.get_base_layer().weight.ndim != 2:
            raise TypeError(f"{name}: spectral initialization requires a matrix weight")
        yield name, layer, adapter


@dataclass
class SpectralLoRARuntime:
    method: SpectralMethod
    base_model_id: str | None = None
    reconstruction_tolerance: float = 1e-2
    cache_path: Path | None = None
    initialized: bool = False
    layer_count: int = 0
    max_init_relative_error: float = 0.0
    training_rank: int | None = None
    training_alpha: float | None = None
    layer_signature: str | None = None
    _temporary: tempfile.TemporaryDirectory | None = field(default=None, init=False, repr=False)
    _initial_adapter: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.method not in ("pissa", "milora"):
            raise ValueError(f"unsupported spectral LoRA method {self.method!r}")
        if self.cache_path is not None:
            self.cache_path = Path(self.cache_path)

    @property
    def spectral_side(self) -> str:
        return "principal" if self.method == "pissa" else "minor"

    def _validate_layer(self, layer, adapter: str) -> tuple[int, float]:
        rank = int(layer.r[adapter])
        scaling = float(layer.scaling[adapter])
        if rank <= 0 or scaling <= 0:
            raise ValueError(f"invalid spectral LoRA rank/scaling: r={rank}, scaling={scaling}")
        alpha = scaling * rank
        if self.training_rank is None:
            self.training_rank, self.training_alpha = rank, alpha
        elif rank != self.training_rank or abs(alpha - float(self.training_alpha)) > 1e-8:
            raise ValueError("rank/alpha patterns are not supported for spectral LoRA")
        return rank, scaling

    @torch.no_grad()
    def _initialize_exact(self, model) -> None:
        from peft.utils.other import transpose

        count, worst = 0, 0.0
        for name, layer, adapter in _linear_lora_layers(model):
            rank, scaling = self._validate_layer(layer, adapter)
            base = layer.get_base_layer()
            original = transpose(
                base.weight.detach().to(torch.float32), layer.fan_in_fan_out
            ).clone()
            if rank > min(original.shape):
                raise ValueError(f"{name}: rank {rank} exceeds min(weight.shape)={min(original.shape)}")
            U, S, Vh = torch.linalg.svd(original, full_matrices=False)
            if self.method == "pissa":
                U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
            else:
                U_r, S_r, Vh_r = U[:, -rank:], S[-rank:], Vh[-rank:, :]
            root = torch.sqrt(S_r / scaling)
            B = U_r * root.unsqueeze(0)
            A = root.unsqueeze(1) * Vh_r
            layer.lora_A[adapter].weight.copy_(A.to(layer.lora_A[adapter].weight.dtype))
            layer.lora_B[adapter].weight.copy_(B.to(layer.lora_B[adapter].weight.dtype))
            base.weight.copy_(
                transpose((original - scaling * (B @ A)).to(base.weight.dtype), layer.fan_in_fan_out)
            )
            residual = transpose(base.weight.detach().to(torch.float32), layer.fan_in_fan_out)
            A_check = layer.lora_A[adapter].weight.detach().to(torch.float32)
            B_check = layer.lora_B[adapter].weight.detach().to(torch.float32)
            check = residual + scaling * (B_check @ A_check)
            rel = float((check - original).norm() / original.norm().clamp_min(1e-30))
            worst, count = max(worst, rel), count + 1
        if count == 0:
            raise ValueError("spectral LoRA found no adapted linear layers")
        self.layer_count, self.max_init_relative_error = count, worst

    def _cache_metadata(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "algorithm": ALGORITHM,
            "method": self.method,
            "spectral_side": self.spectral_side,
            "base_model_id": self.base_model_id,
            "layer_signature": self.layer_signature,
            "training_rank": self.training_rank,
            "training_alpha": self.training_alpha,
            "layer_count": self.layer_count,
            "max_init_relative_error": self.max_init_relative_error,
        }

    @torch.no_grad()
    def _apply_cache(self, model, path: Path) -> None:
        from peft.utils.other import transpose
        from peft.utils.save_and_load import (
            get_peft_model_state_dict,
            load_peft_weights,
            set_peft_model_state_dict,
        )

        metadata_path = path / "spectral_init.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing spectral cache metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("algorithm") != ALGORITHM:
            raise ValueError(f"unsupported spectral cache contract: {metadata_path}")
        if metadata.get("method") != self.method:
            raise ValueError(f"cache method {metadata.get('method')!r} != requested {self.method!r}")
        if metadata.get("base_model_id") != self.base_model_id:
            raise ValueError(
                f"cache base model {metadata.get('base_model_id')!r} != requested {self.base_model_id!r}"
            )
        observed_signature = _layer_signature(model)
        if metadata.get("layer_signature") != observed_signature:
            raise ValueError("spectral cache adapted-layer signature does not match this model/config")
        weights = path / "adapter_model.safetensors"
        expected_sha = metadata.get("adapter_sha256")
        if not weights.is_file() or not expected_sha:
            raise ValueError(f"spectral cache is missing checksummed safetensors: {path}")
        if _file_sha256(weights) != expected_sha:
            raise ValueError(f"spectral cache checksum mismatch: {weights}")
        first = next(model.parameters())
        state = load_peft_weights(str(path), device=str(first.device))
        expected_keys = set(get_peft_model_state_dict(model, adapter_name="default"))
        if set(state) != expected_keys:
            missing = sorted(expected_keys - set(state))
            extra = sorted(set(state) - expected_keys)
            raise ValueError(
                f"spectral cache tensor coverage mismatch; missing={missing[:3]}, extra={extra[:3]}"
            )
        result = set_peft_model_state_dict(model, state, adapter_name="default")
        if getattr(result, "unexpected_keys", None):
            raise ValueError(f"unexpected spectral cache keys: {result.unexpected_keys}")

        count, worst = 0, 0.0
        for name, layer, adapter in _linear_lora_layers(model):
            rank, scaling = self._validate_layer(layer, adapter)
            base = layer.get_base_layer()
            original = transpose(
                base.weight.detach().to(torch.float32), layer.fan_in_fan_out
            ).clone()
            A = layer.lora_A[adapter].weight.detach().to(torch.float32)
            B = layer.lora_B[adapter].weight.detach().to(torch.float32)
            if A.shape[0] != rank or B.shape[1] != rank:
                raise ValueError(f"{name}: cached factor rank does not match r={rank}")
            base.weight.copy_(
                transpose((original - scaling * (B @ A)).to(base.weight.dtype), layer.fan_in_fan_out)
            )
            residual = transpose(base.weight.detach().to(torch.float32), layer.fan_in_fan_out)
            check = residual + scaling * (B @ A)
            rel = float((check - original).norm() / original.norm().clamp_min(1e-30))
            worst, count = max(worst, rel), count + 1
        if count != int(metadata.get("layer_count", -1)):
            raise ValueError("spectral cache layer coverage does not match the model")
        self.layer_count, self.max_init_relative_error = count, worst

    def initialize(self, model) -> dict:
        if self.initialized:
            raise RuntimeError("spectral runtime was initialized twice")
        self.layer_signature = _layer_signature(model)
        if self.cache_path is None:
            self._initialize_exact(model)
        else:
            self._apply_cache(model, self.cache_path)
        if self.max_init_relative_error > self.reconstruction_tolerance:
            raise RuntimeError(
                f"{self.method} reconstruction error {self.max_init_relative_error:.3e} "
                f"exceeds {self.reconstruction_tolerance:.3e}"
            )
        self._temporary = tempfile.TemporaryDirectory(prefix=f"{self.method}-initial-")
        self._initial_adapter = Path(self._temporary.name) / "initial"
        model.save_pretrained(str(self._initial_adapter), safe_serialization=True)
        self.initialized = True
        return self.metadata()

    def metadata(self) -> dict:
        data = self._cache_metadata()
        data.update({
            "checkpoint_representation": "standard_lora_difference",
            "export_rank": 2 * self.training_rank if self.training_rank is not None else None,
            "export_alpha": 2 * self.training_alpha if self.training_alpha is not None else None,
            "cache_path": str(self.cache_path) if self.cache_path is not None else None,
        })
        return data

    def save_cache(self, path: str | Path) -> Path:
        if not self.initialized or self._initial_adapter is None:
            raise RuntimeError("initialize before saving a spectral cache")
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite spectral cache: {destination}")
        shutil.copytree(self._initial_adapter, destination)
        metadata = self._cache_metadata()
        weights = destination / "adapter_model.safetensors"
        if weights.is_file():
            metadata["adapter_sha256"] = _file_sha256(weights)
        (destination / "spectral_init.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        return destination

    def save_pretrained(self, model, path: str | Path) -> Path:
        if not self.initialized or self._initial_adapter is None:
            raise RuntimeError("initialize before exporting a spectral checkpoint")
        destination = Path(path)
        # PEFT's mutated-initialization conversion loads the saved initial adapter
        # frozen, then deletes it.  Restore the original adapter's exact trainability
        # and module mode so exporting an inline checkpoint cannot freeze subsequent
        # optimization steps.
        was_training = model.training
        active_adapters = list(model.active_adapters)
        if active_adapters != ["default"]:
            raise RuntimeError(
                f"spectral export requires the single default adapter, found {active_adapters}"
            )
        trainability = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
        try:
            model.save_pretrained(
                str(destination), safe_serialization=True,
                path_initial_model_for_weight_conversion=str(self._initial_adapter),
            )
        finally:
            model.set_adapter(active_adapters[0])
            for parameter, requires_grad in trainability:
                parameter.requires_grad_(requires_grad)
            model.train(was_training)
        config = json.loads((destination / "adapter_config.json").read_text())
        if config.get("peft_type") != "LORA" or config.get("r") != 2 * self.training_rank:
            raise RuntimeError("spectral export did not produce the expected standard rank-2r LoRA adapter")
        expected_alpha = 2 * self.training_alpha
        if abs(float(config.get("lora_alpha")) - expected_alpha) > 1e-8:
            raise RuntimeError("spectral export did not preserve LoRA scaling")
        (destination / "spectral_training.json").write_text(
            json.dumps(self.metadata(), indent=2, sort_keys=True) + "\n"
        )
        return destination

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
        self._temporary = None
        self._initial_adapter = None


def save_adapter(model, path: str | Path, runtime: SpectralLoRARuntime | None = None) -> Path:
    destination = Path(path)
    if runtime is None:
        model.save_pretrained(str(destination), safe_serialization=True)
        return destination
    return runtime.save_pretrained(model, destination)

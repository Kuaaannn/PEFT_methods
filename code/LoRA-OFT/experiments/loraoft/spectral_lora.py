"""Project-level PiSSA/MiLoRA initialization and portable checkpoint export.

PEFT implements PiSSA by moving a selected rank-r SVD component from the frozen
base weight into the LoRA factors.  MiLoRA uses the identical construction with
the *minor* rather than principal singular components.  Keeping this small layer
outside PEFT has two advantages: the vendored library remains unchanged, and both
methods use one audited implementation apart from the selected SVD slice.

Training takes place against the residual base::

    W_res = W_0 - s B_0 A_0,      W(t) = W_res + s B(t) A(t)

where ``s`` is PEFT's LoRA scaling.  A residualized base is not a portable PEFT
checkpoint by itself.  ``save_pretrained`` therefore uses PEFT's official
``path_initial_model_for_weight_conversion`` path to export

    s (B(t) A(t) - B_0 A_0)

as an ordinary rank-2r LoRA adapter relative to the original pretrained base.
Standard evaluation consequently needs no spectral-method special case.
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_name(layer) -> str:
    active = list(layer.active_adapters)
    if active != ["default"]:
        raise ValueError(f"spectral LoRA requires the single default adapter, found {active}")
    return active[0]


def _layer_signature(
    model,
    *,
    rank_override: int | None = None,
    scaling_override: float | None = None,
) -> str:
    """Identify the exact adapted layout without hashing multi-GB base weights."""
    rows = []
    for name, layer, adapter in _linear_lora_layers(model):
        base = layer.get_base_layer()
        rows.append({
            "name": name,
            "shape": list(base.weight.shape),
            "fan_in_fan_out": bool(layer.fan_in_fan_out),
            "rank": int(layer.r[adapter]) if rank_override is None else rank_override,
            "scaling": (
                float(layer.scaling[adapter])
                if scaling_override is None
                else scaling_override
            ),
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
        weight = layer.get_base_layer().weight
        if weight.ndim != 2:
            raise TypeError(f"{name}: expected a matrix weight, found shape {tuple(weight.shape)}")
        yield name, layer, adapter


@dataclass
class SpectralLoRARuntime:
    """Mutable run-scoped state needed to initialize and export PiSSA/MiLoRA."""

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
    _cache_source_rank: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.method not in ("pissa", "milora"):
            raise ValueError(f"unsupported spectral LoRA method {self.method!r}")
        if self.cache_path is not None:
            self.cache_path = Path(self.cache_path)

    @property
    def spectral_side(self) -> str:
        return "principal" if self.method == "pissa" else "minor"

    def _validate_common_layer_config(self, layer, adapter: str) -> tuple[int, float]:
        rank = int(layer.r[adapter])
        scaling = float(layer.scaling[adapter])
        if rank <= 0 or scaling <= 0:
            raise ValueError(f"invalid spectral LoRA rank/scaling: r={rank}, scaling={scaling}")
        alpha = scaling * rank
        if self.training_rank is None:
            self.training_rank = rank
            self.training_alpha = alpha
        elif rank != self.training_rank or abs(alpha - float(self.training_alpha)) > 1e-8:
            raise ValueError("rank/alpha patterns are not supported for spectral LoRA runs")
        return rank, scaling

    @torch.no_grad()
    def _initialize_exact(self, model) -> None:
        from peft.utils.other import transpose

        count = 0
        worst = 0.0
        for name, layer, adapter in _linear_lora_layers(model):
            rank, scaling = self._validate_common_layer_config(layer, adapter)
            base = layer.get_base_layer()
            dtype = base.weight.dtype
            oriented = transpose(
                base.weight.detach().to(torch.float32), layer.fan_in_fan_out
            ).clone()
            if rank > min(oriented.shape):
                raise ValueError(f"{name}: rank {rank} exceeds min(weight.shape)={min(oriented.shape)}")

            U, S, Vh = torch.linalg.svd(oriented, full_matrices=False)
            if self.method == "pissa":
                U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
            else:
                U_r, S_r, Vh_r = U[:, -rank:], S[-rank:], Vh[-rank:, :]
            root = torch.sqrt(S_r / scaling)
            B = U_r * root.unsqueeze(0)
            A = root.unsqueeze(1) * Vh_r

            layer.lora_A[adapter].weight.copy_(A.to(layer.lora_A[adapter].weight.dtype))
            layer.lora_B[adapter].weight.copy_(B.to(layer.lora_B[adapter].weight.dtype))
            residual = oriented - scaling * (B @ A)
            base.weight.copy_(transpose(residual.to(dtype), layer.fan_in_fan_out))

            residual_check = transpose(base.weight.detach().to(torch.float32), layer.fan_in_fan_out)
            A_check = layer.lora_A[adapter].weight.detach().to(torch.float32)
            B_check = layer.lora_B[adapter].weight.detach().to(torch.float32)
            reconstructed = residual_check + scaling * (B_check @ A_check)
            rel = float((reconstructed - oriented).norm() / oriented.norm().clamp_min(1e-30))
            worst = max(worst, rel)
            count += 1

        if count == 0:
            raise ValueError("spectral LoRA found no adapted linear layers")
        self.layer_count = count
        self.max_init_relative_error = worst

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
        from peft.utils.save_and_load import (
            get_peft_model_state_dict,
            load_peft_weights,
            set_peft_model_state_dict,
        )
        from peft.utils.other import transpose

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
        try:
            source_rank = int(metadata["training_rank"])
            source_alpha = float(metadata["training_alpha"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"spectral cache has invalid source rank/alpha: {path}") from exc
        if source_rank <= 0 or source_alpha <= 0:
            raise ValueError(f"spectral cache has invalid source rank/alpha: {path}")
        source_scaling = source_alpha / source_rank

        # Schema-v1 caches include rank and scaling in their signature.  Validate
        # their original (bank) configuration while allowing the live adapter to
        # request a smaller rank from the same adapted matrix layout.
        observed_source_signature = _layer_signature(
            model, rank_override=source_rank, scaling_override=source_scaling
        )
        if metadata.get("layer_signature") != observed_source_signature:
            raise ValueError("spectral cache adapted-layer signature does not match this model/config")

        weights = path / "adapter_model.safetensors"
        expected_sha = metadata.get("adapter_sha256")
        if not weights.is_file() or not expected_sha:
            raise ValueError(f"spectral cache is missing checksummed safetensors: {path}")
        if _file_sha256(weights) != expected_sha:
            raise ValueError(f"spectral cache checksum mismatch: {weights}")

        first = next(model.parameters())
        layers = list(_linear_lora_layers(model))
        if not layers:
            raise ValueError("spectral LoRA found no adapted linear layers")
        target_rank = None
        target_scaling = None
        for _, layer, adapter in layers:
            rank, scaling = self._validate_common_layer_config(layer, adapter)
            target_rank = rank
            target_scaling = scaling
        assert target_rank is not None and target_scaling is not None
        if target_rank > source_rank:
            raise ValueError(
                f"spectral cache rank {source_rank} cannot initialize requested rank {target_rank}"
            )

        state = load_peft_weights(str(path), device=str(first.device))
        expected_keys = set(get_peft_model_state_dict(model, adapter_name="default"))
        if set(state) != expected_keys:
            missing = sorted(expected_keys - set(state))
            extra = sorted(set(state) - expected_keys)
            raise ValueError(
                f"spectral cache tensor coverage mismatch; missing={missing[:3]}, extra={extra[:3]}"
            )
        # The maximum-rank cache is a compact spectral bank.  PiSSA's components
        # are stored from largest downward, whereas MiLoRA's selected tail keeps
        # the SVD's descending order and therefore uses the final requested rows.
        # Rescaling makes this correct even if a future grid changes alpha/r.
        start = 0 if self.method == "pissa" else source_rank - target_rank
        stop = start + target_rank
        factor_scale = (source_scaling / target_scaling) ** 0.5
        sliced_state = {}
        for key, value in state.items():
            if key.endswith("lora_A.weight"):
                if value.ndim != 2 or value.shape[0] != source_rank:
                    raise ValueError(f"{key}: cached A factor is not rank {source_rank}")
                value = value[start:stop, :]
            elif key.endswith("lora_B.weight"):
                if value.ndim != 2 or value.shape[1] != source_rank:
                    raise ValueError(f"{key}: cached B factor is not rank {source_rank}")
                value = value[:, start:stop]
            else:
                raise ValueError(f"unexpected non-LoRA tensor in spectral cache: {key}")
            sliced_state[key] = (value * factor_scale).contiguous()

        result = set_peft_model_state_dict(model, sliced_state, adapter_name="default")
        # PEFT reports frozen base parameters as missing because an adapter checkpoint
        # intentionally contains adapter tensors only. Unexpected keys are the actual
        # incompatibility signal; factor coverage is checked explicitly below.
        if getattr(result, "unexpected_keys", None):
            raise ValueError(f"unexpected spectral cache keys: {result.unexpected_keys}")

        count = 0
        worst = 0.0
        for name, layer, adapter in layers:
            rank, scaling = self._validate_common_layer_config(layer, adapter)
            base = layer.get_base_layer()
            original = transpose(
                base.weight.detach().to(torch.float32), layer.fan_in_fan_out
            ).clone()
            A = layer.lora_A[adapter].weight.detach().to(torch.float32)
            B = layer.lora_B[adapter].weight.detach().to(torch.float32)
            if A.shape[0] != rank or B.shape[1] != rank:
                raise ValueError(f"{name}: cached factor rank does not match r={rank}")
            residual = original - scaling * (B @ A)
            base.weight.copy_(transpose(residual.to(base.weight.dtype), layer.fan_in_fan_out))
            check = transpose(base.weight.detach().to(torch.float32), layer.fan_in_fan_out) + scaling * (B @ A)
            rel = float((check - original).norm() / original.norm().clamp_min(1e-30))
            worst = max(worst, rel)
            count += 1

        expected = int(metadata.get("layer_count", -1))
        if count != expected:
            raise ValueError(f"spectral cache covers {expected} layers but model has {count}")
        self.layer_count = count
        self.max_init_relative_error = worst
        self._cache_source_rank = source_rank

    def initialize(self, model) -> dict:
        """Apply spectral initialization and preserve its initial adapter for conversion."""
        if self.initialized:
            raise RuntimeError("spectral LoRA runtime was initialized twice")
        self.layer_signature = _layer_signature(model)
        if self.cache_path is None:
            self._initialize_exact(model)
        else:
            self._apply_cache(model, self.cache_path)
        if self.max_init_relative_error > self.reconstruction_tolerance:
            raise RuntimeError(
                f"{self.method} initialization reconstruction error "
                f"{self.max_init_relative_error:.3e} exceeds {self.reconstruction_tolerance:.3e}"
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
            "cache_source_rank": self._cache_source_rank,
            "cache_rank_sliced": (
                self._cache_source_rank is not None
                and self.training_rank is not None
                and self._cache_source_rank != self.training_rank
            ),
        })
        return data

    def save_cache(self, path: str | Path) -> Path:
        if not self.initialized or self._initial_adapter is None:
            raise RuntimeError("initialize the spectral runtime before saving its cache")
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite spectral cache: {destination}")
        shutil.copytree(self._initial_adapter, destination)
        (destination / "spectral_init.json").write_text(
            json.dumps(self._cache_metadata(), indent=2, sort_keys=True) + "\n"
        )
        weights = destination / "adapter_model.safetensors"
        if weights.is_file():
            metadata = json.loads((destination / "spectral_init.json").read_text())
            metadata["adapter_sha256"] = _file_sha256(weights)
            (destination / "spectral_init.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
        return destination

    def save_pretrained(self, model, path: str | Path) -> Path:
        """Export a standard LoRA adapter relative to the untouched pretrained base."""
        if not self.initialized or self._initial_adapter is None:
            raise RuntimeError("spectral runtime must be initialized before checkpoint export")
        destination = Path(path)
        # PEFT's official mutated-initialization conversion temporarily loads the
        # initial adapter with ``is_trainable=False`` and then deletes it.  That
        # operation leaves the original adapter parameters frozen even though the
        # active adapter name is restored.  An inline checkpoint at step 125 would
        # therefore make the next loss contain no differentiable path.  Treat export
        # as a transaction: preserve both module mode and the exact trainability of
        # every pre-existing parameter, including the frozen residual base.
        was_training = model.training
        active_adapters = list(model.active_adapters)
        if active_adapters != ["default"]:
            raise RuntimeError(
                f"spectral export requires the single default adapter, found {active_adapters}"
            )
        trainability = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
        try:
            model.save_pretrained(
                str(destination),
                safe_serialization=True,
                path_initial_model_for_weight_conversion=str(self._initial_adapter),
            )
        finally:
            model.set_adapter(active_adapters[0])
            for parameter, requires_grad in trainability:
                parameter.requires_grad_(requires_grad)
            model.train(was_training)
        config = json.loads((destination / "adapter_config.json").read_text())
        if config.get("peft_type") != "LORA":
            raise RuntimeError("spectral export did not produce a standard LoRA checkpoint")
        if config.get("r") != 2 * self.training_rank:
            raise RuntimeError(f"spectral export rank is {config.get('r')}, expected {2 * self.training_rank}")
        expected_alpha = 2 * self.training_alpha
        if abs(float(config.get("lora_alpha")) - expected_alpha) > 1e-8:
            raise RuntimeError(
                f"spectral export alpha is {config.get('lora_alpha')}, expected {expected_alpha}"
            )
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
    """The only checkpoint writer training code should call."""
    destination = Path(path)
    if runtime is None:
        model.save_pretrained(str(destination), safe_serialization=True)
        return destination
    return runtime.save_pretrained(model, destination)

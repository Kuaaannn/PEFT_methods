from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .io import digest, read_json, writable_output

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MODELS = {"qwen": "Qwen/Qwen2.5-7B", "llama": "meta-llama/Meta-Llama-3.1-8B"}
METHODS = ("lora", "oft", "dora", "pissa", "milora", "hra")
SEEDS = (13, 37, 73)
LRS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)


@dataclass(frozen=True)
class RunConfig:
    task: str
    model_key: str
    model_revision: str
    snapshot: str
    method: str
    lr: float
    seed: int
    data_manifest: str
    output: str
    rank: int = 7
    block_size: int = 32
    alpha: int = 14
    epochs: int = 1
    effective_batch: int = 32
    micro_batch: int = 2
    max_length: int = 256
    warmup_steps: int = 0
    warmup_fraction: float = 0.03
    scheduler: str = "cosine"
    spectral_cache: str | None = None
    expected_trainable: int | None = None
    schema: int = 1

    def __post_init__(self):
        if self.task not in ("coding",) or self.model_key not in MODELS:
            raise ValueError("Unknown task/model")
        if self.method not in METHODS or self.schema != 1:
            raise ValueError("Unknown method/schema")
        if not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("Pin an immutable 40-character model revision, not main")
        for name in ("rank", "block_size", "alpha", "epochs", "effective_batch", "micro_batch", "max_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.effective_batch % self.micro_batch or self.max_length < 8:
            raise ValueError("Invalid accumulation/sequence length")
        if self.method == "hra" and self.rank % 2:
            raise ValueError("HRA rank must be even")
        if isinstance(self.lr, bool) or not math.isfinite(self.lr) or self.lr <= 0 or type(self.seed) is not int:
            raise ValueError("Invalid LR/seed")
        if not 0 <= self.seed < 2**32:
            raise ValueError("Seed must fit NumPy's unsigned 32-bit seed range")
        if self.expected_trainable is not None and (type(self.expected_trainable) is not int or self.expected_trainable <= 0):
            raise ValueError("expected_trainable must be a positive integer")
        if self.scheduler not in ("linear", "cosine") or not 0 <= self.warmup_fraction < 1:
            raise ValueError("Invalid scheduler/warmup")
        if self.warmup_steps < 0 or (self.warmup_steps and self.warmup_fraction):
            raise ValueError("Use either warmup steps or fraction")
        for name in ("snapshot", "data_manifest", "output"):
            if not Path(getattr(self, name)).is_absolute():
                raise ValueError(f"{name} must be absolute")
        writable_output(self.output)

    @property
    def model_id(self):
        return MODELS[self.model_key]

    @property
    def identity(self):
        return digest(asdict(self))

    def as_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path, task=None):
        config = cls(**read_json(path))
        if task is not None and config.task != task:
            raise ValueError(f"{task} entry point cannot run {config.task}")
        return config

"""Thin adapter around the repository's standard checkpoint evaluation.

An intervention changes only the installed transformer weights. Dataset construction, prompt
encoding, model dtype, sampling, seeds, DINO scoring and drift scoring all remain on the same path
used by :mod:`evaluate`. No checkpoint-analysis-specific autocast or TF32 policy is applied.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import torch

from transformers import set_seed

from data import get_train_valid_test_datasets
from evaluate import build_drift_reference, evaluate_checkpoint
from run import precompute_prompt_caches
from utils import TrainStatus, get_dino_encoder

import forgetting


class StandardEvaluator:
    """Reuse one standard-evaluation context for every weight variant of a checkpoint."""

    def __init__(self, pipeline, train_config, device: str = "cuda", print_fn=print):
        self.pipeline = pipeline
        self.config = train_config
        self.device = device
        self.print_fn = print_fn

        set_seed(train_config.seed)
        _, _, self.test_dataset = get_train_valid_test_datasets(
            train_config=train_config, print_fn=print_fn
        )
        eval_prompts = (
            [sample["prompt"] for sample in self.test_dataset]
            + list(train_config.drift_image_prompts)
            + list(train_config.sample_image_prompts)
        )
        _, _, self.prompt_cache = precompute_prompt_caches(
            pipeline,
            train_prompts=[],
            eval_prompts=eval_prompts,
            device_type=device,
            train_config=train_config,
        )
        pipeline.text_encoder = None
        self.dino_components = get_dino_encoder(
            train_config.dino_model_id, train_config.dino_image_size
        )
        self.drift_cache = forgetting.load_drift_cache(train_config, print_fn=print_fn)

    def bank_hash(self) -> str:
        """Content identity of the standard evaluation inputs and settings."""
        payload = {
            "evaluator": "evaluate.evaluate_checkpoint",
            "config": dataclasses.asdict(self.config),
            "test_prompts": [sample["prompt"] for sample in self.test_dataset],
        }
        digest = hashlib.blake2b(digest_size=16)
        digest.update(json.dumps(payload, sort_keys=True, default=str).encode())

        # Hash the actual reference images, prompt encodings, and base-drift reference rather
        # than treating a matching config as proof that cached inputs match.
        for sample in self.test_dataset:
            image = sample["raw_image"]
            digest.update(str((image.mode, image.size)).encode())
            digest.update(image.tobytes())
        self._update_tensor_tree(digest, self.prompt_cache)
        self._update_tensor_tree(digest, self.drift_cache)
        return digest.hexdigest()

    @classmethod
    def _update_tensor_tree(cls, digest, value) -> None:
        """Deterministically hash nested standard-cache values without changing them."""
        if torch.is_tensor(value):
            tensor = value.detach().contiguous().cpu()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(f"key:{key}".encode())
                cls._update_tensor_tree(digest, value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(f"sequence:{len(value)}".encode())
            for item in value:
                cls._update_tensor_tree(digest, item)
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())

    def prepare_base_reference(self) -> None:
        """Prepare drift's base comparison while base weights are installed.

        Standard evaluation normally obtains this either from its on-disk cache or by disabling a
        live PEFT adapter. Interventions are already merged weights, so the equivalent supported
        route is an in-memory standard-format cache.
        """
        if self.drift_cache is None:
            processor, dino_model = self.dino_components
            self.drift_cache = build_drift_reference(
                pipeline=self.pipeline,
                train_config=self.config,
                prompt_cache=self.prompt_cache,
                processor=processor,
                dino_model=dino_model,
            )

    def measure(self) -> dict:
        """Evaluate the currently installed weights through the standard harness unchanged."""
        if self.drift_cache is None:
            raise RuntimeError("base drift reference has not been prepared")
        set_seed(self.config.seed)
        result = evaluate_checkpoint(
            pipeline=self.pipeline,
            train_config=self.config,
            test_dataset=self.test_dataset,
            prompt_cache=self.prompt_cache,
            print_verbose=self.print_fn,
            drift_cache=self.drift_cache,
            dino_components=self.dino_components,
        )
        if result.status != TrainStatus.SUCCESS or not result.metrics:
            raise RuntimeError(result.error_msg or f"standard evaluation returned {result.status}")
        return dict(result.metrics[-1])

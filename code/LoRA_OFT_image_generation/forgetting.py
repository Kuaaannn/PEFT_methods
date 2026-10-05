# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forgetting / prior preservation metrics for the image generation benchmark.

The benchmark already measures drift, i.e. how far the images generated for concept-unrelated prompts move away from
the ones the base model generates. This module refines that in three ways:

- the drift is split by prompt into the concept class (prompts naming the trained class, e.g. "cat") and the rest, so
  that bleed into the class prior is not averaged together with a loss of general capability
- a held-out flow matching validation loss on general image/caption pairs, the closest analogue to perplexity for a
  generative image model: same objective as training, on data the model never saw, at fixed noise and fixed timesteps
  so that the estimate is low variance and comparable across runs
- a CLIP text-image alignment score on the drift images, which catches a loss of prompt following that a DINO cosine
  similarity between images cannot see

It also lets the base model images be cached instead of regenerated per run, which is what makes drift available for
full fine-tuning at all: there is no adapter to disable there.

Everything is optional. The caches are built by ``build_forgetting_caches.py``; when they are missing, ``run.py``
falls back to exactly its previous behaviour.
"""

import os
import warnings
from typing import Any, Optional

import torch
from diffusers.training_utils import compute_loss_weighting_for_sd3


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "forgetting-cache")
VAL_CACHE_FILE = os.path.join(CACHE_PATH, "val_cache.pt")

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
# the drift generator in run.py is seeded with config.seed + this offset, the cached baseline has to match it
DRIFT_SEED_OFFSET = 100_000_000
# noise for the validation loss is drawn from a fixed seed, independent of config.seed, so that the loss is a
# deterministic function of the model and stays comparable across seeds
VAL_NOISE_SEED = 12345
# fractions of the training timestep range at which the validation loss is evaluated; stratifying instead of sampling
# keeps the variance low, and reporting per bucket matters because the loss is dominated by the high noise end
VAL_TIMESTEP_FRACTIONS = (0.125, 0.375, 0.625, 0.875)


def drift_cache_file(seed: int) -> str:
    return os.path.join(CACHE_PATH, f"drift_cache_seed{seed}.pt")


def _generation_meta(config) -> dict[str, Any]:
    """The settings a cached baseline has to agree on to be comparable with a run."""
    return {
        "model_id": config.model_id,
        "dtype": config.dtype,
        "resolution": config.resolution,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "max_sequence_length": config.max_sequence_length,
        "text_encoder_out_layers": list(config.text_encoder_out_layers),
    }


def _check_meta(cached: dict[str, Any], expected: dict[str, Any], path: str) -> bool:
    mismatched = {key: (cached.get(key), value) for key, value in expected.items() if cached.get(key) != value}
    if mismatched:
        warnings.warn(f"Ignoring incompatible cache {path}, mismatching settings: {sorted(mismatched)}")
        return False
    return True


def load_drift_cache(config, *, print_fn=print) -> Optional[dict[str, Any]]:
    """Base model DINO embeddings (and CLIP scores) for the drift prompts, or None if unavailable."""
    path = drift_cache_file(config.seed)
    if not os.path.exists(path):
        return None
    cache = torch.load(path, map_location="cpu", weights_only=False)
    expected = {**_generation_meta(config), "prompts": list(config.drift_image_prompts), "seed": config.seed}
    if not _check_meta(cache["meta"], expected, path):
        return None
    print_fn(f"Using cached base model drift baseline from {path}")
    return cache


def load_val_cache(config, *, print_fn=print) -> Optional[dict[str, Any]]:
    """Pre-encoded held-out image/caption pairs for the validation loss, or None if unavailable."""
    if not os.path.exists(VAL_CACHE_FILE):
        return None
    cache = torch.load(VAL_CACHE_FILE, map_location="cpu", weights_only=False)
    expected = {
        "model_id": config.model_id,
        "dtype": config.dtype,
        "resolution": config.resolution,
        "max_sequence_length": config.max_sequence_length,
        "text_encoder_out_layers": list(config.text_encoder_out_layers),
    }
    if not _check_meta(cache["meta"], expected, VAL_CACHE_FILE):
        return None
    print_fn(f"Using held-out validation set from {VAL_CACHE_FILE} ({cache['latents'].shape[0]} pairs)")
    return cache


def derive_class_token(config) -> str:
    """The class the concept belongs to, i.e. the word following the identifier in the instance prompts.

    For the default dataset the instance prompts read "sks cat ...", so the class is "cat" and the drift prompts
    mentioning a cat measure bleed into the class prior rather than general forgetting.
    """
    prompts = config.instance_prompts
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        words = prompt.lower().split()
        if "sks" in words:
            index = words.index("sks")
            if index + 1 < len(words):
                return words[index + 1].strip(",.\"'")
    return ""


def _mentions(prompt: str, token: str) -> bool:
    return bool(token) and token in [word.strip(",.\"'") for word in prompt.lower().split()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def compute_clip_scores(images, prompts: list[str], *, model_id: str = CLIP_MODEL_ID) -> list[float]:
    """Cosine similarity between each image and its own prompt in CLIP space.

    Runs on CPU on purpose: this is called after the training memory has been measured, but full fine-tuning still
    holds the optimizer state on the accelerator at that point, so there is no room to spare for another model.
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    image_embeds = torch.nn.functional.normalize(outputs.image_embeds, dim=-1)
    text_embeds = torch.nn.functional.normalize(outputs.text_embeds, dim=-1)
    return (image_embeds * text_embeds).sum(dim=-1).tolist()


def summarize_drift(
    *,
    prompts: list[str],
    cosine_sim: torch.Tensor,
    config,
    images: Optional[list] = None,
    drift_cache: Optional[dict[str, Any]] = None,
    print_fn=print,
) -> dict[str, Any]:
    """Split the drift by prompt group and, if images are given, add the CLIP alignment of the adapted model."""
    per_prompt = {prompt: (1 - sim) / 2.0 for prompt, sim in zip(prompts, cosine_sim.tolist())}
    class_token = derive_class_token(config)
    is_class = {prompt: _mentions(prompt, class_token) for prompt in prompts}
    details: dict[str, Any] = {
        "drift class": _mean([value for prompt, value in per_prompt.items() if is_class[prompt]]),
        "drift general": _mean([value for prompt, value in per_prompt.items() if not is_class[prompt]]),
        "drift class_token": class_token,
        "drift per prompt": per_prompt,
    }

    if images is None:
        return details
    try:
        scores = compute_clip_scores(images, prompts)
    except Exception as exc:  # scoring is auxiliary, a missing CLIP checkpoint should not fail the run
        print_fn(f"CLIP scoring failed: {exc}")
        return details

    general = [score for prompt, score in zip(prompts, scores) if not is_class[prompt]]
    details["clip general"] = _mean(general)
    details["clip per prompt"] = dict(zip(prompts, scores))
    if drift_cache is not None and "clip" in drift_cache:
        base_scores = drift_cache["clip"]
        base_general = [score for prompt, score in zip(prompts, base_scores) if not is_class[prompt]]
        details["clip general base"] = _mean(base_general)
        # positive means the adapted model follows the general prompts worse than the base model did
        details["clip general drop"] = _mean(base_general) - _mean(general)
    return details


@torch.inference_mode()
def compute_val_loss(
    *,
    pipeline,
    scheduler,
    config,
    device_type: str,
    get_sigmas,
    cache: Optional[dict[str, Any]] = None,
    print_fn=print,
) -> dict[str, Any]:
    """Flow matching loss on held-out general image/caption pairs, the perplexity analogue.

    ``get_sigmas`` is passed in rather than imported so that this module stays independent of ``run.py``. ``scheduler``
    must be the untouched training copy, since the pipeline's own scheduler gets re-timestepped by image generation.
    """
    if cache is None:
        cache = load_val_cache(config, print_fn=print_fn)
    if cache is None:
        return {}

    transformer = pipeline.transformer
    num_train_timesteps = scheduler.config.num_train_timesteps
    num_samples = cache["latents"].shape[0]
    per_bucket: dict[str, float] = {}
    all_losses: list[float] = []

    for fraction in VAL_TIMESTEP_FRACTIONS:
        index = min(int(fraction * num_train_timesteps), num_train_timesteps - 1)
        generator = torch.Generator(device=device_type).manual_seed(VAL_NOISE_SEED + index)
        losses: list[float] = []
        for i in range(0, num_samples, config.batch_size):
            latents = cache["latents"][i : i + config.batch_size].to(device_type)
            prompt_embeds = cache["prompt_embeds"][i : i + config.batch_size].to(device_type)
            text_ids = cache["text_ids"][i : i + config.batch_size].to(device_type)
            current_batch_size = latents.shape[0]

            model_input_ids = pipeline._prepare_latent_ids(latents).to(latents.device)
            noise = torch.randn(latents.shape, generator=generator, device=device_type, dtype=latents.dtype)
            indices = torch.full((current_batch_size,), index, dtype=torch.long)
            timesteps = scheduler.timesteps[indices].to(device=latents.device)
            sigmas = get_sigmas(
                timesteps, scheduler, step_indices=indices, n_dim=latents.ndim, dtype=latents.dtype
            ).to(device_type)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            packed_noisy_latents = pipeline._pack_latents(noisy_latents)

            if transformer.config.guidance_embeds:
                guidance = torch.full([1], config.guidance_scale, device=device_type).expand(current_batch_size)
            else:
                guidance = None

            model_pred = transformer(
                hidden_states=packed_noisy_latents,
                timestep=timesteps / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=model_input_ids,
                return_dict=False,
            )[0]
            model_pred = model_pred[:, : packed_noisy_latents.size(1)]
            model_pred = pipeline._unpack_latents_with_ids(model_pred, model_input_ids)
            weighting = compute_loss_weighting_for_sd3(config.weighting_scheme, sigmas=sigmas)
            target = noise - latents
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1
            )
            losses.extend(loss.tolist())
        per_bucket[f"{fraction:g}"] = _mean(losses)
        all_losses.extend(losses)

    return {
        "val loss": _mean(all_losses),
        "val loss per timestep": per_bucket,
        "val loss samples": num_samples,
    }

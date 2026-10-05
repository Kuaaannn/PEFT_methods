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

"""Build the offline caches for the forgetting metrics in ``forgetting.py``.

Both caches depend only on the base model, so they are built once and reused by every experiment:

- the drift baseline: the images the *base* model generates for the drift prompts, stored as DINO embeddings and CLIP
  scores, one cache per seed. Runs then only generate their own side of the comparison, which saves a generation pass
  per PEFT run and, more importantly, gives full fine-tuning a drift number at all.
- the validation set: held-out general image/caption pairs encoded to latents and prompt embeddings, so that the
  validation loss during a run is a plain forward pass.

Usage::

    python build_forgetting_caches.py                     # both caches, seeds 0 1 2
    python build_forgetting_caches.py --only val
    python build_forgetting_caches.py --only drift --seeds 0
"""

import argparse
import dataclasses
import hashlib
import os

import huggingface_hub
import torch
from datasets import load_dataset
from diffusers.training_utils import offload_models
from tqdm import tqdm

import forgetting
from data import _build_train_pixel_values, _to_rgb
from peft.utils import infer_device
from run import _generate_images, precompute_latent_cache, precompute_prompt_caches
from utils import get_dino_embeddings, get_dino_encoder, get_pipeline, get_train_config


# a single shard of the parquet mirror of the COCO 2014 validation captions: general text-image pairs the base model
# was not fine-tuned on, and far more than the few hundred pairs the validation loss needs
VAL_DATASET_ID = "sayakpaul/coco-30-val-2014"
VAL_DATASET_FILE = "data/train-00000-of-00010-45de7542ea7caa89.parquet"
VAL_DATASET_REVISION = "abdde3200f533ddae5bed2438057f1f7ea2d5131"
VAL_DATASET_FILE_SHA256 = "9a440ab916ffa16bbfa725b9ee24dad99f2f93a73ed1a7ec4d913227d583d749"
VAL_IMAGE_COLUMN = "image"
VAL_CAPTION_COLUMN = "caption"


def load_val_pairs(num_samples: int):
    path = huggingface_hub.hf_hub_download(
        VAL_DATASET_ID,
        VAL_DATASET_FILE,
        repo_type="dataset",
        revision=VAL_DATASET_REVISION,
    )
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != VAL_DATASET_FILE_SHA256:
        raise RuntimeError(f"held-out validation shard has unexpected SHA-256: {digest.hexdigest()}")
    ds = load_dataset("parquet", data_files=path, split=f"train[:{num_samples}]")
    images = [_to_rgb(image) for image in ds[VAL_IMAGE_COLUMN]]
    captions = [str(caption) for caption in ds[VAL_CAPTION_COLUMN]]
    return images, captions


def encode_captions(pipeline, captions: list[str], config, device_type: str):
    prompt_embeds, text_ids = [], []
    with torch.no_grad(), offload_models(pipeline.text_encoder, device=device_type, offload=True):
        for caption in tqdm(captions, desc="encoding captions"):
            embeds, ids = pipeline.encode_prompt(
                prompt=caption,
                max_sequence_length=config.max_sequence_length,
                text_encoder_out_layers=config.text_encoder_out_layers,
            )
            prompt_embeds.append(embeds.to("cpu"))
            text_ids.append(ids.to("cpu"))
    return torch.cat(prompt_embeds, dim=0), torch.cat(text_ids, dim=0)


def build_drift_cache(*, pipeline, config, prompt_cache, seeds: list[int], device_type: str) -> None:
    processor, dino_model = get_dino_encoder(config.dino_model_id, config.dino_image_size)
    prompts = list(config.drift_image_prompts)
    batch_size = config.batch_size_eval

    for seed in seeds:
        seed_config = dataclasses.replace(config, seed=seed)
        images = []
        with torch.inference_mode(), offload_models(pipeline.vae, device=device_type, offload=True):
            generator = torch.Generator(device=pipeline.transformer.device).manual_seed(
                seed + forgetting.DRIFT_SEED_OFFSET
            )
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"drift baseline seed {seed}"):
                outputs = _generate_images(
                    pipeline,
                    generator=generator,
                    prompts=prompts[i : i + batch_size],
                    prompt_cache=prompt_cache,
                    config=seed_config,
                )
                images.extend(outputs.images)

        embeddings = get_dino_embeddings(images, processor, dino_model, batch_size=batch_size)
        cache = {
            "dino": embeddings.to("cpu"),
            "clip": forgetting.compute_clip_scores(images, prompts),
            "meta": {**forgetting._generation_meta(seed_config), "prompts": prompts, "seed": seed},
        }
        path = forgetting.drift_cache_file(seed)
        temporary = path + ".tmp"
        torch.save(cache, temporary)
        os.replace(temporary, path)
        print(f"Wrote {path}")


def build_val_cache(*, pipeline, config, images, prompt_embeds, text_ids, device_type: str) -> None:
    pixel_values = _build_train_pixel_values(images, config.resolution)
    latents = precompute_latent_cache(
        pipeline=pipeline, vae=pipeline.vae, pixel_values=pixel_values, train_config=config, device_type=device_type
    )
    cache = {
        "latents": latents.to("cpu"),
        "prompt_embeds": prompt_embeds,
        "text_ids": text_ids,
        "meta": {
            "model_id": config.model_id,
            "dtype": config.dtype,
            "resolution": config.resolution,
            "max_sequence_length": config.max_sequence_length,
            "text_encoder_out_layers": list(config.text_encoder_out_layers),
            "dataset_id": VAL_DATASET_ID,
            "dataset_revision": VAL_DATASET_REVISION,
            "dataset_file": VAL_DATASET_FILE,
            "dataset_file_sha256": VAL_DATASET_FILE_SHA256,
            "num_samples": latents.shape[0],
        },
    }
    temporary = forgetting.VAL_CACHE_FILE + ".tmp"
    torch.save(cache, temporary)
    os.replace(temporary, forgetting.VAL_CACHE_FILE)
    print(f"Wrote {forgetting.VAL_CACHE_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["drift", "val"], help="build only one of the two caches")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seeds to build a drift baseline for")
    parser.add_argument("--num-val-samples", type=int, default=200, help="held-out image/caption pairs")
    parser.add_argument("--config", default="", help="Training configuration with the prepared local model snapshot")
    args = parser.parse_args()

    build_drift = args.only != "val"
    build_val = args.only != "drift"
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]

    os.makedirs(forgetting.CACHE_PATH, exist_ok=True)
    # the caches are shared by all experiments, so they use the defaults without any per-experiment overrides
    config = get_train_config(args.config)
    device_type = infer_device()
    pipeline = get_pipeline(
        model_id=config.model_id,
        dtype=config.dtype,
        compile=False,
        peft_config=None,
        autocast_adapter_dtype=config.autocast_adapter_dtype,
        use_gc=config.use_gc,
        device_type=device_type,
    )

    # everything that needs the text encoder happens first, so that it can be dropped before the transformer and the
    # VAE need the memory, mirroring what run.py does
    val_images = prompt_embeds = text_ids = None
    if build_val:
        val_images, captions = load_val_pairs(args.num_val_samples)
        prompt_embeds, text_ids = encode_captions(pipeline, captions, config, device_type)
    prompt_cache = None
    if build_drift:
        *_, prompt_cache = precompute_prompt_caches(
            pipeline,
            train_prompts=[],
            eval_prompts=list(config.drift_image_prompts),
            device_type=device_type,
            train_config=config,
        )
    pipeline.text_encoder = None
    getattr(torch, device_type, torch.cuda).empty_cache()

    if build_val:
        build_val_cache(
            pipeline=pipeline,
            config=config,
            images=val_images,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            device_type=device_type,
        )
    if build_drift:
        pipeline.transformer.to(device_type)
        build_drift_cache(
            pipeline=pipeline, config=config, prompt_cache=prompt_cache, seeds=seeds, device_type=device_type
        )


if __name__ == "__main__":
    main()

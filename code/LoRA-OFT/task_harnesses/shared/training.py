"""Setup/export utilities. The two task modules own their Trainer invocation."""
from __future__ import annotations

import math
from pathlib import Path

from .data import load_bank, training_audit_errors, training_examples
from .io import atomic_json, file_hash, read_json, source_identity
from .runtime import (attach_for_training, base_model, bridge, require_cuda,
                      tokenizer_for, validate_snapshot, implementation_identity)


def assert_final_only(output):
    """Reject unintended Trainer/model snapshots; never delete user artifacts."""
    forbidden = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "pytorch_model.bin", "model.safetensors"}
    unexpected = [str(path) for path in Path(output).rglob("*")
                  if path.name.startswith("checkpoint-") or path.name in forbidden]
    if unexpected:
        raise RuntimeError(f"Unexpected intermediate/full-model checkpoints: {unexpected[:8]}")


def begin(config):
    torch = require_cuda()
    output = Path(config.output)
    complete = output / "complete.json"
    if complete.exists():
        previous = read_json(complete)
        if previous["config_hash"] != config.identity:
            raise ValueError("Output belongs to another configuration")
        raise FileExistsError(f"Training already complete: {output}; evaluate the existing adapter")
    if (output / "run.json").exists():
        raise FileExistsError("Incomplete training attempt: use a new output; no optimizer-state resume")
    from transformers import set_seed
    set_seed(config.seed)
    tokenizer = tokenizer_for(config)
    bank, rows = load_bank(config.data_manifest, config.task)
    examples, audit = training_examples(rows["train"], tokenizer, config)
    errors = training_audit_errors(config.task, audit)
    if errors:
        raise ValueError(errors)
    print("TRAINING_DATA", {k: v for k, v in audit.items() if k != "excluded"}, flush=True)
    base_identity = validate_snapshot(config)
    model, spectral, capacity = attach_for_training(config, base_model(config))
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    manifest = {"schema": 1, "config": config.as_dict(), "config_hash": config.identity,
                "source_hash": source_identity(), "bank_hash": file_hash(config.data_manifest),
                "base": base_identity, "capacity": capacity, "tokenization": audit,
                "spectral": spectral.metadata() if spectral else None,
                "implementation": implementation_identity(),
                "checkpoint_policy": "final_adapter_only_no_optimizer_snapshots",
                "evaluation_during_training": False,
                "status": "training"}
    atomic_json(output / "run.json", manifest)
    torch.cuda.reset_peak_memory_stats()
    return model, tokenizer, spectral, examples, manifest


def arguments(config, n):
    from transformers import TrainingArguments
    steps = math.ceil(n / config.effective_batch) * config.epochs
    # Match math's int(warmup_ratio * max_steps), including floor rounding.
    warmup = config.warmup_steps or int(steps * config.warmup_fraction)
    if warmup >= steps:
        raise ValueError("Warmup must be shorter than training")
    return TrainingArguments(
        output_dir=str(Path(config.output) / "trainer"),
        per_device_train_batch_size=config.micro_batch,
        gradient_accumulation_steps=config.effective_batch // config.micro_batch,
        num_train_epochs=config.epochs, learning_rate=config.lr,
        lr_scheduler_type=config.scheduler, warmup_steps=warmup,
        weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        max_grad_norm=1.0, optim="adamw_torch", bf16=True, fp16=False,
        gradient_checkpointing=False,
        save_strategy="no", eval_strategy="no", logging_strategy="steps", logging_steps=10,
        report_to="none", seed=config.seed, data_seed=config.seed,
        remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=0, dataloader_pin_memory=True,
        train_sampling_strategy="random", load_best_model_at_end=False, push_to_hub=False)


def finish(config, model, tokenizer, spectral, trainer, manifest, elapsed):
    torch = require_cuda()
    bridge()
    from loraoft.spectral_lora import save_adapter
    output = Path(config.output)
    assert_final_only(output)
    staging, final = output / "adapter.partial", output / "adapter"
    if staging.exists() or final.exists():
        raise FileExistsError("Refusing to overwrite an existing adapter")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(f"Nonfinite trained parameter; refusing a complete checkpoint: {name}")
    save_adapter(model, staging, spectral)
    tokenizer.save_pretrained(staging)
    staging.rename(final)
    assert_final_only(output)
    # Store hashes so evaluation cannot accidentally use an adapter from another attempt.
    artifacts = {p.name: file_hash(p) for p in sorted(final.iterdir()) if p.is_file()}
    atomic_json(output / "complete.json", {
        **manifest, "status": "complete", "adapter_files": artifacts,
        "optimizer_steps": trainer.state.global_step, "train_seconds": elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(0)})

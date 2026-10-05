"""The single training entry point: every method, every task, one code path.

A method-specific branch outside `loraoft.methods` is how unfair comparisons get in, so
the only place `spec.method` is inspected here is where PEFT genuinely requires it
(attaching an adapter versus unfreezing base weights).

For paired seeds, data order is
driven by an RNG keyed on the SEED ALONE, method-independent, so seed 13 presents the same
example order to LoRA and to OFT. Adapter initialisation and dropout draw from a separate
substream keyed by (method, seed). Without this split, "paired differences" are not paired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .manifest import FailureTable, RunManifest, RunResult, TrainStatus, write_parquet
from .paths import reject_quarantined
from .methods import (MethodSpec, build_method, freeze_non_target_parameters,
                      resolve_param, target_matrix_names)
from .telemetry import MatrixRecorder, StepRecorder, full_ft_matrix_telemetry
from .tracking import tracker_for_run

# Dense early checkpoints followed by regular checkpoints separate immediate
# optimizer shrinkage from learned rotation.
def checkpoint_schedule(max_steps: int, every: int = 500) -> list[int]:
    dense = [t for t in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if t <= max_steps]
    regular = list(range(every, max_steps + 1, every))
    return sorted(set(dense + regular + [max_steps]))


def data_order_generator(seed: int) -> torch.Generator:
    """Method-INDEPENDENT. Same seed => same example order for every method."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def init_seed(method: str, seed: int) -> int:
    """Deterministic substream for adapter init / dropout, keyed by (method, seed)."""
    h = hashlib.sha256(f"{method}:{seed}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def run_id_for(experiment: str, spec: MethodSpec, lr: float, seed: int, task: str,
               run_tag: str | None = None) -> str:
    """Identity of a run, and therefore the name of its directory.

    `run_tag` exists because a sweep may vary a dimension that none of the other
    arguments capture -- effective batch size, for instance. Without it two such cells
    resolve to the same directory, the second dies on the immutable manifest, and the
    sweep reads the FIRST cell's result as though the second had run.
    """
    base = f"{experiment}-{task}-{spec.label()}-lr{lr:g}-s{seed}"
    return f"{base}-{run_tag}" if run_tag else base


@torch.no_grad()
def base_frobenius(model, names: list[str]) -> dict[str, float]:
    """||W_0||_F per adapted matrix, read before the first optimizer step.

    Tier-1 telemetry reports relative update norms against these. Capturing them lazily
    (or not at all) makes every `delta_rel_fro` NaN, which is invisible until analysis.
    """
    params = dict(model.named_parameters())
    out = {}
    missing = []
    for n in names:
        p = resolve_param(params, n)
        if p is None:
            missing.append(n)
        else:
            out[n] = float(p.detach().float().norm())
    if missing:
        # Refuse rather than returning a partial map. A missing base norm does not raise
        # downstream, it turns delta_rel_fro into NaN for that matrix and stays invisible
        # until analysis.
        raise KeyError(f"no base weight found for {len(missing)} matrices, "
                       f"e.g. {missing[:3]}")
    return out


@torch.no_grad()
def capture_base_weights(model, names: list[str]) -> dict[str, torch.Tensor]:
    """fp32 CPU copy of W_0, for full fine-tuning only.

    Adapter methods reconstruct their update from the adapter itself, so they never pay
    this; for a 1.5B full-FT run it is ~5GB of host RAM, which is available here.
    """
    params = dict(model.named_parameters())
    out = {}
    for n in names:
        p = resolve_param(params, n)
        if p is not None:
            out[n] = p.detach().float().cpu().clone()
    return out


def train(
    *,
    experiment_id: str,
    task: str,
    model_id: str,
    spec: MethodSpec,
    learning_rate: float,
    seed: int,
    out_dir: Path,
    max_steps: int = 5000,
    batch_size: int = 4,
    grad_accum: int = 1,
    max_seq_length: int = 768,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.03,
    grad_norm_clip: float = 1.0,
    lr_scheduler: str = "cosine",
    dtype: str = "bfloat16",
    divergence_threshold: float = 20.0,
    eval_steps: int = 1000,
    telemetry_every: int = 50,
    checkpoint_every: int = 500,
    selection_metric: str = "valid_accuracy",
    save_adapter: bool = True,
    evaluator=None,
    max_train_examples: int | None = None,
    group_holdout: int | None = None,
    dev_subset: int | None = None,
    retention_batch_size: int = 8,
    expected_trainable_params: int | None = None,
    autocast: bool = True,
    retention_fn=None,
    save_eval_checkpoints: bool = True,
    save_full_model: bool = False,
    svd_cache=None,
    spectral_cache: Path | None = None,
    run_tag: str | None = None,
) -> RunResult:
    """Run one cell. Always writes a manifest first and a status last."""
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_constant_schedule_with_warmup,
                              get_cosine_schedule_with_warmup)
    from peft import get_peft_model

    # None disables the context entirely; otherwise the forward runs in this dtype.
    autocast_dtype = getattr(torch, dtype) if autocast else None

    run_id = run_id_for(experiment_id, spec, learning_rate, seed, task, run_tag)
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    failures = FailureTable(out_dir / "failure_table.jsonl")
    ckpt_steps = checkpoint_schedule(max_steps, checkpoint_every)

    manifest = RunManifest(
        run_id=run_id, experiment_id=experiment_id, method=spec.method,
        capacity_kind=spec.capacity_kind, capacity=spec.capacity,
        placement=spec.placement, task=task, model_id=model_id, seed=seed,
        learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size,
        grad_accum=grad_accum, max_steps=max_steps, max_seq_length=max_seq_length,
        lr_scheduler=lr_scheduler, warmup_ratio=warmup_ratio,
        grad_norm_clip=grad_norm_clip, dtype=dtype, selection_metric=selection_metric,
        divergence_threshold=divergence_threshold, checkpoint_steps=ckpt_steps,
        group_holdout=group_holdout, dev_subset=dev_subset, eval_steps=eval_steps,
        retention_batch_size=retention_batch_size,
        expected_trainable_params=expected_trainable_params,
        method_kwargs={**spec.as_row(),
                       "spectral_cache": str(spectral_cache) if spectral_cache else None},
    )
    manifest.write(run_dir / "manifest.json")
    track = tracker_for_run(manifest)

    result = RunResult(run_id=run_id, status=TrainStatus.RUNNING)
    t_start = time.perf_counter()
    torch_dtype = getattr(torch, dtype)
    spectral_runtime = None

    try:
        torch.manual_seed(init_seed(spec.method, seed))
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch_dtype, attn_implementation="sdpa")
        model.config.use_cache = False

        peft_cfg, _ = build_method(spec)
        if peft_cfg is not None:
            # HRA has one d x r parameter for every adapted matrix.  Constructing its
            # paired Householder initialization for a 7--8B model while all target
            # layers are still on CPU serializes ~80M adapter values on the login-side
            # host of the allocated node.  FLUX does not hit this path because its
            # accelerator-managed transformer is already on CUDA when adapters are
            # attached.  Move only HRA before wrapping so each adapter is born and
            # initialized on the A100; other methods retain the established loading
            # order and therefore their existing reproducibility.
            if spec.method == "hra":
                model = model.cuda()
            model = get_peft_model(model, peft_cfg,
                                   autocast_adapter_dtype=spec.autocast_adapter_dtype)
        model = model.cuda()
        if spec.method in {"pissa", "milora"}:
            from .spectral_lora import SpectralLoRARuntime
            spectral_runtime = SpectralLoRARuntime(
                spec.method, base_model_id=model_id, cache_path=spectral_cache
            )
            spectral_metadata = spectral_runtime.initialize(model)
            (run_dir / "spectral_training.json").write_text(
                json.dumps(spectral_metadata, indent=2, sort_keys=True) + "\n"
            )
        n_trainable, n_total = freeze_non_target_parameters(model, spec)
        result.trainable_params, result.total_params = n_trainable, n_total
        if (expected_trainable_params is not None
                and n_trainable != expected_trainable_params):
            raise RuntimeError(
                f"trainable parameter count {n_trainable} does not match the frozen "
                f"sweep target {expected_trainable_params}"
            )
        from .data.metamath import load_splits
        if task != "metamath":
            raise NotImplementedError(f"task {task!r} not wired yet")
        ds_train, ds_valid, ds_test, split_ids = load_splits(
            tokenizer, max_seq_length=max_seq_length,
            max_train_examples=max_train_examples, group_holdout=group_holdout)
        (run_dir / "split_ids.json").write_text(json.dumps(split_ids.__dict__))

        from functools import partial

        from .data.metamath import collate_metamath
        collator = partial(collate_metamath, tokenizer=tokenizer)
        from .data.bucketing import LengthBucketSampler
        cols = ds_train.select_columns(["input_ids", "attention_mask"])
        sampler = LengthBucketSampler(
            [len(x) for x in ds_train["input_ids"]], batch_size,
            generator=data_order_generator(seed))
        loader = DataLoader(cols, batch_sampler=sampler, collate_fn=collator,
                            num_workers=2)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        # `lr_scheduler` was recorded in the manifest but never read: every run built a
        # cosine whatever the flag said, so a run launched with `--lr-scheduler constant`
        # trained on cosine and wrote a manifest claiming otherwise. Wiring it makes the
        # recorded field true.
        #
        # `constant` exists for budget experiments. Under cosine a checkpoint at step k
        # is a hot-learning-rate point part-way through a schedule annealing to zero, NOT
        # a model trained to budget k -- measured at 1.8 pp apart in the experiment -- so each budget
        # needs its own run. Held constant, every intermediate checkpoint is a genuine
        # operating point, and with a warmup pinned to a fixed number of steps a longer
        # run reproduces a shorter one's trajectory over its whole prefix.
        n_warmup = int(warmup_ratio * max_steps)
        if lr_scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, max_steps)
        elif lr_scheduler == "constant":
            scheduler = get_constant_schedule_with_warmup(optimizer, n_warmup)
        else:
            raise ValueError(
                f"unknown lr_scheduler {lr_scheduler!r}; expected 'cosine' or 'constant'")

        # Full fine-tuning of a 1.5B keeps ~6GB of fp32 shadow copy for update tracking;
        # for adapters it is tens of MB, so it is only disabled for `full`.
        steps = StepRecorder(run_id, seed, params, track_update=spec.method != "full")
        mats = MatrixRecorder(run_id, seed, every=telemetry_every)

        # ||W_0||_F per adapted matrix, captured BEFORE the first optimizer step. Tier-1
        # reports relative update norms against it; without this every `delta_rel_fro`
        # would silently be NaN.
        target_names = target_matrix_names(model, spec)
        base_fro = base_frobenius(model, target_names)
        # Full fine-tuning has no adapter to read a delta from, so it needs the base
        # weights themselves. Adapters do not, and a 6GB fp32 copy is not free.
        w0_cache = (capture_base_weights(model, target_names)
                    if spec.method == "full" else {})
        (run_dir / "target_matrices.json").write_text(json.dumps(target_names))

        # This schedule records lightweight checkpoint metadata only. Adapter weights
        # are deliberately persisted once, at the terminal step, after training.
        ckpt_set = set(ckpt_steps)
        eval_set = set(range(eval_steps, max_steps + 1, eval_steps)) | {max_steps}
        ckpt_rows: list[dict] = []

        # Snapshot the frozen base once, so evaluation can merge for a 75x faster
        # forward and restore exactly afterwards (see loraoft.evaluate.merged_for_eval).
        eval_base_snapshot = None
        # Gated on ANY merged-context consumer, not just the generator. With accuracy
        # moved post-hoc the generator is normally None, and gating on it alone left
        # retention to be scored on an UNMERGED model -- a different set of weights from
        # the one post-hoc accuracy will describe.
        if peft_cfg is not None and (evaluator is not None or retention_fn is not None):
            from .evaluate import snapshot_base_weights
            eval_base_snapshot = snapshot_base_weights(model)

        loss_val = float("nan")
        model.train()
        step, tokens, examples = 0, 0, 0
        data_iter = iter(loader)
        while step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            # Gradient accumulation. Effective batch = batch_size x grad_accum.
            #
            # This is how the effective batch is raised, NOT a bigger micro-batch: at
            # vocab 151936 the LM-head logits alone need ~35 GB at micro-batch 32
            # (bf16 logits + fp32 cross-entropy + its gradient), and micro-batch 32 OOMs
            # on an 80 GB card. Accumulation keeps peak memory at the micro-batch 4 level
            # (~22 GB) while giving any effective batch we want.
            n_tok = 0
            examples_this_step = 0
            loss_val = 0.0
            for micro in range(grad_accum):
                if micro:                      # first micro-batch is already fetched
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        batch = next(data_iter)
                batch = {k: v.cuda() for k, v in batch.items()}
                n_tok += int(batch["attention_mask"].sum())
                examples_this_step += batch["input_ids"].shape[0]
                # Scale so the accumulated gradient is the mean over the effective batch,
                # not the sum -- otherwise the effective learning rate silently scales
                # with grad_accum and no LR comparison across batch sizes is valid.
                # Use the same autocast context for every method. Adapter parameters
                # remain FP32; eligible forward operations follow the selected dtype.
                with torch.autocast("cuda", dtype=autocast_dtype,
                                    enabled=autocast_dtype is not None):
                    loss = model(**batch).loss / grad_accum
                loss.backward()
                loss_val += float(loss.detach()) * grad_accum

            loss_val /= grad_accum
            gn_pre = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))
            gn_post = float(torch.nn.utils.clip_grad_norm_(params, grad_norm_clip)) \
                if grad_norm_clip > 0 else gn_pre
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            tokens += n_tok
            examples += examples_this_step

            t0row = steps.record(
                step=step, epoch_frac=step / max_steps, examples_seen=examples,
                tokens_seen=tokens, lr=scheduler.get_last_lr()[0],
                loss=loss_val, grad_norm_pre=gn_pre, grad_norm_post=gn_post,
                tokens_this_step=n_tok)
            track.log_step(t0row.as_row())

            # A divergence is a terminal RESULT, recorded in the failure table and shown
            # in the LR surface -- not a run to silently retry.
            if not math.isfinite(loss_val) or loss_val > divergence_threshold:
                result.status = TrainStatus.DIVERGED
                result.error_msg = f"loss {loss_val} at step {step}"
                failures.append(run_id, TrainStatus.DIVERGED, result.error_msg,
                                method=spec.method, lr=learning_rate, seed=seed, step=step)
                track.set_summary(diverged_at_step=step, diverged_loss=loss_val)
                break

            if mats.due(step):
                before = len(mats.rows)
                if peft_cfg is not None:
                    mats.record(model, step, base_fro, spec.method)
                else:
                    for r in full_ft_matrix_telemetry(model, w0_cache, target_names):
                        r.update(run_id=run_id, seed=seed, step=step,
                                 method=spec.method)
                        mats.rows.append(r)
                track.log_matrix_rows(mats.rows[before:], step)
                for r in mats.rows[before:]:
                    v = r.get("ortho_residual_norm")
                    if v is not None and v > result.ortho_residual_worst:
                        result.ortho_residual_worst = float(v)

            if step in ckpt_set:
                ckpt_rows.append({"run_id": run_id, "step": step, "tokens": tokens,
                                  "train_loss": loss_val,
                                  "lr": scheduler.get_last_lr()[0]})

            # Evaluation is scheduled INDEPENDENTLY of the checkpoint log. It used to be
            # nested inside `if step in ckpt_set`, so it fired only on the intersection
            # of the two schedules: --eval-steps 125 on a 625-step run requested
            # {125,250,375,500,625} and delivered {500,625}, because ckpt_set is dyadic
            # ({1,2,4,...,256,500,512,625}). Every run in the the experiment corpus therefore has
            # exactly two eval points, both inside the last 20% of training, and no
            # learning curve exists for any of them.
            need_eval = (evaluator is not None or retention_fn is not None
                         or (save_eval_checkpoints and step == max_steps))
            if need_eval and step in eval_set:
                t_eval = time.perf_counter()
                from .evaluate import merged_for_eval
                ev: dict[str, Any] = {}

                # Accuracy is scored POST-HOC, not here. Future runs retain exactly one
                # adapter artifact (`adapter/`) at max_steps; intermediate validation and
                # retention metrics remain in result.json without duplicating adapter
                # tensors at every eval point. eval/run_eval.py relabels this canonical
                # adapter to max_steps, so the standard merged-vLLM evaluation is
                # unchanged apart from evaluating only the requested final checkpoint.
                if save_eval_checkpoints and peft_cfg is not None and step == max_steps:
                    ev["checkpoint_path"] = "adapter"

                # Retention stays INLINE. It is forward-only and cheap, and it is the
                # only way full fine-tuning is measured at all -- its 3.1 GB checkpoints
                # cannot be kept for post-hoc scoring. Reported as an absolute NLL; the
                # delta against the base reference is formed at analysis time, so
                # changing that reference never requires retraining.
                #
                # Anything scored here runs inside the merged context, because retention
                # must describe the same weights that accuracy will describe later.
                if evaluator is not None or retention_fn is not None:
                    with merged_for_eval(model, eval_base_snapshot) as em:
                        if evaluator is not None:
                            ev.update(evaluator(em, step))
                        if retention_fn is not None:
                            ev.update(retention_fn(em))

                result.eval_time_s += time.perf_counter() - t_eval
                ev.update(run_id=run_id, seed=seed, step=step)
                result.metrics.append(ev)
                track.log_eval(ev, step)
                model.train()

        result.final_train_loss = loss_val
        result.total_tokens = tokens
        result.train_time_s = time.perf_counter() - t_start
        result.peak_memory_bytes = torch.cuda.max_memory_reserved()
        if result.status == TrainStatus.RUNNING:
            result.status = TrainStatus.SUCCESS

        # The orthogonality verdict is a RESULT, not an error: an `oft` cell whose
        # rotation angle outgrew NS5 stopped preserving singular values, and that is the
        # finding. The run stays SUCCESS and is recorded in the failure table so it can
        # never be reported as a clean spectrum-preserving arm by accident.
        from .telemetry import ORTHO_GATE
        result.ortho_gate_exceeded = result.ortho_residual_worst > ORTHO_GATE
        if result.ortho_gate_exceeded:
            failures.append(run_id, result.status, "ortho_gate_exceeded",
                            residual=result.ortho_residual_worst, gate=ORTHO_GATE,
                            method=spec.method, learning_rate=learning_rate)
            print(f"!! orthogonality gate exceeded: "
                  f"{result.ortho_residual_worst:.2e} > {ORTHO_GATE:.0e}")

        write_parquet(steps.rows, run_dir / "training_metrics.parquet")
        if ckpt_rows:
            write_parquet(ckpt_rows, run_dir / "checkpoint_index.parquet")
        if mats.rows:
            write_parquet(mats.rows, run_dir / "matrix_metrics.parquet")
        # Full fine-tuning has no adapter to snapshot, so post-hoc scoring would skip it
        # entirely and the `full` arm would be the one method evaluated differently --
        # exactly the asymmetry the merge-everything policy exists to remove. Its final
        # weights are written instead, once, and eval/run_eval.py points vLLM straight at
        # them with no merge step. This is 3.1 GB per run at 1.5B, so it is opt-in and
        # the post-hoc pass deletes each one after scoring it.
        if (save_full_model and peft_cfg is None
                and result.status == TrainStatus.SUCCESS):
            model.save_pretrained(str(run_dir / "model"), safe_serialization=True)
            tokenizer.save_pretrained(str(run_dir / "model"))

        # `save_eval_checkpoints` historically meant "write an adapter at every eval
        # step". It now means "retain the terminal adapter for post-hoc evaluation".
        # `save_adapter` keeps the same public contract. If either consumer needs the
        # weights, write one canonical copy and never create ckpt/step* duplicates.
        if ((save_adapter or save_eval_checkpoints) and peft_cfg is not None
                and result.status == TrainStatus.SUCCESS):
            from .spectral_lora import save_adapter as save_peft_adapter
            save_peft_adapter(model, run_dir / "adapter", spectral_runtime)

    except torch.cuda.OutOfMemoryError as e:
        result.status = TrainStatus.FAILED
        result.error_msg = f"OOM: {e}"
        failures.append(run_id, TrainStatus.FAILED, "oom", method=spec.method,
                        lr=learning_rate, seed=seed)
    except Exception as e:                                     # noqa: BLE001
        result.status = TrainStatus.FAILED
        result.error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        failures.append(run_id, TrainStatus.FAILED, type(e).__name__,
                        method=spec.method, lr=learning_rate, seed=seed)
    finally:
        if spectral_runtime is not None:
            spectral_runtime.close()
        result.total_time_s = time.perf_counter() - t_start
        result.write(run_dir / "result.json")
        best = None
        if result.metrics:
            vals = [m.get(selection_metric) for m in result.metrics
                    if isinstance(m.get(selection_metric), (int, float))]
            best = max(vals) if vals else None
        track.set_summary(
            status=result.status.value, error_msg=result.error_msg[:500],
            trainable_params=result.trainable_params,
            total_params=result.total_params,
            peak_memory_bytes=result.peak_memory_bytes,
            train_time_s=result.train_time_s, eval_time_s=result.eval_time_s,
            total_time_s=result.total_time_s, total_tokens=result.total_tokens,
            final_train_loss=result.final_train_loss,
            # Record the orthogonality check alongside accuracy.
            ortho_residual_worst=result.ortho_residual_worst,
            ortho_gate_exceeded=result.ortho_gate_exceeded,
            **({f"best_{selection_metric}": best} if best is not None else {}))
        track.upload_artifacts(run_dir, name=f"{run_id}-artifacts")
        # A diverged run is a completed experiment, not a crashed job, so it exits 0.
        track.finish(status=result.status.value,
                     exit_code=1 if result.status is TrainStatus.FAILED else 0)

    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Run one training cell.")
    p.add_argument("--experiment", default="math")
    p.add_argument("--task", default="metamath")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--method", required=True,
                   choices=["lora", "dora", "pissa", "milora", "hra", "oft"])
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--alpha", type=int, default=None)
    p.add_argument("--spectral-cache", type=Path, default=None,
                   help="optional validated PiSSA/MiLoRA initial-adapter cache")
    p.add_argument("--placement", default="all-linear")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="results/math")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--max-train-examples", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-seq-length", type=int, default=768)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--no-eval", action="store_true",
                   help="skip in-process validation (training/telemetry only)")
    p.add_argument("--group-holdout", type=int, default=None,
                   help="dev problems whose MetaMathQA rewrites are removed from "
                        "training. REQUIRED for a scientific run: without it the "
                        "validation set is 98.8%% inside the training data.")
    p.add_argument("--run-tag", default=None,
                   help="suffix appended to the run id; distinguishes cells that differ "
                        "in a dimension the id would not otherwise capture")
    p.add_argument("--dev-subset", type=int, default=300,
                   help="dev items scored during training; the full set is scored "
                        "post-hoc for selection")
    p.add_argument("--retention", action="store_true",
                   help="score the frozen general-text bank at every eval point")
    p.add_argument("--retention-batch-size", type=int, default=8,
                   help="micro-batch for retention NLL; lower this for larger models")
    p.add_argument("--expected-trainable-params", type=int, default=None,
                   help="abort before training if the instantiated adapter count differs")
    # Preregistered defaults, but reachable: a value recorded in the manifest that
    # nothing can set is a value nobody has actually chosen.
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--grad-norm-clip", type=float, default=1.0)
    p.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "constant"],
                   help="constant is for budget experiments: every intermediate "
                        "checkpoint is then a model trained to that budget, not a "
                        "mid-anneal point")
    p.add_argument("--divergence-threshold", type=float, default=20.0)
    p.add_argument("--telemetry-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--selection-metric", default="valid_accuracy")
    p.add_argument("--no-save-adapter", action="store_true")
    p.add_argument("--save-full-model", action="store_true",
                   help="full-FT only: write final weights (3.1 GB at 1.5B) so the post-hoc "
                        "vLLM pass can score it like every other method")
    p.add_argument("--no-eval-checkpoints", action="store_true",
                   help="do not retain the final adapter for post-hoc vLLM scoring. "
                        "Intermediate adapter snapshots are never written; combine with "
                        "--no-save-adapter to suppress the final adapter entirely")
    p.add_argument("--no-autocast", action="store_true",
                   help="disable the shared torch.autocast context for forward passes; "
                        "adapter parameter storage remains unchanged.")
    p.add_argument("--inline-gen-eval", action="store_true",
                   help="also generate in-process with transformers during training. "
                        "OFF by default: it cost 57.5 min/run for OFT against 5.0 for "
                        "LoRA, making eval time method-dependent. Accuracy normally "
                        "comes from the post-hoc vLLM pass instead.")
    p.add_argument("--allow-contaminated-dev", action="store_true",
                   help="opt out of --group-holdout. Only for reproducing PEFT's "
                        "harness; never for a scientific cell.")
    args = p.parse_args()
    reject_quarantined(args.out)

    spec = MethodSpec(method=args.method, placement=args.placement, r=args.rank,
                      alpha=args.alpha, block_size=args.block_size)

    if args.group_holdout is None and not args.allow_contaminated_dev:
        raise SystemExit(
            "Refusing to run without --group-holdout.\n"
            "MetaMathQA is ~28 rewrites of each GSM8K-train problem, so a validation "
            "sample drawn from GSM8K train is 98.8% inside the training data and "
            "selecting on it selects for memorisation. Pass --group-holdout 1000, or "
            "--allow-contaminated-dev if you are deliberately reproducing PEFT's harness."
        )

    evaluator = retention_fn = svd_cache = None
    if not args.no_eval:
        from transformers import AutoTokenizer
        from .data.metamath import load_splits
        from .evaluate import gsm8k_evaluator
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _, ds_valid, _, _ = load_splits(tok, max_seq_length=args.max_seq_length,
                                        print_fn=lambda *_: None,
                                        group_holdout=args.group_holdout)
        # Two-tier: a subset scored during training for the curve, the full set scored
        # post-hoc for selection. Scoring 1000 items x 5 points with transformers
        # generate was half the wall clock of an OFT run.
        n = min(args.dev_subset, len(ds_valid))
        if args.inline_gen_eval:
            evaluator = gsm8k_evaluator(tok, ds_valid["query"][:n],
                                        ds_valid["response"][:n])

        if args.retention:
            from .data.retention import load_bank
            from .evaluate import token_nll
            bank = load_bank()
            rows, digest = bank["rows"], bank["sha256"]

            def retention_fn(model, _rows=rows, _tok=tok, _d=digest):
                return {"retention_nll": token_nll(model, _tok, _rows, max_length=768,
                                                   batch_size=args.retention_batch_size),
                        "retention_bank_sha": _d[:12]}


    res = train(experiment_id=args.experiment, task=args.task, model_id=args.model,
                spec=spec, learning_rate=args.lr, seed=args.seed,
                out_dir=Path(args.out), max_steps=args.max_steps,
                max_train_examples=args.max_train_examples,
                batch_size=args.batch_size, weight_decay=args.weight_decay,
                max_seq_length=args.max_seq_length, eval_steps=args.eval_steps,
                grad_accum=args.grad_accum, group_holdout=args.group_holdout,
                dev_subset=args.dev_subset,
                retention_batch_size=args.retention_batch_size,
                expected_trainable_params=args.expected_trainable_params,
                autocast=not args.no_autocast,
                run_tag=args.run_tag,
                retention_fn=retention_fn, svd_cache=svd_cache,
                spectral_cache=args.spectral_cache,
                save_eval_checkpoints=not args.no_eval_checkpoints,
                save_full_model=args.save_full_model,
                dtype=args.dtype, warmup_ratio=args.warmup_ratio,
                grad_norm_clip=args.grad_norm_clip, lr_scheduler=args.lr_scheduler,
                divergence_threshold=args.divergence_threshold,
                telemetry_every=args.telemetry_every,
                checkpoint_every=args.checkpoint_every,
                selection_metric=args.selection_metric,
                save_adapter=not args.no_save_adapter,
                evaluator=evaluator)
    print(f"{res.run_id}: {res.status.value} loss={res.final_train_loss:.4f} "
          f"time={res.train_time_s:.0f}s params={res.trainable_params}")
    if res.status is TrainStatus.FAILED:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()

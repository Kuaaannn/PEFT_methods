"""Post-hoc accuracy scoring: merge every checkpoint, score them all with one vLLM engine.

Runs in the TRAINING venv, because merging needs PEFT and the eval venv deliberately does
not have it. Generation happens in the EVAL venv, in the persistent worker
(`eval/server.py`); the two talk over a filesystem queue and never import each other.

Why accuracy moved out of the training loop
-------------------------------------------
In earlier runs, in-process `transformers.generate` cost 57.5 min per OFT run against
5.0 min per LoRA run -- eval time was a function of the method under test, and it was 36%
of every OFT run's wall clock. Scoring post-hoc, from merged weights, under one engine,
makes the cost identical for every method and pays the engine's 40-60 s startup once
instead of once per run.

Fairness
--------
Every method is merged before scoring, including LoRA. vLLM's native multi-LoRA serving
would be faster, but it evaluates LoRA unmerged (W_0 x + BA x) while OFT must be merged,
which puts different numerics on different arms inside the headline comparison. Merging
everything costs speed and buys the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

from loraoft.paths import reject_quarantined
from loraoft.eval_settings import (GSM8K_DEV_N, GSM8K_MAX_LENGTH,
                                    GSM8K_MAX_NEW_TOKENS)


def dev_prompts(model_id: str, group_holdout: int | None, n: int,
                max_seq_length: int = 768):
    """Rebuild the dev split exactly as training saw it.

    Derived from the manifest rather than from anything the run wrote, so a checkpoint
    can be rescored years later with no dependency on the run directory's contents.
    """
    from transformers import AutoTokenizer

    from loraoft.data.metamath import QUERY_TEMPLATE, load_splits

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # valid_size must be raised with n: load_splits defaults to 500, so asking for 1000
    # items would silently return 500 and report a tighter interval than it earned.
    _, ds_valid, _, _ = load_splits(tok, max_seq_length=max_seq_length,
                                    valid_size=n, print_fn=lambda *_: None,
                                    group_holdout=group_holdout)
    if len(ds_valid) < n:
        print(f"warning: asked for {n} dev items, split yielded {len(ds_valid)}")
    k = min(n, len(ds_valid))
    queries = list(ds_valid["query"][:k])
    golds = list(ds_valid["response"][:k])
    return [QUERY_TEMPLATE.format(query=q) for q in queries], golds


def submit(queue: Path, model_path: Path, prompts: list[str], *,
           max_new_tokens: int, max_length: int, timeout_s: float) -> list[dict]:
    """Hand one job to the worker and block until it answers."""
    job_id = uuid.uuid4().hex[:12]
    result_path = queue / f"{job_id}.result.json"
    job = {"model_path": str(model_path), "prompts": prompts,
           "max_new_tokens": max_new_tokens, "max_length": max_length,
           "result_path": str(result_path)}
    tmp = queue / f"{job_id}.tmp"
    tmp.write_text(json.dumps(job))
    tmp.rename(queue / f"{job_id}.job.json")          # atomic: never a half-written job

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if result_path.exists():
            out = json.loads(result_path.read_text())
            result_path.unlink()
            if out.get("status") != "ok":
                raise RuntimeError(f"worker error: {out.get('error')}")
            tier = out.get("swap_tier")
            if tier == "rebuild":
                # A silent fall-through here costs ~40 s of engine startup per
                # checkpoint -- ~13 h across a full sweep -- while still returning
                # correct answers, so it must be loud.
                print(f"  !! weight swap fell back to a full engine rebuild; "
                      f"{'; '.join(out.get('swap_errors') or [])}")
            return out["rows"], tier
        time.sleep(1.0)
    raise TimeoutError(
        f"no result after {timeout_s:.0f}s. Is the worker running in the eval venv?\n"
        f"  .venv-eval/bin/python -m eval.server --base-model <id> --queue {queue}")


def checkpoints_of(run: Path) -> list[tuple[int, Path]]:
    """(step, path) for every adapter snapshot, plus the final adapter as its own point."""
    out = []
    ck = run / "ckpt"
    if ck.is_dir():
        for d in sorted(ck.glob("step*")):
            if (d / "adapter_config.json").exists():
                out.append((int(d.name[4:]), d))
    # `adapter/` is the final weights, and eval_set always contains max_steps, so when
    # snapshots exist the last one IS `adapter/`. Adding both would score the final point
    # twice and emit two rows with the same step -- which silently doubles that point's
    # weight in any per-step aggregate. Only fall back to `adapter/` when there are no
    # snapshots at all (e.g. a run trained with --no-eval-checkpoints).
    final = run / "adapter"
    if not out and (final / "adapter_config.json").exists():
        out.append((-1, final))                       # -1: relabelled to max_steps below
    # Full fine-tuning has no adapter: its final weights ARE the model, so they are
    # scored directly. Without this the `full` arm would be silently absent from the
    # post-hoc table while every other method appeared, which reads as "full FT was
    # evaluated and lost" rather than "full FT was never evaluated".
    full = run / "model"
    if not out and (full / "config.json").exists():
        out.append((-1, full))
    return sorted(out)


def score(rows: list[dict], golds: list[str]) -> dict:
    from loraoft.data.metamath import is_correct

    ok = [is_correct(r["text"], g) for r, g in zip(rows, golds)]
    n = len(ok)
    trunc = sum(r["truncated"] for r in rows)
    untr = [c for c, r in zip(ok, rows) if not r["truncated"]]
    return {
        "eval_n": n,
        "eval_accuracy": sum(ok) / n if n else float("nan"),
        "eval_n_correct": sum(ok),
        "eval_truncation_rate": trunc / n if n else float("nan"),
        "eval_accuracy_untruncated": (sum(untr) / len(untr) if untr else float("nan")),
        "eval_mean_gen_tokens": (sum(r["n_generated_tokens"] for r in rows) / n
                                 if n else float("nan")),
    }


def shard_items(items, index: int, count: int):
    """Round-robin partition with a size difference of at most one."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"invalid shard {index}/{count}")
    return items[index::count]


def _shard_part_path(runs_root: Path, group: str, index: int, count: int) -> Path:
    return runs_root / f"eval_metrics.{group}.shard-{index:02d}-of-{count:02d}.parquet"


def _shard_done_path(runs_root: Path, group: str, index: int, count: int) -> Path:
    return runs_root / f"eval_metrics.{group}.shard-{index:02d}-of-{count:02d}.done"


def _output_path(a, runs_root: Path) -> Path:
    if a.out:
        return Path(a.out)
    if a.num_shards > 1:
        return _shard_part_path(runs_root, a.shard_group, a.shard_index,
                                a.num_shards)
    return runs_root / "eval_metrics.parquet"


def _flush(rows_out, a, runs_root, quiet=False):
    """Merge rows into the parquet. Safe to call repeatedly mid-sweep."""
    import pandas as pd
    from loraoft.manifest import write_parquet
    if not rows_out:
        return
    out = _output_path(a, runs_root)
    new = pd.DataFrame(rows_out)
    if out.exists():
        prior = pd.read_parquet(out)
        merged = (pd.concat([prior, new], ignore_index=True)
                    .drop_duplicates(subset=["run_id", "step"], keep="last"))
    else:
        merged = new
    write_parquet(merged.to_dict("records"), out)
    if not quiet:
        print(f"\nwrote {len(merged)} rows -> {out}")


def _load_reusable_rows(a, runs_root: Path) -> dict[tuple[str, int], dict]:
    """Load compatible completed rows for this shard from current and prior runs."""
    import pandas as pd

    paths = []
    current = _output_path(a, runs_root)
    if current.exists():
        paths.append(current)
    if a.reuse_shard_group:
        paths.extend(sorted(runs_root.glob(
            f"eval_metrics.{a.reuse_shard_group}.shard-*.parquet")))
    paths = list(dict.fromkeys(paths))
    if not paths:
        return {}

    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        missing = {"run_id", "step", "eval_n"} - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path} lacks reusable columns: {sorted(missing)}")
        frames.append(frame.loc[frame["eval_n"] == a.dev_n])
    merged = (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["run_id", "step"], keep="last"))
    return {(str(row["run_id"]), int(row["step"])): row
            for row in merged.to_dict("records")}




def _best_lr_summaries(scores):
    """Return final-checkpoint summaries without pooling distinct capacities."""
    final_scores = scores.loc[scores.groupby("run_id")["step"].idxmax()]
    summaries = {}
    capacity_counts = final_scores.groupby("method")["capacity"].nunique()
    for (method, capacity), rows in final_scores.groupby(["method", "capacity"]):
        means = rows.groupby("learning_rate")["eval_accuracy"].mean()
        best_lr = means.idxmax()
        kind = "b" if method == "oft" else "r"
        prefix = f"{method}_{kind}{int(capacity)}"
        summaries[f"{prefix}/best_lr"] = float(best_lr)
        summaries[f"{prefix}/best_mean_accuracy"] = float(means.loc[best_lr])
        if capacity_counts.loc[method] == 1:
            summaries[f"{method}/best_lr"] = float(best_lr)
            summaries[f"{method}/best_mean_accuracy"] = float(means.loc[best_lr])
    pooled_aliases = {method for method, count in capacity_counts.items() if count > 1}
    return summaries, pooled_aliases


def _finalize_shards(a, runs_root: Path) -> Path | None:
    """Merge one array generation exactly once after every shard has finished."""
    import fcntl
    import pandas as pd

    from loraoft.manifest import write_parquet

    parts = [_shard_part_path(runs_root, a.shard_group, i, a.num_shards)
             for i in range(a.num_shards)]
    done = [_shard_done_path(runs_root, a.shard_group, i, a.num_shards)
            for i in range(a.num_shards)]
    lock_path = runs_root / f".eval-finalize-{a.shard_group}.lock"
    marker = runs_root / f"eval_metrics.{a.shard_group}.complete"
    canonical = runs_root / "eval_metrics.parquet"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing_done = [p.name for p in done if not p.exists()]
        if missing_done:
            print(f"{len(missing_done)} evaluation shards still running; final merge deferred")
            return None
        missing = [p.name for p in parts if not p.exists()]
        if missing:
            print(f"{len(missing)} evaluation shards still running; final merge deferred")
            return None

        try:
            expected_rows = sum(int(p.read_text().strip().removeprefix("rows="))
                                for p in done)
        except ValueError as exc:
            raise RuntimeError("invalid evaluation shard completion marker") from exc
        if marker.exists() and canonical.exists():
            canonical_rows = len(pd.read_parquet(canonical))
            if canonical_rows == expected_rows:
                return canonical
            print(f"stale canonical evaluation table has {canonical_rows} rows; "
                  f"rebuilding expected {expected_rows}")

        merged = (pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
                    .drop_duplicates(subset=["run_id", "step"], keep="last"))
        if len(merged) != expected_rows:
            raise RuntimeError(
                f"shard merge has {len(merged)} unique rows; done markers report "
                f"{expected_rows}")
        if a.final_only:
            expected = sum((d / "manifest.json").exists()
                           for d in runs_root.iterdir() if d.is_dir())
            if len(merged) != expected:
                raise RuntimeError(
                    f"shard merge has {len(merged)} final rows; expected {expected}")
        write_parquet(merged.to_dict("records"), canonical)
        print(f"merged {len(parts)} shards / {len(merged)} rows -> {canonical}")
        marker.write_text(f"rows={len(merged)}\n")
        return canonical


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, help="directory of run directories")
    p.add_argument("--queue", default="/dev/shm/loraoft-queue")
    p.add_argument("--scratch", default="/dev/shm/loraoft-merged")
    p.add_argument("--dev-n", type=int, default=GSM8K_DEV_N,
                   help="dev items to score; 1000 is the selection set (n=300 gives a "
                        "2.1pp binomial SD, larger than the effects under test)")
    p.add_argument("--max-new-tokens", type=int, default=GSM8K_MAX_NEW_TOKENS)
    p.add_argument("--max-length", type=int, default=GSM8K_MAX_LENGTH)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--limit", type=int, default=None, help="score only the first N runs")
    p.add_argument("--num-shards", type=int, default=1,
                   help="split sorted run directories across this many workers")
    p.add_argument("--shard-index", type=int, default=0,
                   help="zero-based worker index within --num-shards")
    p.add_argument("--shard-group", default=None,
                   help="unique array/submission ID used to isolate shard outputs")
    p.add_argument("--final-only", action="store_true",
                   help="score only each run's final adapter")
    p.add_argument("--reuse-shard-group", default=None,
                   help="reuse matching run/step rows from an earlier shard group; "
                        "the evaluation settings must be identical")
    p.add_argument("--prune", action="store_true",
                   help="delete each intermediate adapter snapshot once it has been "
                        "scored. The full sweep writes ~53.6 GB of snapshots against "
                        "~24 GB free, so keeping them all is not an option; pruning as "
                        "we score bounds peak disk instead of the total. The FINAL "
                        "adapter is always kept -- it is the run's artefact.")
    p.add_argument("--out", default=None, help="parquet path (default: <runs>/eval_metrics.parquet)")
    a = p.parse_args()

    if a.num_shards < 1 or not 0 <= a.shard_index < a.num_shards:
        p.error("--shard-index must be in [0, --num-shards)")
    if a.num_shards > 1:
        if a.out:
            p.error("parallel evaluation derives one --out per shard; do not pass --out")
        if not a.shard_group or not all(c.isalnum() or c in "-_" for c in a.shard_group):
            p.error("parallel evaluation requires a filesystem-safe --shard-group")
    if a.reuse_shard_group and not all(
            c.isalnum() or c in "-_" for c in a.reuse_shard_group):
        p.error("--reuse-shard-group must be filesystem-safe")

    runs_root = reject_quarantined(a.runs)
    queue = Path(a.queue); queue.mkdir(parents=True, exist_ok=True)
    scratch = Path(a.scratch); scratch.mkdir(parents=True, exist_ok=True)

    all_run_dirs = sorted(d for d in runs_root.iterdir()
                          if d.is_dir() and (d / "manifest.json").exists())
    if a.limit:
        all_run_dirs = all_run_dirs[:a.limit]
    run_dirs = shard_items(all_run_dirs, a.shard_index, a.num_shards)
    print(f"{len(run_dirs)}/{len(all_run_dirs)} runs under {runs_root} "
          f"(shard {a.shard_index + 1}/{a.num_shards})")

    reusable_rows = _load_reusable_rows(a, runs_root)
    if reusable_rows:
        print(f"found {len(reusable_rows)} reusable evaluation rows")
    prompt_cache: dict[tuple, tuple[list[str], list[str]]] = {}
    pools: dict[str, BaseModelPool] = {}
    rows_out: list[dict] = []
    for i, run in enumerate(run_dirs, 1):
        man = json.loads((run / "manifest.json").read_text())
        cks = checkpoints_of(run)
        if a.final_only:
            cks = cks[-1:] if cks else []
        if not cks:
            print(f"[{i}/{len(run_dirs)}] {run.name}: no adapter -- skipped "
                  f"(method={man['method']})")
            continue

        for step, ck_path in cks:
            result_step = man["max_steps"] if step == -1 else step
            reusable = reusable_rows.get((man["run_id"], result_step))
            if reusable is not None:
                row = dict(reusable)
                rows_out.append(row)
                _flush([row], a, runs_root, quiet=True)
                print(f"[{i}/{len(run_dirs)}] {run.name} step{result_step}: "
                      "reused existing result")
                continue

            key = (man["model_id"], man.get("group_holdout"), a.dev_n)
            if key not in prompt_cache:
                prompt_cache[key] = dev_prompts(
                    man["model_id"], man.get("group_holdout"), a.dev_n,
                    man.get("max_seq_length", 768))
            prompts, golds = prompt_cache[key]
            merged = scratch / f"{run.name}-step{step}"
            t0 = time.perf_counter()
            is_adapter = (ck_path / "adapter_config.json").exists()
            try:
                if is_adapter:
                    base_id = man["model_id"]
                    if base_id not in pools:
                        pools[base_id] = BaseModelPool(base_id)
                    pools[base_id].merge_into(ck_path, merged)
                else:
                    merged = ck_path                  # already a plain HF model
                gen, tier = submit(queue, merged, prompts,
                                   max_new_tokens=a.max_new_tokens,
                                   max_length=a.max_length, timeout_s=a.timeout)
                row = score(gen, golds)
                row["swap_tier"] = tier
            finally:
                # Never accumulate 3.1 GB models -- but only delete what we created.
                if is_adapter:
                    shutil.rmtree(merged, ignore_errors=True)
            row.update(run_id=man["run_id"], method=man["method"],
                       capacity=man["capacity"], learning_rate=man["learning_rate"],
                       seed=man["seed"], step=result_step,
                       eval_wall_s=time.perf_counter() - t0)
            rows_out.append(row)
            print(f"[{i}/{len(run_dirs)}] {run.name} step{row['step']}: "
                  f"acc {row['eval_accuracy']:.4f} n={row['eval_n']} "
                  f"({row['eval_wall_s']:.0f}s)")

            # Persist every expensive score so a retry can skip it immediately.
            _flush([row], a, runs_root, quiet=True)

            # Prune only INTERMEDIATE snapshots under ckpt/, never `adapter/`: the final
            # adapter is the run's artefact and must survive for any later analysis.
            if a.prune and is_adapter and ck_path.parent.name == "ckpt":
                shutil.rmtree(ck_path, ignore_errors=True)

            # Full-FT weights are 3.1 GB EACH and, unlike an adapter, are not a compact
            # artefact worth keeping: they are the whole model, and they are exactly
            # reproducible from the manifest. Twelve of them is 37 GB against ~27 GB
            # free, so they are reclaimed as soon as they have been scored.
            if a.prune and not is_adapter:
                shutil.rmtree(ck_path, ignore_errors=True)

    out = _output_path(a, runs_root)
    if rows_out:
        import pandas as pd

        from loraoft.manifest import write_parquet

        # MERGE with what is already there. The grid runs in waves and this pass is run
        # once per wave, so writing fresh would silently discard every earlier wave's
        # results -- the table would always contain only the last wave. Keyed on
        # (run_id, step), last write wins, so re-scoring a cell updates it in place.
        new = pd.DataFrame(rows_out)
        if out.exists():
            prior = pd.read_parquet(out)
            before = len(prior)
            merged = (pd.concat([prior, new], ignore_index=True)
                        .drop_duplicates(subset=["run_id", "step"], keep="last"))
            print(f"\nmerged {len(new)} new rows into {before} existing "
                  f"-> {len(merged)} total")
        else:
            merged = new
            print(f"\nwrote {len(merged)} rows")
        write_parquet(merged.to_dict("records"), out)
        print(f"-> {out}")
    else:
        print("\nno rows produced")

    if a.num_shards > 1:
        import pandas as pd

        done = _shard_done_path(runs_root, a.shard_group, a.shard_index,
                                a.num_shards)
        tmp = done.with_name(f".{done.name}.{uuid.uuid4().hex}.tmp")
        persisted_rows = len(pd.read_parquet(out)) if out.exists() else 0
        tmp.write_text(f"rows={persisted_rows}\n")
        tmp.replace(done)
        _finalize_shards(a, runs_root)


class BaseModelPool:
    """Hold one base model in memory and re-use it for every merge.

    `merge_and_unload()` mutates the base in place, so a naive loop reloads 3.1 GB from
    disk per checkpoint. The full sweep is ~1185 checkpoints; at ~40 s of load and
    construction each that is ~13 h of pure model loading. Instead the pristine base
    weights are snapshotted once on CPU and restored after each merge.
    """

    def __init__(self, base_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base_id = base_id
        self.model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(base_id)
        # bf16 CPU copy; restoring by copy is exact, unlike unmerge_adapter(), which was
        # measured to leave 7.2e-3 relative error in the base weights after one round trip.
        self.pristine = {k: v.detach().clone()
                         for k, v in self.model.state_dict().items()}

    def merge_into(self, adapter_dir: Path, out_dir: Path) -> Path:
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(self.model, str(adapter_dir))
        merged = peft_model.merge_and_unload()
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out_dir), safe_serialization=True)
        self.tokenizer.save_pretrained(str(out_dir))
        # A merged model must never be mistaken for an adapter downstream: vLLM handed a
        # PEFT directory silently loads the BASE model and returns plausible untuned
        # numbers with no error.
        assert not (out_dir / "adapter_config.json").exists()
        self.restore()
        return out_dir

    def restore(self) -> None:
        """Put the base back exactly as loaded, so the next merge starts clean."""
        import torch

        with torch.no_grad():
            sd = self.model.state_dict()
            for k, v in self.pristine.items():
                sd[k].copy_(v)


if __name__ == "__main__":
    main()

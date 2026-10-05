"""Development and selected-test evaluation through the original evaluators."""
from __future__ import annotations
import argparse
import contextlib
import gzip
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

from common import FLUX, HARNESS, LLM, ROOT, environment, evaluation_identity, execute, python, read, run_directory, sha, shared_options, validated_metrics, write
from grid import filter_options, filtered


@contextlib.contextmanager
def math_worker(snapshot, scratch):
    with tempfile.TemporaryDirectory(prefix="math-eval-", dir=scratch) as temporary:
        temp = Path(temporary)
        queue = temp / "queue"; queue.mkdir()
        command = [str(python("eval")), "-m", "eval.server", "--base-model", snapshot, "--queue", str(queue)]
        with (temp / "worker.log").open("w") as log:
            process = subprocess.Popen(command, cwd=LLM, env=environment("eval"), stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                deadline = time.monotonic() + 900
                while not (queue / "READY").exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("vLLM initialization failed.\n" + (temp / "worker.log").read_text()[-12000:])
                    time.sleep(1)
                yield queue, temp
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()


def math_evaluate(a, rows):
    from datasets import load_dataset
    from eval.run_eval import BaseModelPool, dev_prompts, score, submit
    from loraoft.data.metamath import QUERY_TEMPLATE, is_correct, parse_answer
    from loraoft.eval_settings import GSM8K_MAX_LENGTH, GSM8K_MAX_NEW_TOKENS
    root = a.output.resolve()
    assets = read(root / "data/assets.json")
    scratch = root / "scratch"; scratch.mkdir(exist_ok=True)
    for model in sorted({r["model"] for r in rows}):
        snapshot = assets[model]
        if a.split == "dev":
            prompts, golds = dev_prompts(snapshot, 1000, 1000, 768)
        else:
            data = load_dataset("openai/gsm8k", "main", revision="740312add88f781978c0658806c59bc2815b9866", split="test")
            if len(data) != 1319:
                raise ValueError("Official GSM8K test must contain 1,319 examples")
            prompts = [QUERY_TEMPLATE.format(query=r["question"]) for r in data]
            golds = [r["answer"] for r in data]
        pool = BaseModelPool(snapshot)
        with math_worker(snapshot, scratch) as (queue, temp):
            for cell in [r for r in rows if r["model"] == model]:
                run = run_directory(root, cell)
                adapter = run / "adapter"
                identity = dict(**evaluation_identity(root, cell, a.split),
                                n=len(golds), max_new_tokens=GSM8K_MAX_NEW_TOKENS,
                                max_total_tokens=GSM8K_MAX_LENGTH)
                if identity["model_snapshot"] != snapshot:
                    raise ValueError("Prepared model differs from the training snapshot")
                out = root / cell["task"] / cell["name"] / a.split
                if (out / "metrics.json").exists():
                    if read(out / "metrics.json")["identity"] != identity:
                        raise ValueError(f"Evaluation identity changed: {out}")
                    continue
                merged = temp / cell["name"]
                try:
                    pool.merge_into(adapter, merged)
                    generated, tier = submit(queue, merged, prompts, max_new_tokens=GSM8K_MAX_NEW_TOKENS,
                                             max_length=GSM8K_MAX_LENGTH, timeout_s=7200)
                finally:
                    if merged.exists():
                        shutil.rmtree(merged)
                if len(generated) != len(golds):
                    raise RuntimeError("Incomplete generation results")
                if a.split == "test" and tier not in ("collective_rpc", "v0-attr"):
                    raise RuntimeError(f"Official test requires the original weight-swap path; received {tier!r}")
                metrics = {k: None if isinstance(v, float) and not math.isfinite(v) else v
                           for k, v in score(generated, golds).items()}
                final = [r for r in read(run / "result.json")["metrics"] if r["step"] == 625]
                if len(final) != 1 or "retention_nll" not in final[0]:
                    raise ValueError("Missing final-step retention result")
                ref_name = "qwen25_7b" if model == "qwen" else "llama31_8b"
                base_nll = read(LLM / f"results/base_reference_{ref_name}.json")["base_retention_nll"]
                metrics.update(retention_nll=final[0]["retention_nll"], base_retention_nll=base_nll,
                               forgetting_nll_delta=final[0]["retention_nll"] - base_nll)
                out.mkdir(parents=True, exist_ok=True)
                import json
                with gzip.open(out / "predictions.jsonl.gz", "wt") as stream:
                    for index, (generation, gold) in enumerate(zip(generated, golds)):
                        stream.write(json.dumps({**generation, "id": index, "gold": gold,
                                                 "parsed_answer": parse_answer(generation["text"]),
                                                 "correct": is_correct(generation["text"], gold)}) + "\n")
                write(out / "metrics.json", dict(status="complete", identity=identity, metrics=metrics,
                                                 swap_tier=tier, predictions_sha256=sha(out / "predictions.jsonl.gz")))
        del pool


def selected_rows(root, task):
    selection = read(root / task / "selected.json")
    for row in selection["runs"]:
        if sha(root / row["run"] / "adapter/adapter_model.safetensors") != row["checkpoint_sha256"]:
            raise ValueError("A selected checkpoint changed after development selection")
        if sha(root / row["development_file"]) != row["development_sha256"]:
            raise ValueError("Development evidence changed after selection")
        validated_metrics(root, row["cell"], "dev")
    return [r["cell"] for r in selection["runs"]]


def evaluable_rows(root, rows):
    result = []
    for cell in rows:
        failure = root / cell["task"] / "failures" / f"{cell['name']}.json"
        if failure.exists():
            record = read(failure)
            if record.get("cell") != cell or record.get("status") != "execution_failed":
                raise ValueError(f"Invalid training failure record: {failure}")
            print(f"Skip recorded training failure {cell['name']}")
        else:
            run_directory(root, cell)
            result.append(cell)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("math", "coding", "cat", "objects", "hra-extension"))
    p.add_argument("split", choices=("dev", "test"))
    shared_options(p); filter_options(p)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(); root = a.output.resolve()
    if a.split == "test" and not a.dry_run:
        rows = selected_rows(root, a.task)
        rows = [r for r in rows if all(getattr(a, k) in (None, r[k]) for k in ("model", "method", "level", "lr", "seed", "subject", "index"))]
        if not rows:
            p.error("No selected checkpoints match the requested filters")
    else:
        rows = filtered(a.task, a)
    if a.split == "dev" and not a.dry_run:
        rows = evaluable_rows(root, rows)
        if not rows:
            p.error("No successfully trained checkpoints remain for development evaluation")
    if a.task in ("math", "hra-extension"):
        if a.worker:
            math_evaluate(a, rows)
        else:
            command = [python("train"), Path(__file__).resolve(), a.task, a.split, "--output", root, "--worker"]
            for k in ("model", "method", "level", "lr", "seed", "subject", "index"):
                if getattr(a, k) is not None:
                    command += ["--" + k, getattr(a, k)]
            execute(command, cwd=LLM, dry_run=a.dry_run)
    elif a.task == "coding":
        for cell in rows:
            run = root / "coding" / cell["name"]
            command = [python("train"), "-m", "task_harnesses.coding.evaluate", "--run", run,
                       "--split", a.split, "--output", run / a.split]
            if a.split == "test":
                command += ["--sandbox", root / "sandbox/sandbox.json"]
            if not a.dry_run:
                evaluation_identity(root, cell, a.split)
            execute(command, cwd=HARNESS, dry_run=a.dry_run)
            if not a.dry_run:
                validated_metrics(root, cell, a.split)
    else:
        # Training performs the exact step-700 dev and step-750 test evaluations.
        # Read their recorded values; do not change generation seeds by rerunning.
        for cell in rows:
            run = root / a.task / cell["name"]
            if a.dry_run:
                print(f"Read {a.split} metrics from {run / 'training.json'}")
                continue
            identity = evaluation_identity(root, cell, a.split)
            source = run / "training.json"; result = read(source)
            step = 700 if a.split == "dev" else 750
            field = "valid dino_similarity" if a.split == "dev" else "test dino_similarity"
            metrics = [r for r in result["train_info"]["metrics"] if r.get("step") == step and field in r]
            if result["train_info"]["status"] != "success" or len(metrics) != 1:
                raise ValueError(f"Missing successful step-{step} evaluation: {source}")
            write(run / a.split / "metrics.json", dict(status="complete", identity=identity, metrics=metrics[0],
                                                       training_sha256=sha(source)), immutable=True)


if __name__ == "__main__":
    main()

"""Evaluate pretrained models through the original task evaluators."""
import argparse
import gc
import gzip
import json
from pathlib import Path

from common import FLUX, LLM, execute, python, read, sha, shared_options, write
from evaluate_runs import math_worker
from grid import cells, image_configs


def math_baseline(a):
    from datasets import load_dataset
    from eval.run_eval import score, submit
    from loraoft.data.metamath import QUERY_TEMPLATE
    from loraoft.eval_settings import GSM8K_MAX_LENGTH, GSM8K_MAX_NEW_TOKENS
    root = a.output.resolve()
    assets = read(root / "data/assets.json")
    data = load_dataset("openai/gsm8k", "main", revision="740312add88f781978c0658806c59bc2815b9866", split="test")
    if len(data) != 1319:
        raise ValueError("Expected all 1,319 official GSM8K test examples")
    prompts = [QUERY_TEMPLATE.format(query=r["question"]) for r in data]
    golds = [r["answer"] for r in data]
    scratch = root / "scratch"; scratch.mkdir(exist_ok=True)
    for model in ([a.model] if a.model else ("qwen", "llama")):
        out = root / "math/base" / model
        with math_worker(assets[model], scratch) as (queue, _):
            generated, tier = submit(queue, Path(assets[model]), prompts,
                                     max_new_tokens=GSM8K_MAX_NEW_TOKENS,
                                     max_length=GSM8K_MAX_LENGTH, timeout_s=7200)
        if len(generated) != len(golds):
            raise RuntimeError("Incomplete baseline generations")
        out.mkdir(parents=True, exist_ok=True)
        with gzip.open(out / "predictions.jsonl.gz", "wt") as stream:
            for index, (generation, gold) in enumerate(zip(generated, golds)):
                stream.write(json.dumps(dict(generation, id=index, gold=gold)) + "\n")
        suffix = "qwen25_7b" if model == "qwen" else "llama31_8b"
        metrics = score(generated, golds)
        metrics["retention_nll"] = read(LLM / f"results/base_reference_{suffix}.json")["base_retention_nll"]
        write(out / "metrics.json", dict(status="complete", model_snapshot=assets[model], metrics=metrics,
                                         swap_tier=tier, n=1319, predictions_sha256=sha(out / "predictions.jsonl.gz")))


def image_baselines(a):
    import torch
    from transformers import set_seed
    from utils import get_pipeline, get_train_config, init_accelerator
    from pipeline.evaluator import StandardEvaluator
    from report import summarize
    root = a.output.resolve(); snapshot = read(root / "data/assets.json")["flux"]
    rows = [r for r in cells(a.task) if r["method"] == "lora" and r["level"] == 0
            and r["lr"] == (5e-6 if a.task == "cat" else 3e-5)
            and a.subject in (None, r["subject"])]
    if not rows:
        raise ValueError("No baseline subjects match the requested filter")
    records = []
    for cell in rows:
        out = root / a.task / "base" / cell["subject"] / f"seed{cell['seed']}"
        _, config, _ = image_configs(cell)
        config["model_id"] = snapshot
        write(out / "config.json", config, immutable=True)
        if not (out / "metrics.json").exists():
            init_accelerator(); set_seed(cell["seed"])
            cfg = get_train_config(str(out / "config.json"))
            pipeline = get_pipeline(model_id=snapshot, dtype=cfg.dtype, compile=False,
                                    peft_config=None, autocast_adapter_dtype=cfg.autocast_adapter_dtype,
                                    use_gc=cfg.use_gc, device_type="cuda")
            evaluator = StandardEvaluator(pipeline, cfg)
            pipeline.transformer.to("cuda").eval()
            with torch.no_grad():
                evaluator.prepare_base_reference()
                metrics = evaluator.measure()
            write(out / "metrics.json", dict(status="complete", metrics=metrics, config_sha256=sha(out / "config.json")))
            del evaluator, pipeline
            gc.collect(); torch.cuda.empty_cache()
        value = read(out / "metrics.json")
        if value.get("status") != "complete" or value["config_sha256"] != sha(out / "config.json"):
            raise ValueError("Baseline configuration or completion status changed")
        records.append((dict(cell, method="base", capacity=0), value["metrics"]))
    write(root / a.task / "base" / ("summary.json" if not a.subject else f"summary-{a.subject}.json"), summarize(records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("math", "cat", "objects"))
    p.add_argument("--model", choices=("qwen", "llama"))
    p.add_argument("--subject")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    shared_options(p); a = p.parse_args()
    if (a.task == "math" and a.subject) or (a.task != "math" and a.model):
        p.error("Use --model for math or --subject for images")
    if a.worker:
        (math_baseline if a.task == "math" else image_baselines)(a)
        return
    env = "train" if a.task == "math" else "flux"
    command = [python(env), Path(__file__).resolve(), a.task, "--output", a.output.resolve(), "--worker"]
    for key in ("model", "subject"):
        if getattr(a, key):
            command += ["--" + key, getattr(a, key)]
    execute(command, cwd=LLM if env == "train" else FLUX, env_name=env, dry_run=a.dry_run)


if __name__ == "__main__":
    main()

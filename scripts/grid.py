"""The paper grids and portable run configurations. No model imports."""
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

from common import CONFIGS, FLUX, MODELS, read

METHODS = ("lora", "oft", "dora", "pissa", "milora", "hra")
LANGUAGE_LRS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
IMAGE_LRS = (5e-6, 1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3)
LANGUAGE_SEEDS = (13, 37, 73)
IMAGE_SEEDS = (0, 1, 2)


def capacities(task, model, method):
    if task in ("cat", "objects"):
        return (32, 64, 128, 256) if method == "oft" else (16, 32, 62, 124) if method == "hra" else (4, 8, 16, 32)
    caps = ((32, 64, 128) if method == "oft" else (16, 32, 64) if method == "hra"
            else (7, 14, 28) if model == "qwen" else (7, 15, 30))
    return caps[:1] if task == "coding" else caps


def cells(task):
    language = task in ("math", "coding", "hra-extension")
    models = ("llama",) if task == "hra-extension" else ("qwen", "llama") if language else ("flux",)
    methods = ("hra",) if task == "hra-extension" else ("lora", "oft") if task == "objects" else METHODS
    subjects = [s["id"] for s in read(CONFIGS / "objects.json")["subjects"]] if task == "objects" else (task,)
    lrs = ((3e-3, 1e-2) if task == "hra-extension" else LANGUAGE_LRS if language
           else (3e-5, 5e-5, 1e-4) if task == "objects" else IMAGE_LRS)
    seeds = LANGUAGE_SEEDS if language else IMAGE_SEEDS
    result = []
    for model, method, subject in itertools.product(models, methods, subjects):
        for level, cap in enumerate(capacities(task, model, method)):
            for lr, seed in itertools.product(lrs, seeds):
                tag = "b" if method == "oft" else "r"
                name = f"{task}-{model}-{subject}-{method}-{tag}{cap}-lr{lr:g}-s{seed}"
                result.append(dict(index=len(result), task=task, model=model, method=method,
                                   subject=subject, level=level, capacity=cap, lr=lr, seed=seed, name=name))
    return result


def filter_options(parser):
    parser.add_argument("--model", choices=("qwen", "llama", "flux"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--level", type=int, choices=range(4), help="Capacity index, starting from 0")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--subject")
    parser.add_argument("--index", type=int, help="One index in the full grid")


def filtered(task, args):
    result = [r for r in cells(task) if all(getattr(args, k, None) in (None, r[k])
               for k in ("model", "method", "level", "lr", "seed", "subject", "index"))]
    if not result:
        raise ValueError("No paper configuration matches these filters")
    return result


def image_configs(cell):
    method, cap = cell["method"], cell["capacity"]
    adapter = read(FLUX / f"experiments/{method}/example/adapter_config.json")
    if method == "oft":
        adapter.update(oft_block_size=cap, r=0)
    else:
        adapter["r"] = cap
        if method != "hra":
            adapter["lora_alpha"] = cap
    if cell["task"] == "objects":
        training = copy.deepcopy(next(s["training_config"] for s in read(CONFIGS / "objects.json")["subjects"]
                                     if s["id"] == cell["subject"]))
    else:
        training = read(FLUX / "default_training_params.json")
    training["seed"] = cell["seed"]
    training["optimizer_kwargs"]["lr"] = cell["lr"]
    method_config = None
    if method in ("pissa", "milora"):
        method_config = read(FLUX / f"experiments/{method}/example/method_config.json")
        method_config.update(training_rank=cap, lora_rank_anchor=cap,
                             expected_trainable_params=1_198_080 * cap)
    return adapter, training, method_config


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("math", "coding", "cat", "objects", "hra-extension"))
    filter_options(p)
    a = p.parse_args()
    for row in filtered(a.task, a):
        print(row["index"], row["name"])

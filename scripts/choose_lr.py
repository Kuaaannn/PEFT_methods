"""Select one learning rate per configuration from complete development seeds."""
from __future__ import annotations
import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics

from common import read, run_directory, sha, shared_options, validated_metrics, write
from grid import cells


def choose(surface, expected, *, maximize):
    eligible = []
    for lr, records in surface.items():
        keys = [(r["subject"], r["seed"]) for r in records]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate subject/seed result")
        if set(keys) != expected or any(not math.isfinite(r["score"]) for r in records):
            continue
        mean = statistics.mean(r["score"] for r in records)
        eligible.append(((-mean if maximize else mean), lr, mean))
    if not eligible:
        raise ValueError("No learning rate has all required seeds and subjects")
    _, lr, mean = min(eligible)
    return lr, mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("math", "coding", "cat", "objects", "hra-extension"))
    shared_options(p)
    p.add_argument("--model", choices=("qwen", "llama", "flux"))
    p.add_argument("--method", choices=("lora", "oft", "dora", "pissa", "milora", "hra"))
    p.add_argument("--level", type=int, choices=range(4))
    a = p.parse_args(); root = a.output.resolve()
    all_cells = [r for r in cells(a.task) if all(getattr(a, k) in (None, r[k]) for k in ("model", "method", "level"))]
    # The HRA appendix compares both extension LRs against the five main-grid LRs.
    if a.task == "hra-extension":
        all_cells += [r for r in cells("math") if r["model"] == "llama" and r["method"] == "hra"
                      and a.level in (None, r["level"])]
    if not all_cells:
        p.error("No configurations match the requested filters")
    if a.dry_run:
        print(f"Select from {len(all_cells)} development evaluations, requiring every seed and subject at each eligible LR")
        return
    groups = defaultdict(list)
    for cell in all_cells:
        groups[cell["model"], cell["method"], cell["level"]].append(cell)
    selected, configs, incomplete = [], [], []
    for key, group in groups.items():
        expected = {(r["subject"], r["seed"]) for r in group}
        if a.task == "hra-extension":
            expected = {("math", seed) for _, seed in expected}
        surface, by_lr = defaultdict(list), defaultdict(list)
        for cell in group:
            path = root / cell["task"] / cell["name"] / "dev/metrics.json"
            if not path.exists():
                incomplete.append(cell["name"])
                failure = root / cell["task"] / "failures" / f"{cell['name']}.json"
                if not failure.exists():
                    raise FileNotFoundError(f"Missing development result with no recorded failed training: {path}")
                record = read(failure)
                if record.get("cell") != cell or record.get("status") != "execution_failed":
                    raise ValueError(f"Invalid training failure record: {failure}")
                continue
            data = validated_metrics(root, cell, "dev")
            metrics = data["metrics"]
            field = "response_nll" if a.task == "coding" else "valid dino_similarity" if a.task in ("cat", "objects") else "eval_accuracy"
            score = float(metrics[field])
            subject = "math" if a.task == "hra-extension" else cell["subject"]
            surface[cell["lr"]].append(dict(subject=subject, seed=cell["seed"], score=score))
            run = run_directory(root, cell)
            by_lr[cell["lr"]].append(dict(cell=cell, run=str(run.relative_to(root)), checkpoint_sha256=sha(run / "adapter/adapter_model.safetensors"),
                                          development_file=str(path.relative_to(root)), development_sha256=sha(path)))
        lr, mean = choose(surface, expected, maximize=a.task != "coding")
        selected.extend(by_lr[lr])
        configs.append(dict(model=key[0], method=key[1], level=key[2], lr=lr, mean_development_score=mean,
                            eligible_lrs=[x for x, records in surface.items() if {(r['subject'], r['seed']) for r in records} == expected]))
    # A second partial selection must not replace a different previously frozen cohort.
    result = dict(task=a.task, rule="mean over complete seeds and subjects; lower LR on ties; development only",
                  configurations=configs, runs=selected, missing_development=incomplete)
    write(root / a.task / "selected.json", result, immutable=True)
    print(f"Selected {len(selected)} checkpoints; {len(incomplete)} missing development results are listed in selected.json")


if __name__ == "__main__":
    main()

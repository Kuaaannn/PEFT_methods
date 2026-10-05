"""Expansion and validation of pinned checkpoint-analysis selections."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_selected_scores(selection: dict, repository: Path) -> None:
    qwen_source = repository / selection["source_files"]["qwen"]["path"]
    with qwen_source.open(newline="") as source:
        qwen_rows = list(csv.DictReader(source))
    llama_source = repository / selection["source_files"]["llama"]["path"]
    with llama_source.open(newline="") as source:
        llama_rows = list(csv.DictReader(source))
    for model in selection["models"]:
        activation_probe_layers = model.get("activation_probe_layers",
                                            model.get("analysis_layers"))
        if not activation_probe_layers:
            raise ValueError(f"{model['model_key']}: missing activation_probe_layers")
        for pair in model["budget_pairs"]:
            for arm in pair["arms"]:
                if model["model_key"] == "qwen":
                    candidates = {}
                    for row in qwen_rows:
                        if (row["method"] == arm["method"]
                                and int(row["capacity"]) == arm["capacity"]
                                and int(row["step"]) == selection["selection_checkpoint_step"]):
                            candidates.setdefault(float(row["learning_rate"]), []).append(
                                float(row["eval_accuracy"]))
                    surface = {lr: sum(values) / len(values) for lr, values in candidates.items()}
                    lr, mean = sorted(surface.items(), key=lambda item: (-item[1], item[0]))[0]
                else:
                    method = "OFT" if arm["method"] == "flashoft" else "LoRA"
                    capacity = ("b" if arm["method"] == "flashoft" else "r") + str(arm["capacity"])
                    rows = [row for row in llama_rows
                            if row["method"] == method and row["capacity"] == capacity]
                    if len(rows) != 1:
                        raise ValueError(f"Missing unique Llama selection row for {method}/{capacity}")
                    lr, mean = float(rows[0]["learning_rate"]), float(rows[0]["mean_accuracy"])
                if lr != arm["learning_rate"] or abs(mean - arm["mean_dev_accuracy"]) > 1e-12:
                    raise ValueError(f"Pinned winner disagrees with source: {model['model_key']}/{arm}")


def expand_selection(path: str | Path, *, validate_files: bool = True) -> tuple[dict, list[dict]]:
    path = Path(path).resolve()
    selection = json.loads(path.read_text())
    if "models" not in selection:
        tasks = selection["tasks"]
        if len(tasks) != selection.get("n_checkpoints"):
            raise ValueError("Pinned task-list count disagrees with n_checkpoints")
        if len({task["run"] for task in tasks}) != len(tasks):
            raise ValueError("Pinned task-list contains duplicate runs")
        if validate_files:
            for task in tasks:
                manifest_path = Path(task["run"]) / "manifest.json"
                if file_hash(manifest_path) != task["run_manifest_sha256"]:
                    raise ValueError(f"Pinned run manifest changed: {manifest_path}")
                manifest = json.loads(manifest_path.read_text())
                expected = {key: task[key] for key in
                            ("run_id", "model_id", "method", "capacity",
                             "learning_rate", "seed")}
                expected["expected_trainable_params"] = task["trainable_params"]
                mismatch = {key: (manifest.get(key), value) for key, value in expected.items()
                            if manifest.get(key) != value}
                if mismatch:
                    raise ValueError(f"{manifest_path}: pinned task mismatch {mismatch}")
        return selection, tasks
    if validate_files:
        repository = path.parents[3]
        for source in selection["source_files"].values():
            source_path = repository / source["path"]
            if file_hash(source_path) != source["sha256"]:
                raise ValueError(f"Pinned selection source changed: {source_path}")
        _validate_selected_scores(selection, repository)
    seeds = selection["seeds"]
    tasks = []
    for model in selection["models"]:
        activation_probe_layers = model.get("activation_probe_layers",
                                            model.get("analysis_layers"))
        if not activation_probe_layers:
            raise ValueError(f"{model['model_key']}: missing activation_probe_layers")
        for pair in model["budget_pairs"]:
            methods = {arm["method"] for arm in pair["arms"]}
            if methods != {"lora", "flashoft"}:
                raise ValueError(f"{model['model_key']}/{pair['budget']}: not a LoRA/OFT pair")
            for arm in pair["arms"]:
                for seed in seeds:
                    run_id = arm["run_id_template"].format(seed=seed)
                    run = Path(arm["run_root"]) / run_id
                    task = {
                        "array_index": len(tasks), "model_key": model["model_key"],
                        "model_id": model["model_id"], "budget": pair["budget"],
                        "method": arm["method"], "capacity": arm["capacity"],
                        "trainable_params": arm["trainable_params"],
                        "learning_rate": arm["learning_rate"], "seed": seed,
                        "mean_dev_accuracy": arm["mean_dev_accuracy"],
                        "checkpoint_step": selection["selection_checkpoint_step"],
                        "run": str(run), "run_id": run_id,
                        "activation_probe_layers": activation_probe_layers,
                        "random_rank": pair["random_rank"],
                    }
                    if validate_files:
                        manifest_path = run / "manifest.json"
                        manifest = json.loads(manifest_path.read_text())
                        expected = {
                            "run_id": run_id, "model_id": task["model_id"],
                            "method": task["method"], "capacity": task["capacity"],
                            "learning_rate": task["learning_rate"], "seed": task["seed"],
                            "expected_trainable_params": task["trainable_params"],
                        }
                        mismatch = {key: (manifest.get(key), value) for key, value in expected.items()
                                    if manifest.get(key) != value}
                        if mismatch:
                            raise ValueError(f"{manifest_path}: pinned selection mismatch {mismatch}")
                    tasks.append(task)
    if len(tasks) != selection["n_checkpoints"] or len({task["run"] for task in tasks}) != len(tasks):
        raise ValueError("Pinned selection count or uniqueness check failed")
    return selection, tasks

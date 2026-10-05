"""Launch spectral, restoration, component and hyperspherical-energy analyses."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile

from common import CODE, CONFIGS, FLUX, HARNESS, LLM, ROOT, environment, execute, python, read, run_directory, sha, shared_options, write
from evaluate_runs import selected_rows


def rows_for(a):
    rows = selected_rows(a.output.resolve(), a.task)
    if a.stage == "he" or (a.task == "cat" and a.stage == "intervene"):
        rows = [r for r in rows if r["level"] == (3 if a.task == "cat" else 2)]
    for key in ("model", "method", "seed", "level"):
        if getattr(a, key) is not None:
            rows = [r for r in rows if r[key] == getattr(a, key)]
    if a.task == "math" and any(r["model"] == "llama" and r["method"] == "hra" for r in rows):
        if not (a.output.resolve() / "hra-extension/selected.json").exists():
            raise FileNotFoundError("Run and select the HRA extension before language geometry, as in the paper")
        extended = {(r["model"], r["method"], r["level"], r["seed"]): r
                    for r in selected_rows(a.output.resolve(), "hra-extension")}
        rows = [extended[r["model"], r["method"], r["level"], r["seed"]]
                if r["model"] == "llama" and r["method"] == "hra" else r for r in rows]
    if not rows:
        raise ValueError("No selected checkpoints match the requested analysis")
    return rows


def image_manifest(root, rows):
    from pipeline.manifest import Checkpoint
    payload = []
    for row in rows:
        run = run_directory(root, row)
        result = read(run / "training.json")
        config = read(run / "adapter/adapter_config.json")
        name = f"{row['method']}/paper-{row['name']}"
        ck = Checkpoint(checkpoint_id=name, method=row["method"], capacity_kind="block_size" if row["method"] == "oft" else "rank",
                        capacity=row["capacity"], lr=row["lr"], seed=row["seed"], subject="benchmarkcat",
                        dataset_id=result["run_info"]["train_config"]["dataset_id"], max_steps=750,
                        num_trainable_params=result["train_info"]["num_trainable_params"],
                        weights=str(run / "adapter/adapter_model.safetensors"),
                        eval_config=result["run_info"]["train_config"], adapter_config=config,
                        checkpoint_hash=sha(run / "adapter/adapter_model.safetensors"),
                        provenance={"config_path": str(FLUX / "experiments" / row["method"] / f"paper-{row['name']}" / "training_params.json")})
        payload.append(dataclasses.asdict(ck))
    from pipeline import manifest as M
    # Use the existing manifest serialization schema.
    write(M.MANIFEST, {"base_model": read(root / "data/assets.json")["flux"], "n": len(payload), "checkpoints": payload})
    return [r["checkpoint_id"] for r in payload]


def math_intervention(root, row, devices):
    visible = devices.split(",")
    if len(visible) != 2 or not all(visible) or visible[0] == visible[1]:
        raise ValueError("Supply two different devices, for example --gpus 0,1")
    run = run_directory(root, row)
    snapshot = read(root / "data/assets.json")[row["model"]]
    scratch = root / "scratch"; scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="intervention-", dir=scratch) as temporary:
        temp = Path(temporary)
        # Unix socket names must remain short, even when the release lives in a long path.
        with tempfile.TemporaryDirectory(prefix="peft-socket-") as socket_dir:
            sock = Path(socket_dir) / "worker.sock"
            rfd, wfd = os.pipe()
            env = environment("eval"); env["CUDA_VISIBLE_DEVICES"] = visible[1]
            worker = subprocess.Popen([str(python("eval")), "-m", "eval.checkpoint_server", "--base-model", snapshot,
                                       "--socket", str(sock), "--ready-fd", str(wfd), "--max-model-len", "1024",
                                       "--gpu-fraction", "0.85"], cwd=LLM, env=env, pass_fds=(wfd,), start_new_session=True)
            os.close(wfd)
            try:
                ready, _, _ = select.select([rfd], [], [], 900)
                if not ready or os.read(rfd, 1) != b"1":
                    raise RuntimeError("Checkpoint vLLM worker did not initialize")
                execute([python("train"), "-m", "scripts.run_interventions", "--run", run,
                         "--steps", "625", "--base", snapshot, "--banks", root / f"data/banks_{row['model']}.json",
                         "--plan", CONFIGS / "math_interventions.json", "--output", root / "math/geometry" / row["name"],
                         "--evaluation-id", "paper", "--generation-batch-size", "16", "--inference-dtype", "bfloat16",
                         "--vllm-socket", sock, "--vllm-scratch", temp / "patches", "--factor-cache", temp / "factors"],
                        cwd=LLM, extra_env={"CUDA_VISIBLE_DEVICES": visible[0]})
            finally:
                os.close(rfd)
                if worker.poll() is None:
                    os.killpg(worker.pid, signal.SIGTERM)
                    try:
                        worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid, signal.SIGKILL); worker.wait()


def he(root, task, rows):
    from he_analysis import plan as P
    from he_analysis.runner import run as run_he
    domain = "flux" if task == "cat" else "llm"
    assets = read(root / "data/assets.json")
    selection = root / task / "selected.json"
    plan = dict(domain=domain, selection=str(selection), selection_sha256=sha(selection),
                output_root=str(root / task / "he" / hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:12]),
                settings=P.SETTINGS, bases={}, modules={}, tasks=[],
                sources={}, input_sources={})
    for row in rows:
        model = "flux_cat" if domain == "flux" else row["model"]
        run = run_directory(root, row); adapter = run / "adapter"
        if model not in plan["bases"]:
            base, all_shapes = P.base_metadata(domain, Path(assets[row["model"]]))
            if domain == "llm":
                names = read(run / "target_matrices.json")
            else:
                from pipeline.modules import adapter_weight_names
                header, _ = P.tensor_header(adapter / "adapter_model.safetensors")
                names = adapter_weight_names(header)
            shapes = {name: all_shapes[name] for name in names}
            expected = {"qwen": 196, "llama": 224, "flux_cat": 80}[model]
            if len(shapes) != expected:
                raise ValueError(f"Expected all {expected} matrices for {model}")
            plan["bases"][model] = base
            plan["modules"][model] = P.entries_from_shapes(shapes)
        shapes = {e["name"]: e["shape"] for e in plan["modules"][model]}
        identity = P.validate_adapter(adapter, row["method"], row["capacity"], shapes)
        plan["tasks"].append(dict(index=len(plan["tasks"]), model=model, method=row["method"], capacity=row["capacity"],
                                  budget="large",
                                  seed=row["seed"], learning_rate=row["lr"], step=750 if domain == "flux" else 625,
                                  adapter=str(adapter), adapter_identity=identity, expected_weights_sha256=sha(adapter / "adapter_model.safetensors"),
                                  source_checkpoint_id=row["name"]))
        plan["input_sources"][str(adapter / "adapter_config.json")] = sha(adapter / "adapter_config.json")
    for source in (CODE / "he_analysis").glob("*.py"):
        plan["sources"][str(source)] = sha(source)
    extension = root / "hra-extension/selected.json"
    if task == "math" and extension.exists():
        plan["input_sources"][str(extension)] = sha(extension)
    plan["identity"] = P.digest(plan)
    write(Path(plan["output_root"]) / "plan.json", plan, immutable=True)
    for index in range(len(plan["tasks"])):
        run_he(plan, index)
    from he_analysis.report import report
    report(plan)


def worker(a):
    root = a.output.resolve(); rows = rows_for(a)
    if a.stage == "he":
        he(root, a.task, rows)
    elif a.task == "cat":
        from pipeline import manifest as M, model as MD
        group = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:12]
        M.MANIFEST = root / "cat/geometry-inputs" / f"{group}.json"
        ids = image_manifest(root, rows)
        # Relocate model loading to the exact prepared snapshot. Tensor code is unchanged.
        M.BASE_MODEL = MD.BASE_MODEL = read(root / "data/assets.json")["flux"]
        if a.stage == "intervene":
            from pipeline.run_protocol import main as entry
            sys.argv = ["pipeline.run_protocol", "--checkpoints", *ids, "--store", str(root / "cat/geometry")]
            entry()
        elif a.stage == "rotations":
            import importlib.util
            spec = importlib.util.spec_from_file_location("image_rotations", FLUX / "scripts/analyze_singular_rotations.py")
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            sys.argv = ["image_rotations", "--checkpoint", ids[0], "--output", str(root / "cat/rotations"),
                        "--reference-root", str(root / "references/flux"), "--build-reference-only", "--no-plots"]
            module.main()
            for row, ck in zip(rows, ids):
                sys.argv = ["image_rotations", "--checkpoint", ck, "--output", str(root / "cat/rotations"),
                            "--reference-root", str(root / "references/flux"),
                            "--budget", ("xs", "small", "medium", "large")[row["level"]], "--no-plots"]
                module.main()
    elif a.stage == "banks":
        assets = read(root / "data/assets.json")
        for model in sorted({r["model"] for r in rows}):
            command = [python("train"), "-m", "scripts.prepare_intervention_data", "--model-key", model,
                       "--snapshot", assets[model], "--output", root / f"data/banks_{model}.json"]
            for row in rows:
                if row["model"] == model:
                    command += ["--run", run_directory(root, row)]
            execute(command, cwd=LLM)
    elif a.stage == "rotations":
        for model in sorted({row["model"] for row in rows}):
            first = next(row for row in rows if row["model"] == model)
            execute([python("train"), "-m", "scripts.analyze_singular_rotations", "--run", run_directory(root, first),
                     "--step", "625", "--output", root / "math/rotations", "--reference-root", root / "references" / model,
                     "--build-reference-only", "--no-plots"], cwd=LLM)
        for row in rows:
            execute([python("train"), "-m", "scripts.analyze_singular_rotations", "--run", run_directory(root, row),
                     "--step", "625", "--output", root / "math/rotations", "--reference-root", root / "references" / row["model"],
                     "--budget", ("small", "medium", "large")[row["level"]],
                     "--no-plots"], cwd=LLM)
    elif a.task == "coding":
        for row in rows:
            run = run_directory(root, row)
            command = [python("train"), "-m", "task_harnesses.coding.intervene", "--run", run, "--split", "test",
                       "--output", root / "coding/geometry" / row["name"], "--references", root / "references" / row["model"],
                       "--scratch", root / "scratch", "--sandbox", root / "sandbox/sandbox.json"]
            execute(command, cwd=HARNESS)
    else:
        for row in rows:
            math_intervention(root, row, a.gpus)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("math", "coding", "cat"))
    p.add_argument("stage", choices=("banks", "intervene", "he", "rotations"))
    shared_options(p)
    p.add_argument("--model", choices=("qwen", "llama", "flux"))
    p.add_argument("--method", choices=("lora", "oft", "dora", "pissa", "milora", "hra"))
    p.add_argument("--seed", type=int)
    p.add_argument("--level", type=int, choices=range(4))
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"), help="Two device IDs for math interventions")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.task == "coding" and a.stage != "intervene":
        p.error("Coding uses its original intervention command; choose intervene")
    if a.task == "cat" and a.stage == "banks":
        p.error("Image banks are prepared by scripts/prepare.py cat --stage data")
    if a.worker:
        worker(a); return
    env = "flux" if a.task == "cat" else "train"
    command = [python(env), Path(__file__).resolve(), a.task, a.stage, "--output", a.output.resolve(), "--gpus", a.gpus, "--worker"]
    for key in ("model", "method", "seed", "level"):
        if getattr(a, key) is not None:
            command += ["--" + key, getattr(a, key)]
    execute(command, cwd=FLUX if env == "flux" else LLM, env_name=env, dry_run=a.dry_run)


if __name__ == "__main__":
    main()

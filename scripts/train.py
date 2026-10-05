"""Run one or more configurations using the unchanged experiment implementations."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

from common import FLUX, HARNESS, LLM, ROOT, execute, python, read, run_directory, shared_options, training_identity, write
from grid import filter_options, filtered, image_configs


def math_command(cell, output, snapshot):
    command = [python("train"), "-m", "loraoft.train", "--experiment", cell["name"],
               "--model", snapshot, "--method", cell["method"], "--lr", cell["lr"], "--seed", cell["seed"],
               "--out", output / cell["name"], "--group-holdout", 1000, "--max-train-examples", 20000,
               "--max-steps", 625, "--batch-size", 2, "--grad-accum", 16,
               "--max-seq-length", 768, "--eval-steps", 125,
               "--dev-subset", 300, "--retention", "--retention-batch-size", 2,
               "--weight-decay", 0, "--grad-norm-clip", 1, "--lr-scheduler", "cosine", "--warmup-ratio", .03,
               "--dtype", "bfloat16"]
    if cell["method"] == "oft":
        command += ["--block-size", cell["capacity"]]
    else:
        command += ["--rank", cell["capacity"]]
        if cell["method"] != "hra":
            command += ["--alpha", 2 * cell["capacity"]]
    return command


def coding_config(cell, output, snapshot):
    return dict(schema=1, task="coding", model_key=cell["model"], model_revision=Path(snapshot).name,
                snapshot=str(snapshot), method=cell["method"], lr=cell["lr"], seed=cell["seed"],
                data_manifest=str(output.parent / "data/coding/manifest.json"),
                output=str(output / cell["name"]), rank=cell["capacity"] if cell["method"] != "oft" else 7,
                block_size=32, alpha=14, epochs=1, effective_batch=32, micro_batch=1, max_length=1024,
                warmup_steps=0, warmup_fraction=.03, scheduler="cosine", spectral_cache=None,
                expected_trainable=None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("math", "coding", "cat", "objects", "hra-extension"))
    shared_options(p); filter_options(p)
    p.add_argument("--keep-going", action="store_true", help="Run cells in separate processes and record failures without stopping the grid")
    a = p.parse_args()
    rows = filtered(a.task, a)
    root = a.output.resolve()
    if a.keep_going:
        failed = []
        for cell in rows:
            command = [sys.executable, Path(__file__).resolve(), a.task, "--index", cell["index"], "--output", root]
            try:
                execute(command, dry_run=a.dry_run)
            except subprocess.CalledProcessError as error:
                failed.append(cell["name"])
                write(root / a.task / "failures" / f"{cell['name']}.json",
                      dict(cell=cell, returncode=error.returncode, status="execution_failed"))
        print(f"Grid finished with {len(failed)} failed cells; failure records are retained")
        if failed:
            raise SystemExit(1)
        return
    assets_path = root / "data/assets.json"
    assets = read(assets_path) if assets_path.exists() else {}
    if not a.dry_run and not assets:
        p.error("Run scripts/prepare.py first with the same --output directory")
    output = root / a.task
    for cell in rows:
        if not a.dry_run and cell["model"] not in assets:
            p.error(f"Prepare {cell['model']} with the same --output directory first")
        snapshot = assets.get(cell["model"], f"<prepared-{cell['model']}-snapshot>")
        if not a.dry_run:
            try:
                run = run_directory(root, cell)
                if a.task == "coding" and read(run / "complete.json")["status"] != "complete":
                    raise ValueError("Incomplete coding run")
                if a.task in ("cat", "objects") and read(run / "training.json")["train_info"]["status"] != "success":
                    raise ValueError("Incomplete image run")
            except (FileNotFoundError, ValueError):
                pass
            else:
                receipt = read(output / cell["name"] / "execution.json")
                if receipt != training_identity(root, cell, snapshot):
                    raise ValueError("Completed training identity changed. Use a new output directory.")
                print(f"Reuse completed training {cell['name']}")
                continue
            existing = output / cell["name"]
            if existing.exists() and any(existing.iterdir()):
                raise FileExistsError(f"Partial training is retained. Use a new output directory: {existing}")
        if not a.dry_run:
            write(output / "configs" / f"{cell['name']}.cell.json", cell, immutable=True)
        if a.task in ("math", "hra-extension"):
            execute(math_command(cell, output, snapshot), cwd=LLM, dry_run=a.dry_run)
            if not a.dry_run:
                run_directory(root, cell)
        elif a.task == "coding":
            config = coding_config(cell, output, snapshot)
            path = output / "configs" / f"{cell['name']}.json"
            if not a.dry_run:
                write(path, config, immutable=True)
            execute([python("train"), "-m", "task_harnesses.coding.train", "--config", path],
                    cwd=HARNESS, dry_run=a.dry_run)
        else:
            adapter, training, method = image_configs(cell)
            training["model_id"] = str(snapshot)
            adapter["base_model_name_or_path"] = str(snapshot)
            if method:
                method["base_model_id"] = str(snapshot)
            prefix = "paper-"
            exp = Path("experiments") / cell["method"] / (prefix + cell["name"])
            if not a.dry_run:
                if a.task == "objects":
                    if not (FLUX / training["dataset_id"]).is_dir():
                        raise FileNotFoundError("Prepare the object images before starting training")
                for filename, value in (("adapter_config.json", adapter), ("training_params.json", training),
                                        ("method_config.json", method)):
                    if value is not None:
                        write(FLUX / exp / filename, value, immutable=True)
            execute([python("flux"), ROOT / "scripts/image_train.py", str(exp), "--output", output / cell["name"]],
                    cwd=FLUX, env_name="flux", dry_run=a.dry_run)
            if not a.dry_run:
                source = output / cell["name"] / "training.json"
                result = read(source)
                if result["train_info"]["status"] != "success":
                    raise RuntimeError(f"Training did not finish successfully: {cell['name']}")
        if not a.dry_run:
            run = run_directory(root, cell)
            write(output / cell["name"] / "execution.json",
                  training_identity(root, cell, snapshot), immutable=True)
            (root / a.task / "failures" / f"{cell['name']}.json").unlink(missing_ok=True)
    print(f"{'Previewed' if a.dry_run else 'Finished'} {len(rows)} configurations")


if __name__ == "__main__":
    main()

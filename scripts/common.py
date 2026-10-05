"""Paths and process execution for the portable experiment launchers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
LLM = CODE / "LoRA-OFT/experiments"
HARNESS = CODE / "LoRA-OFT"
FLUX = CODE / "LoRA_OFT_image_generation"
CONFIGS = ROOT / "scripts/configs"
MODELS = {
    "qwen": ("Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796"),
    "llama": ("meta-llama/Meta-Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
    "flux": ("black-forest-labs/FLUX.2-klein-base-4B", "a3b4f4849157f664bdbc776fd7453c2783562f4d"),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f"Existing configuration differs. Use a new output directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def python(environment):
    override = os.environ.get("PAPER_" + environment.upper() + "_PYTHON")
    # Preserve the venv executable path rather than resolving its symlink to base Python.
    return Path(override).expanduser().absolute() if override else ROOT / ".venvs" / environment / "bin/python"


def environment(name):
    env = dict(os.environ)
    paths = ([str(FLUX), str(CODE / "third_party_snapshots/flux"), str(CODE)]
             if name == "flux" else [str(LLM), str(HARNESS), str(CODE)])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    return env


def execute(argv, *, cwd=ROOT, env_name="train", dry_run=False, extra_env=None):
    argv = list(map(str, argv))
    print(f"cd {shlex.quote(str(cwd))}\n{shlex.join(argv)}", flush=True)
    if not dry_run:
        env = environment(env_name)
        env.update(extra_env or {})
        subprocess.run(argv, cwd=cwd, env=env, check=True)


def shared_options(parser):
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without downloading or running anything")


def gpu():
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Launch independent single-GPU jobs, not distributed training")
    return torch


def run_directory(output, cell):
    path = Path(output) / cell["task"] / cell["name"]
    if cell["task"] in ("math", "hra-extension"):
        matches = list(path.glob("*/manifest.json"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one completed math run under {path}")
        path = matches[0].parent
        result = read(path / "result.json")
        if result["status"] != "success":
            raise ValueError(f"Training did not succeed: {path}")
    if not (path / "adapter/adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing final adapter: {path}")
    return path


def training_identity(output, cell, snapshot):
    run = run_directory(output, cell)
    files = list((run / "adapter").rglob("*"))
    files += [run / name for name in ("manifest.json", "result.json", "complete.json", "training.json")]
    return dict(cell=cell, snapshot=snapshot,
                checkpoint_sha256=sha(run / "adapter/adapter_model.safetensors"),
                files={p.relative_to(run).as_posix(): sha(p) for p in sorted(files) if p.is_file()})


def evaluation_identity(output, cell, split):
    receipt = read(Path(output) / cell["task"] / cell["name"] / "execution.json")
    current = training_identity(output, cell, receipt["snapshot"])
    if receipt != current:
        raise ValueError(f"Training artifacts changed: {cell['name']}")
    return dict(cell=cell, split=split, checkpoint_sha256=current["checkpoint_sha256"],
                model_snapshot=current["snapshot"], training_files=current["files"])


def validated_metrics(output, cell, split):
    """Require the recorded evaluation to belong to the current checkpoint."""
    path = Path(output) / cell["task"] / cell["name"] / split / "metrics.json"
    value = read(path)
    if value.get("status") != "complete":
        raise ValueError(f"Incomplete evaluation: {path}")
    expected = evaluation_identity(output, cell, split)
    identity = value.get("identity", {})
    if cell["task"] == "coding":
        run = run_directory(output, cell)
        manifest = read(run / "complete.json")
        files = identity.get("adapter_files", {})
        if (identity.get("split") != split or not files
                or files != manifest["adapter_files"]
                or identity.get("config_hash") != manifest["config_hash"]
                or identity.get("bank_hash") != manifest["bank_hash"]
                or sha(Path(manifest["config"]["data_manifest"])) != manifest["bank_hash"]
                or any(Path(name).name != name or sha(run / "adapter" / name) != digest
                       for name, digest in files.items())):
            raise ValueError(f"Evaluation identity changed: {path}")
    elif any(identity.get(key) != item for key, item in expected.items()):
        raise ValueError(f"Evaluation identity changed: {path}")
    return value

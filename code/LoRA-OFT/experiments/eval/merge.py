"""Merge an adapter into its base model. Runs in the TRAINING venv.

The eval venv holds vLLM and deliberately does not hold PEFT/accelerate (their dependency
sets conflict), so merging cannot happen there. This script is the handoff.

Output goes to /dev/shm by default and is meant to be deleted after evaluation.
Delete each merged model after use to limit temporary storage.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def merge_adapter(run_dir: str | Path, out_dir: str | Path,
                  verify: bool = True) -> Path:
    """Merge `run_dir/adapter` into its base model and write a plain HF model.

    With `verify`, checks that the merged forward matches the unmerged one before
    writing. PEFT computes DoRA's and OFT's delta through nontrivial paths, and a merge
    that disagreed with the forward would corrupt every geometry row downstream while
    leaving training numbers healthy.
    """
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = Path(run_dir)
    adapter = run_dir / "adapter"
    if not adapter.exists():
        raise FileNotFoundError(f"{adapter} does not exist")

    cfg = PeftConfig.from_pretrained(str(adapter))
    base_id = cfg.base_model_name_or_path
    if base_id is None:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        base_id = manifest["model_id"]

    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()

    if verify:
        x = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            y_unmerged = model(x).logits.float()
        merged = model.merge_and_unload()
        with torch.no_grad():
            y_merged = merged(x).logits.float()
        rel = float((y_unmerged - y_merged).norm() / y_unmerged.norm())
        if rel > 1e-4:
            raise RuntimeError(
                f"merge/forward mismatch {rel:.2e} for {run_dir.name}. The merged model "
                "does not compute what training computed; do not evaluate it."
            )
    else:
        merged = model.merge_and_unload()

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_id).save_pretrained(str(out_dir))

    # A merged model must never be mistaken for an adapter directory downstream.
    assert not (out_dir / "adapter_config.json").exists()
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run directory containing adapter/")
    p.add_argument("--out", default=None,
                   help="output directory (default: /dev/shm/loraoft-merged/<run>)")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    run = Path(args.run)
    out = Path(args.out) if args.out else Path("/dev/shm/loraoft-merged") / run.name
    path = merge_adapter(run, out, verify=not args.no_verify)
    print(path)


if __name__ == "__main__":  # pragma: no cover
    main()

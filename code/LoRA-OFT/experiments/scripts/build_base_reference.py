"""Score the untuned base model on the frozen retention bank.

The recorded bank checksum binds each reference to its evaluation documents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Runnable directly, not only via `make` (which sets PYTHONPATH=.).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--out", default="results/base_reference.json")
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16")
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from loraoft.data.retention import load_bank
    from loraoft.evaluate import token_nll

    bank = load_bank()
    rows, digest = bank["rows"], bank["sha256"]

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype)).cuda().eval()

    nll = token_nll(model, tok, rows, max_length=a.max_length,
                    batch_size=a.batch_size)

    out = {
        "model_id": a.model,
        "dtype": a.dtype,
        "retention_bank_sha": digest,
        "retention_bank_sha12": digest[:12],
        "n_rows": len(rows),
        "max_length": a.max_length,
        "batch_size": a.batch_size,
        "base_retention_nll": nll,
    }
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    print("Forgetting for a run is now  run.retention_nll - base_retention_nll,"
          "\nvalid only when retention_bank_sha matches.")


if __name__ == "__main__":
    main()

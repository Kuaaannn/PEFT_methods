"""Frozen general-text bank for the retention metric.

Snapshotted to disk and hashed rather than streamed. PEFT calls
`load_dataset("HuggingFaceFW/finewiki", streaming=True)` inside the training loop, which
puts a network dependency mid-run and silently changes content if the dataset is updated.
Retention is a delta against a cached base reference, so the text must be identical for
every run in the program or the deltas are not comparable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_PATH = Path("results/retention_bank.json")


def build_bank(n: int = 200, path: Path = DEFAULT_PATH) -> dict:
    """Download once, freeze to disk with a content hash. Idempotent."""
    if path.exists():
        return json.loads(path.read_text())
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/finewiki", split="train", streaming=True)
    rows = [r["text"] for r in ds.take(n)]
    digest = hashlib.sha256("\u0000".join(rows).encode()).hexdigest()
    bank = {"source": "HuggingFaceFW/finewiki", "n": len(rows),
            "sha256": digest, "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank))
    return bank


def load_bank(path: Path = DEFAULT_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Build it once with `python -m loraoft.data.retention` so "
            "every run scores the same text."
        )
    return json.loads(path.read_text())


if __name__ == "__main__":                                    # pragma: no cover
    b = build_bank()
    print(f"retention bank: {b['n']} rows, sha256 {b['sha256'][:16]}...")

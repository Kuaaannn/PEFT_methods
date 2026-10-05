"""Immutable run manifests, source hashes, and training status.

Manifests are written before training; success, failure, and divergence are
recorded separately. Runtime versions and numerical settings remain local.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


class TrainStatus(str, enum.Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"          # crash, OOM, NaN -- an error
    DIVERGED = "diverged"      # loss crossed the preregistered threshold -- a RESULT
    CANCELED = "canceled"


def _pkg_versions() -> dict[str, str | None]:
    import importlib.metadata as md
    names = ["torch", "transformers", "peft", "accelerate", "datasets", "numpy",
             "scipy", "safetensors", "vllm"]
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except Exception:
            out[n] = None
    return out


def _peft_provenance() -> dict[str, Any]:
    """Record the PEFT version and compact-WY HRA availability."""
    try:
        import peft
    except ImportError:
        return {"available": False}
    path = Path(peft.__file__).parent
    info: dict[str, Any] = {
        "available": True,
        "version": getattr(peft, "__version__", None),
        "is_site_packages": "site-packages" in str(path),
    }
    try:
        from peft.tuners.hra import layer as hra
        info["has_compact_wy_hra"] = hasattr(hra, "_cwy_factors")
    except ImportError:
        info["has_compact_wy_hra"] = None
    return info


def _system_info():
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}


def _code_rev():
    return None


def _source_hash() -> str:
    """SHA-256 over every source file, so runs are distinguishable by code even
    without git.

    `code_rev` returns None when the tree is not a git repository, which silently left
    earlier manifests unable to say which code produced it -- and a
    mid-sweep fix (merged-for-eval) then made runs non-comparable with no way to tell
    them apart after the fact. A content hash needs no git discipline and cannot be
    forgotten.
    """
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


@dataclass
class RunManifest:
    """Everything needed to identify and reproduce one run. Written before training."""

    run_id: str
    experiment_id: str          # e.g. "the experiment"
    method: str
    capacity_kind: str
    capacity: int
    placement: str
    task: str
    model_id: str
    seed: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    grad_accum: int
    max_steps: int
    max_seq_length: int
    lr_scheduler: str
    warmup_ratio: float
    grad_norm_clip: float
    dtype: str
    selection_metric: str
    divergence_threshold: float
    # `checkpoint_steps` is the lightweight telemetry/index schedule.  Adapter
    # artifacts follow the separate policy below; future runs retain one final
    # adapter rather than materialising every evaluation point.
    checkpoint_steps: list[int]
    checkpoint_artifact_policy: str = "final_only"
    # The eval split is derived from these, not stored: post-hoc scoring rebuilds it
    # from the manifest alone (eval/run_eval.py). Without them a checkpoint cannot be
    # rescored on the same items it was trained against, which is the whole point of
    # keeping checkpoints.
    group_holdout: int | None = None
    dev_subset: int | None = None
    eval_steps: int | None = None
    retention_batch_size: int | None = None
    expected_trainable_params: int | None = None
    method_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset_revision: str | None = None
    trainable_params: int | None = None
    total_params: int | None = None
    # Filled automatically
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = SCHEMA_VERSION
    code_rev: str | None = field(default_factory=_code_rev)
    source_hash: str = field(default_factory=_source_hash)
    packages: dict = field(default_factory=_pkg_versions)
    peft_provenance: dict = field(default_factory=_peft_provenance)
    system: dict = field(default_factory=_system_info)

    def write(self, path: str | Path) -> Path:
        """Write once. Refuses to overwrite -- a manifest is immutable by contract."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError(
                f"{p} exists. A manifest is immutable; to re-run this cell, delete the "
                "whole run directory so the old record is not silently replaced."
            )
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p


@dataclass
class RunResult:
    """Terminal record. Always written, including on failure."""

    run_id: str
    status: TrainStatus
    error_msg: str = ""
    train_time_s: float = 0.0
    eval_time_s: float = 0.0
    total_time_s: float = 0.0
    trainable_params: int = 0
    total_params: int = 0
    peak_memory_bytes: int = 0
    final_train_loss: float = float("nan")
    total_tokens: int = 0
    # Worst normalised ||R^T R - I|| seen at any Tier-1 record, and whether it crossed
    # telemetry.ORTHO_GATE. Under the `oft` arm (Cayley-Neumann) this is a genuine
    # run outcome: NS5 leaves the orthogonal manifold at large rotation angles, so a
    # crossing means the arm stopped preserving singular values at that learning rate.
    ortho_residual_worst: float = 0.0
    ortho_gate_exceeded: bool = False
    metrics: list[dict[str, Any]] = field(default_factory=list)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["status"] = self.status.value
        p.write_text(json.dumps(d, indent=2, default=str))
        return p


class FailureTable:
    """Append-only log of every non-success outcome.

    Diverged cells remain in the table, so reporting includes unsuccessful runs.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, run_id: str, status: TrainStatus, reason: str,
               **context: Any) -> None:
        row = {
            "run_id": run_id,
            "status": status.value,
            "reason": reason,
            "at": datetime.now(timezone.utc).isoformat(),
            **context,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def write_parquet(rows: list[dict], path: str | Path) -> Path:
    """Row-level artefacts as Parquet, with a CSV sibling for human inspection.

    Preserve individual measurement rows alongside any aggregate summaries.
    """
    import pandas as pd

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(p, index=False)
    df.to_csv(p.with_suffix(".csv"), index=False)
    return p

"""Lifecycle policy for checkpoint-intervention artifacts.

The protocol's JSON/JSONL records are the durable scientific result.  Tensor
payloads exist only to build, restore, and evaluate variants inside one worker
run; retaining them after that run duplicates hundreds of GiB of base weights,
trained weights, SVD factors, activations, and reference log-probabilities.

Cleanup is deliberately allow-listed and scoped to one task output directory.
Training checkpoints and rotation references live outside that directory and
cannot match this API's deletion scope.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


POLICY_VERSION = 1

# These small JSON files only index transient tensor payloads.  Keeping them
# after deleting their payloads would make an interrupted task look resumable
# while pointing at files that intentionally no longer exist.
TRANSIENT_INDEXES = frozenset({
    "reference-base.json",
    "reference-trained.json",
    "activation-inputs-base.json",
    "activation-inputs-trained.json",
})


def artifact_kind(path: Path) -> str | None:
    """Return the allow-listed transient kind, or ``None`` for durable data."""
    name = path.name
    if name.endswith(".pt"):
        return "tensor"
    if name.endswith(".factors.sha256"):
        return "factor_checksum"
    if name in TRANSIENT_INDEXES:
        return "tensor_index"
    if name.endswith(".tmp"):
        return "incomplete_atomic_write"
    return None


def transient_artifacts(root: Path) -> list[tuple[Path, str]]:
    """List transient files below a single task output without following links."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        kind = artifact_kind(path)
        if kind is not None:
            found.append((path, kind))
    return found


def prune_transient_artifacts(root: Path) -> dict:
    """Remove only regenerable payloads below ``root`` and report exact bytes."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Task output directory does not exist: {root}")
    artifacts = transient_artifacts(root)
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    for path, kind in artifacts:
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            # A wrapper may be completing cleanup immediately after its worker.
            # Treat an already-removed transient as success, not corruption.
            continue
        counts[kind] += 1
        sizes[kind] += size
    remaining = transient_artifacts(root)
    if remaining:
        examples = ", ".join(str(path) for path, _ in remaining[:5])
        raise RuntimeError(f"Transient artifact cleanup incomplete: {examples}")
    return {
        "policy_version": POLICY_VERSION,
        "scope": str(root),
        "files_removed": sum(counts.values()),
        "bytes_removed": sum(sizes.values()),
        "counts_by_kind": dict(sorted(counts.items())),
        "bytes_by_kind": dict(sorted(sizes.items())),
        "remaining_transient_files": 0,
        "durable_artifacts": [
            "aggregate.jsonl",
            "per_prompt_metrics.jsonl",
            "matrix_geometry.jsonl",
            "per-cell JSON",
            "manifest and diagnostic JSON",
        ],
    }


def write_cleanup_report(root: Path, report: dict, *, outcome: str) -> Path:
    """Atomically record cleanup provenance next to the task's durable results."""
    root = Path(root)
    destination = root / "artifact_cleanup.json"
    event = {
        "outcome": outcome,
        "files_removed": report["files_removed"],
        "bytes_removed": report["bytes_removed"],
        "counts_by_kind": report["counts_by_kind"],
        "bytes_by_kind": report["bytes_by_kind"],
    }
    previous = json.loads(destination.read_text()) if destination.is_file() else None
    events = list(previous.get("events", [])) if previous else []
    if previous and not events:
        events.append({
            "outcome": previous.get("outcome", "legacy_cleanup"),
            "files_removed": previous.get("files_removed", 0),
            "bytes_removed": previous.get("bytes_removed", 0),
            "counts_by_kind": previous.get("counts_by_kind", {}),
            "bytes_by_kind": previous.get("bytes_by_kind", {}),
        })
    events.append(event)
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    for row in events:
        counts.update(row["counts_by_kind"])
        sizes.update(row["bytes_by_kind"])
    payload = {
        **report,
        "outcome": outcome,
        "files_removed": sum(row["files_removed"] for row in events),
        "bytes_removed": sum(row["bytes_removed"] for row in events),
        "counts_by_kind": dict(sorted(counts.items())),
        "bytes_by_kind": dict(sorted(sizes.items())),
        "events": events,
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(destination)
    return destination

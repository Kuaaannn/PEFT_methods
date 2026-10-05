"""Content-keyed output records and table helpers (PROTOCOL.md section 5).

Every measurement is one record carrying the identity, provenance, measurement and cost fields
the contract requires. A record is keyed by the *content* of what produced it - checkpoint hash,
module manifest hash, operator and params, rng key, dtype profile, bank hash - never by a variant
name, so a reused name cannot collide with a different configuration.

Failed and infeasible cells are written too, with null metrics and a reason. Corrected
evaluations receive a new record id; historical records are never edited in place.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import specint
from specint import contract

RUNS = pathlib.Path(__file__).resolve().parent.parent / "runs"


def json_record(value):
    """Return strict-JSON data and identify undefined numeric measurements.

    The standard evaluator may produce NaN for a conditional metric with an empty denominator.
    Such an optional value is represented as JSON ``null`` rather than a non-standard NaN token.
    Primary DINO/drift metrics are validated separately by the runner.
    """
    paths = []

    def visit(node, path):
        if isinstance(node, float) and not math.isfinite(node):
            paths.append(path)
            return None
        if isinstance(node, dict):
            return {key: visit(item, f"{path}.{key}") for key, item in node.items()}
        if isinstance(node, (list, tuple)):
            return [visit(item, f"{path}[{index}]") for index, item in enumerate(node)]
        return node

    result = visit(value, "$")
    if paths and isinstance(result, dict):
        result["undefined_numeric_fields"] = sorted(set(paths))
        result["undefined_numeric_policy"] = (
            "non-finite optional measurements are serialized as JSON null"
        )
    return result


def cache_key(*, checkpoint_hash: str, module_manifest_hash: str, operator: str,
              operator_params: dict, rng_key: Optional[int], dtype_profile: str,
              bank_hash: str) -> str:
    """The resume/output key. Every field that can change a number is in it."""
    payload = json.dumps({
        "checkpoint_hash": checkpoint_hash,
        "module_manifest_hash": module_manifest_hash,
        "operator": operator,
        "operator_params": operator_params,
        "rng_key": rng_key,
        "dtype_profile": dtype_profile,
        "bank_hash": bank_hash,
        "contract_version": contract.version(),
        "library_hash": specint.library_hash(),
    }, sort_keys=True, default=str, allow_nan=False).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


@dataclass
class Record:
    record_id: str
    checkpoint_id: str
    checkpoint_hash: str
    module_manifest_hash: str
    operator_id: str
    operator_params: dict
    rng_key: Optional[int]
    dtype_profile: str
    bank_hash: str
    status: str
    status_reason: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)
    wall_seconds: Optional[float] = None
    peak_allocated_bytes: Optional[int] = None
    peak_reserved_bytes: Optional[int] = None
    hardware: Optional[str] = None
    contract_version: str = field(default_factory=contract.version)
    library_version: str = specint.__version__
    library_hash: str = field(default_factory=specint.library_hash)
    created: float = field(default_factory=time.time)

    def __post_init__(self):
        contract.require_status(self.status)
        contract.require_operator(self.operator_id)

    def as_dict(self) -> dict:
        return asdict(self)


class Store:
    """Append-only JSONL of records, with content-keyed resume."""

    def __init__(self, name: str, root: pathlib.Path = RUNS):
        self.path = root / name / "records.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._seen.add(json.loads(line)["record_id"])

    def has(self, record_id: str) -> bool:
        return record_id in self._seen

    def append(self, rec: Record) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(json_record(rec.as_dict()), default=str, allow_nan=False) + "\n")
        self._seen.add(rec.record_id)

    def all(self) -> list:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]


def write_table(rows: list, path: pathlib.Path, columns=None) -> None:
    """Flat CSV for one of the four required tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    cols = columns or sorted({k for r in rows for k in r})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def flatten(prefix: str, d: dict) -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(f"{key}.", v))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
    return out

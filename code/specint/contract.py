"""Load and validate contract.json (PROTOCOL.md section 6).

The contract fixes names, defaults and status values; the protocol fixes their meaning. This
module is the only place that reads it, so an adapter never hard-codes an operator id.
"""
from __future__ import annotations

import functools
import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent

# Search order, so the library is portable whether contract.json sits beside the package (a
# vendored copy) or one level up (this repository's layout).
_SEARCH = ("SPECINT_CONTRACT", _HERE / "contract.json", _HERE.parent / "contract.json")


def contract_path() -> pathlib.Path:
    import os
    env = os.environ.get("SPECINT_CONTRACT")
    if env:
        return pathlib.Path(env)
    for cand in _SEARCH[1:]:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "contract.json not found beside specint/ or in its parent; set SPECINT_CONTRACT")


@functools.lru_cache(maxsize=4)
def load(path: str | None = None) -> dict:
    p = pathlib.Path(path) if path else contract_path()
    return json.loads(p.read_text())


def version(path: str | None = None) -> str:
    return load(path)["contract_version"]


def operators(path: str | None = None) -> dict:
    return {k: v for k, v in load(path)["operators"].items() if k != "P"}


def defaults(operator: str, path: str | None = None) -> dict:
    return operators(path)[operator].get("params", {})


def status_values(path: str | None = None) -> dict:
    return {k: v for k, v in load(path)["status_values"].items() if k != "P"}


def require_status(status: str, path: str | None = None) -> str:
    valid = status_values(path)
    if status not in valid:
        raise ValueError(f"status {status!r} is not in the contract: {sorted(valid)}")
    return status


def require_operator(operator: str, path: str | None = None) -> str:
    ops = operators(path)
    if operator not in ops:
        raise ValueError(f"operator {operator!r} is not in the contract: {sorted(ops)}")
    return operator


def required_fields(path: str | None = None) -> dict:
    return {k: v for k, v in load(path)["record_required_fields"].items() if k != "P"}

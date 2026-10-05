"""Path guards.

Quarantined run trees cannot be made immutable in this container (`chattr` is
blocked and the process is root, so permission bits are advisory). The only
protection that actually holds is a refusal at the point of use.
"""

from __future__ import annotations

from pathlib import Path

QUARANTINE_MARK = "quarantine"


def reject_quarantined(path: str | Path) -> Path:
    """Refuse to read or write anywhere under a quarantine tree.

    Quarantined corpora carry known defects (see the README in each tree) and
    are kept only as evidence. Silently resuming into one would reintroduce
    `.done` markers that skip real work, which is the specific accident this
    guard exists to prevent.
    """
    p = Path(path)
    parts = {q.lower() for q in p.resolve().parts}
    if QUARANTINE_MARK in parts:
        raise SystemExit(
            f"refusing to use quarantined path: {p}\n"
            "This tree is retained as evidence only and must not be read as a "
            "result or resumed into. See its README.md for the defects."
        )
    return p

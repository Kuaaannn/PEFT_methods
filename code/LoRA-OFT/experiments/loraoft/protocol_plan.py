"""Expand the shared protocol contract into concrete LLM measurement cells.

This module is deliberately stdlib-only: plans can be inspected and validated on a
login node without importing torch.  Numerical execution remains GPU-gated by
``checkpoint_analysis.require_gpu``.
"""
from __future__ import annotations

import json
from pathlib import Path

from .checkpoint_analysis import CORE, SHARED, file_hash


def _contract() -> dict:
    return json.loads((SHARED / "contract.json").read_text())


def detailed_cells(*, band_d_rel: float = 0.01) -> list[dict]:
    """Every Q3/Q4 pilot cell declared by contract.json, including five draws.

    Shape normalization is nonlinear, so both signed shape requests are retained
    as separate cells rather than being treated as an exact antithetic pair.
    """
    if not 0 < band_d_rel:
        raise ValueError("band_d_rel must be positive")
    operators = _contract()["operators"]
    cells = []
    cells.extend({"operator": "update_scale", "params": {"q": q}}
                 for q in operators["update_scale"]["params"]["q"])
    cells.extend({"operator": "spectral_path", "params": {"lam": lam}}
                 for lam in operators["spectral_path"]["params"]["lam"])

    spec = operators["spectral_sign"]["params"]
    for t in spec["t"]:
        cells.extend([
            {"operator": "spectral_sign", "params": {"t": t, "z": "restore"}},
            {"operator": "spectral_sign", "params": {"t": t, "z": "neg_restore"}},
        ])
        for draw in range(spec["n_random"]):
            for z in ("random", "neg_random"):
                cells.append({"operator": "spectral_sign",
                              "params": {"t": t, "z": z, "draw": draw}})

    relative = operators["relative_sign"]["params"]
    for sigma in relative["sigma"]:
        for z in ("ones", "neg_ones"):
            cells.append({"operator": "relative_sign", "params": {"sigma": sigma, "z": z}})
        for draw in range(relative["n_random"]):
            for z in ("random", "neg_random"):
                cells.append({"operator": "relative_sign",
                              "params": {"sigma": sigma, "z": z, "draw": draw}})

    # Five seeded directions for each nonlinear shape request and its matched gain.
    for sigma in operators["spectral_shape"]["params"]["sigma"]:
        for draw in range(relative["n_random"]):
            for z in ("random", "neg_random"):
                params = {"sigma": sigma, "z": z, "draw": draw}
                cells.append({"operator": "spectral_shape", "params": params})
                for sign in (1, -1):
                    cells.append({"operator": "global_gain",
                                  "params": {**params, "sign": sign}})

    for band in operators["spectral_band"]["params"]["band"]:
        for z in ("ones", "neg_ones"):
            cells.append({"operator": "spectral_band",
                          "params": {"band": band, "d_rel": band_d_rel, "z": z}})
        for draw in range(relative["n_random"]):
            for z in ("random", "neg_random"):
                cells.append({"operator": "spectral_band",
                              "params": {"band": band, "d_rel": band_d_rel,
                                         "z": z, "draw": draw}})
    return cells


def plan(panel: str, *, band_d_rel: float = 0.01) -> list[dict]:
    if panel not in {"core", "detailed"}:
        raise ValueError(f"unknown panel {panel!r}")
    cells = [{"operator": op, "params": {}} for op in CORE]
    if panel == "detailed":
        cells.extend(detailed_cells(band_d_rel=band_d_rel))
    validate_plan(cells, require_core=True)
    return cells


def validate_plan(cells: list[dict], *, require_core: bool = False) -> None:
    known = set(_contract()["operators"]) - {"P"}
    if not isinstance(cells, list) or not cells:
        raise ValueError("a plan must be a nonempty JSON list")
    seen = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or set(cell) - {"operator", "params"}:
            raise ValueError(f"plan cell {index} has unknown or missing fields")
        operator, params = cell.get("operator"), cell.get("params", {})
        if operator not in known or not isinstance(params, dict):
            raise ValueError(f"invalid plan cell {index}: {cell!r}")
        identity = json.dumps([operator, params], sort_keys=True, allow_nan=False)
        if identity in seen:
            raise ValueError(f"duplicate plan cell {index}: {cell!r}")
        seen.add(identity)
    if require_core and [c["operator"] for c in cells[:len(CORE)]] != list(CORE):
        raise ValueError("the core must begin with restoration and retain its declared order")


def write_plan(path: str | Path, panel: str = "detailed", *, band_d_rel: float = 0.01) -> None:
    Path(path).write_text(json.dumps(plan(panel, band_d_rel=band_d_rel), indent=2))


def resolve_plan(path: str | Path | None = None, *, panel: str = "core",
                 band_d_rel: float = 0.01) -> tuple[list[dict], dict]:
    """Resolve legacy core-plus-extra lists or an exact, hash-pinned cell subset.

    Subset IDs are one-based positions in the source's complete core-plus-extra
    plan. No implicit core is appended to a subset. This keeps original cell
    definitions in one place and allows metadata-only preflight on login nodes.
    """
    cells = plan(panel, band_d_rel=band_d_rel)
    metadata = {"format": "core_plus_extras", "plan_sha256": None}
    if path is not None:
        path = Path(path)
        payload = json.loads(path.read_text())
        metadata["plan_sha256"] = file_hash(path)
        if isinstance(payload, list):
            validate_plan(payload)
            if any(cell["operator"] in CORE for cell in payload):
                raise ValueError("List plans supply additional cells; core cells are already included")
            cells.extend(payload)
            validate_plan(cells, require_core=True)
        else:
            fields = {"source_plan", "source_plan_sha256", "source_cell_ids"}
            if not isinstance(payload, dict) or set(payload) != fields:
                raise ValueError("Subset plans require source_plan, source_plan_sha256 and source_cell_ids")
            if panel != "core":
                raise ValueError("An explicit subset cannot be combined with a detailed panel")
            source = (path.parent / payload["source_plan"]).resolve()
            if file_hash(source) != payload["source_plan_sha256"]:
                raise ValueError("Subset source plan checksum mismatch")
            extras = json.loads(source.read_text())
            validate_plan(extras)
            if any(cell["operator"] in CORE for cell in extras):
                raise ValueError("Subset source must be a legacy core-plus-extra plan")
            full = cells + extras
            validate_plan(full, require_core=True)
            ids = payload["source_cell_ids"]
            if (not isinstance(ids, list) or not ids
                    or any(type(index) is not int or not 1 <= index <= len(full) for index in ids)
                    or ids != sorted(set(ids)) or ids[0] != 1):
                raise ValueError("Subset IDs must be unique, increasing, in range, and start with restoration (1)")
            cells = [full[index - 1] for index in ids]
            validate_plan(cells)
            return cells, {**metadata, **payload, "format": "source_cell_subset"}
    metadata["source_cell_ids"] = list(range(1, len(cells) + 1))
    return cells, metadata

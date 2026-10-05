"""Immutable fixed-gauge base-SVD artifacts shared by all model families."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


REFERENCE_RECONSTRUCTION_RTOL = 5e-5


def reference_path(root: Path, model_id: str, matrix_name: str) -> Path:
    model_key = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id).strip("-")
    matrix_key = hashlib.sha256(matrix_name.encode()).hexdigest()[:16]
    return Path(root) / model_key / f"{matrix_key}.pt"


def save_reference(path: Path, artifact: dict, metadata: dict, torch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save({**artifact, "metadata": metadata}, temporary)
    temporary.replace(path)


def load_reference(path: Path, W0, expected: dict, torch) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing fixed base SVD {path}; run with --build-reference-only first")
    artifact = torch.load(path, map_location=W0.device, weights_only=True)
    metadata = artifact.get("metadata", {})
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Fixed base SVD {path} has wrong {key}: "
                             f"{metadata.get(key)!r} != {value!r}")
    reconstruction = ((artifact["U"].to(W0.device) *
                       artifact["singular_values"].to(W0.device).unsqueeze(0)) @
                      artifact["Vh"].to(W0.device))
    mismatch = (torch.linalg.vector_norm(reconstruction - W0.float()) /
                torch.linalg.vector_norm(W0.float()).clamp_min(1e-30))
    if float(mismatch) > REFERENCE_RECONSTRUCTION_RTOL:
        raise ValueError(
            f"Fixed base SVD {path} reconstructs a different weight: {float(mismatch)} "
            f"> {REFERENCE_RECONSTRUCTION_RTOL}"
        )
    return artifact

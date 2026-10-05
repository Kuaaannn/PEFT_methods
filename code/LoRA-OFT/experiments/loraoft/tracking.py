"""Local-only training; records are written by the training loop."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Roles are the natural aggregation unit: shapes and behaviour differ sharply between
# attention and MLP matrices, so measurements are stratified by role.
ROLES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Tier-1 fields worth a live series, per role. The rest live in the artifact.
TIER1_AGG_FIELDS = ("delta_rel_fro", "delta_fro", "delta_op", "delta_stable_rank",
                    "delta_r_95", "ortho_residual_max", "generator_fro", "angle_rms", "a_cond",
                    "b_cond")


def _role_of(matrix: str) -> str:
    """Matrix role from either a module name or a parameter name.

    Tier-1 rows key on the MODULE (`model.layers.0.mlp.down_proj`) while
    `target_matrix_names` yields the PARAMETER (`....down_proj.weight`). Matching only
    the `.weight` form silently routed every per-role series to "other" -- no error, just
    an empty breakdown in the dashboard.
    """
    stem = matrix[:-len(".weight")] if matrix.endswith(".weight") else matrix
    for r in ROLES:
        if stem == r or stem.endswith(f".{r}"):
            return r
    return "other"


class Tracker:
    """Tracking is disabled; training logs are written by the training loop."""
    def __init__(self, **kwargs):
        pass
    def log_step(self, *args, **kwargs):
        pass
    def log_eval(self, *args, **kwargs):
        pass
    def log_matrix_rows(self, *args, **kwargs):
        pass
    def set_summary(self, *args, **kwargs):
        pass
    def upload_artifacts(self, *args, **kwargs):
        pass
    def finish(self, *args, **kwargs):
        pass


def tracker_for_run(manifest):
    return Tracker(enabled=False)

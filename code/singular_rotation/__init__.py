"""Shared singular-vector rotation measurements for LoRA/OFT checkpoints."""
import hashlib
from pathlib import Path

from .core import (ANALYSIS_VERSION, analyze_weight_pair,
                   apply_block_rotation_to_vectors, apply_block_rotation_to_weight_right,
                   choose_example_positions, choose_examples, compute_base_svd,
                   verify_oft_convention)
from .core import project_oft_rotation
from .output import CheckpointOutput, render_plots
from .paper import render_paper_report
from .reference import load_reference, reference_path, save_reference


def library_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("__init__.py", "core.py", "output.py", "paper.py", "reference.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()

__all__ = [
    "ANALYSIS_VERSION",
    "CheckpointOutput",
    "analyze_weight_pair",
    "apply_block_rotation_to_vectors",
    "apply_block_rotation_to_weight_right",
    "choose_example_positions",
    "choose_examples",
    "compute_base_svd",
    "project_oft_rotation",
    "load_reference",
    "reference_path",
    "library_hash",
    "render_plots",
    "render_paper_report",
    "save_reference",
    "verify_oft_convention",
]

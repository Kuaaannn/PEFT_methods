"""Checkpoint discovery for this repository (PROTOCOL.md sections 1 and 5).

The old sweep encoded its bookkeeping in directory names (`lora/s4-dog2-r32-lr5e-05-seed0`), with
a leading stage index that meant nothing outside that sweep. Those names are parsed exactly once,
here, into explicit fields; the legacy string survives only as provenance. Nothing downstream
parses a name again, and no measurement from the old runs is read at all.

Build once:

    python -m pipeline.manifest --build

then use ``load()`` / ``select()``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

HERE = pathlib.Path(__file__).resolve().parent.parent
CHECKPOINTS = HERE / "checkpoints"
EXPERIMENT_CONFIGS = HERE / "experiments"
LEGACY_CONFIGS = HERE / "legacy" / "data" / "experiments"
# The abandoned 150-step schedule kept its run directories in a separate archive.
LEGACY_CONFIGS_ARCHIVE = HERE / "legacy" / "results" / "archive" / "steps150" / "experiments"
LEGACY_MANIFEST = HERE / "legacy" / "results" / "checkpoint_manifest.csv"
MANIFEST = HERE / "runs" / "manifest.json"

BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"

# The single place the old naming scheme is interpreted.
_SEED = re.compile(r"-seed(\d+)$")
_LR = re.compile(r"-lr([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)(?=-|$)")
_CAP = re.compile(r"-(r|b)(\d+)-")
_SUBJECT_PREFIXES = ("s4-", "s15-", "s2-", "baseeval-", "s35-", "s3b-")


@dataclass
class Checkpoint:
    """One trained adapter, described in fields rather than in a name."""

    checkpoint_id: str                  # stable, content-addressed-ish id used everywhere downstream
    method: str                         # "lora" | "oft"
    capacity_kind: Optional[str]        # "rank" | "block_size"
    capacity: Optional[int]
    lr: Optional[float]
    seed: Optional[int]
    subject: str
    dataset_id: Optional[str]
    max_steps: Optional[int]
    num_trainable_params: Optional[int]
    weights: str                        # path relative to the repository root
    eval_config: dict = field(default_factory=dict)   # effective settings, defaults merged in
    adapter_config: dict = field(default_factory=dict)
    role: str = "trained"               # "trained" | "base_reference"
    checkpoint_hash: Optional[str] = None
    provenance: dict = field(default_factory=dict)

    def path(self) -> pathlib.Path:
        return HERE / self.weights

    def as_dict(self) -> dict:
        return asdict(self)


def _parse_legacy_name(method: str, name: str) -> dict:
    """Interpret one legacy experiment name. Called once, at build time, and never again."""
    out = {"subject": "benchmarkcat", "lr": None, "seed": None,
           "capacity_kind": None, "capacity": None}
    m = _SEED.search(name)
    if m:
        out["seed"] = int(m.group(1))
    m = _LR.search(name)
    if m:
        out["lr"] = float(m.group(1))
    m = _CAP.search(name)
    if m:
        out["capacity_kind"] = "rank" if m.group(1) == "r" else "block_size"
        out["capacity"] = int(m.group(2))
    for pref in _SUBJECT_PREFIXES:
        if name.startswith(pref):
            rest = name[len(pref):]
            out["subject"] = rest.split("-")[0]
            break
    return out


# Settings that decide what a checkpoint is evaluated *on*. Recorded explicitly at build time,
# with the benchmark defaults already merged in: a run's own file holds only its overrides, so
# reading it alone would report `dataset_id: None` for every benchmark-cat run.
_EVAL_FIELDS = ("dataset_id", "dataset_split", "image_column", "instance_prompts", "valid_size",
                "test_size", "resolution", "num_inference_steps", "guidance_scale",
                "weighting_scheme", "max_sequence_length", "batch_size", "batch_size_eval",
                "dino_model_id", "dino_image_size", "max_steps", "seed")


def _config_path(method: str, name: str) -> Optional[pathlib.Path]:
    for root, sub in ((EXPERIMENT_CONFIGS, method), (LEGACY_CONFIGS, method), (LEGACY_CONFIGS_ARCHIVE, "")):
        p = (root / sub / name / "training_params.json") if sub else (root / name / "training_params.json")
        if p.exists():
            return p
    return None


def _effective_config(path: Optional[pathlib.Path]) -> tuple:
    """The run's config with `default_training_params.json` merged underneath, as run.py did.

    Returns ``(config, found)``. A missing file is **not** silently replaced by the defaults: the
    defaults describe the benchmark cat dataset, so doing that would label a DreamBooth subject's
    checkpoint as having been trained on cats. Callers get ``found=False`` and an empty config.
    """
    if path is None:
        return {}, False
    defaults = json.loads((HERE / "default_training_params.json").read_text())
    own = json.loads(path.read_text())
    return {**defaults, **own}, True


def _hash_file(p: pathlib.Path, chunk: int = 1 << 22) -> str:
    h = hashlib.blake2b(digest_size=16)
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _new_id(method: str, parsed: dict, role: str, experiment_name: str) -> str:
    """A readable id built from fields, not from the legacy string."""
    if role == "base_reference":
        return f"base/{parsed['subject']}/seed{parsed['seed']}"
    cap = f"{'r' if parsed['capacity_kind'] == 'rank' else 'b'}{parsed['capacity']}"
    # Replication checkpoints intentionally coexist with the historical sweep.  Their
    # scientific fields are identical, so omitting the run family would create duplicate
    # IDs and M.get() could silently return the old checkpoint.
    family = "/catrep" if experiment_name.startswith("catrep-") else ""
    return f"{method}/{parsed['subject']}{family}/{cap}/lr{parsed['lr']:g}/seed{parsed['seed']}"


def build(hash_weights: bool = True, verbose: bool = True,
          hash_experiments: Optional[set[str]] = None) -> list:
    """Scan checkpoints/, join to the legacy run configs, and write runs/manifest.json."""
    legacy_rows = {}
    if LEGACY_MANIFEST.exists():
        with LEGACY_MANIFEST.open() as fh:
            for r in csv.DictReader(fh):
                legacy_rows[r["experiment_name"]] = r

    out = []
    dirs = sorted(d for d in CHECKPOINTS.iterdir() if d.is_dir())
    for i, d in enumerate(dirs):
        w = d / "adapter_model.safetensors"
        if not w.exists():
            continue
        method, _, name = d.name.partition("--")
        parsed = _parse_legacy_name(method, name)
        cfg_path = d / "training_params.json"
        if not cfg_path.is_file():
            cfg_path = _config_path(method, name)
        cfg, cfg_found = _effective_config(cfg_path)
        acfg_path = d / "adapter_config.json"
        acfg = json.loads(acfg_path.read_text()) if acfg_path.exists() else {}
        parsed["capacity_kind"] = "block_size" if method == "oft" else "rank"
        method_path = d / "method_config.json"
        method_cfg = json.loads(method_path.read_text()) if method_path.is_file() else {}
        parsed["capacity"] = (acfg.get("oft_block_size") if method == "oft"
                              else method_cfg.get("training_rank", acfg.get("r")))
        parsed["seed"] = cfg.get("seed", 0)
        parsed["lr"] = cfg.get("optimizer_kwargs", {}).get("lr")

        lr = cfg.get("optimizer_kwargs", {}).get("lr", parsed["lr"])
        role = "base_reference" if (lr == 0.0) else "trained"
        if parsed["lr"] is None:
            parsed["lr"] = lr
        row = legacy_rows.get(f"{method}/{name}", {})

        ck = Checkpoint(
            checkpoint_id=_new_id(method, parsed, role, name),
            method=method,
            capacity_kind=parsed["capacity_kind"],
            capacity=parsed["capacity"],
            lr=lr,
            seed=cfg.get("seed", parsed["seed"]),
            subject=parsed["subject"],
            dataset_id=cfg.get("dataset_id"),
            max_steps=cfg.get("max_steps"),
            eval_config={k: cfg.get(k) for k in _EVAL_FIELDS},
            num_trainable_params=int(row["num_trainable_params"]) if row.get("num_trainable_params") else None,
            weights=str(w.relative_to(HERE)),
            adapter_config=acfg,
            role=role,
            checkpoint_hash=(
                _hash_file(w)
                if hash_weights or (hash_experiments and name in hash_experiments)
                else None
            ),
            provenance={"legacy_experiment_name": f"{method}/{name}",
                        "legacy_manifest_row": bool(row),
                        "config_found": cfg_found,
                        "config_path": str(cfg_path.relative_to(HERE)) if cfg_path else None},
        )
        out.append(ck)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(dirs)}", flush=True)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(
        {"base_model": BASE_MODEL, "n": len(out), "checkpoints": [c.as_dict() for c in out]},
        indent=1))
    return out


def load() -> list:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} missing; run `python -m pipeline.manifest --build`")
    d = json.loads(MANIFEST.read_text())
    return [Checkpoint(**c) for c in d["checkpoints"]]


def select(checkpoints=None, **kw) -> list:
    """Filter by explicit fields, e.g. select(method='lora', subject='benchmarkcat', capacity=32).

    ``role='trained'`` is the default: base-reference checkpoints (lr = 0, adapter provably the
    identity) are never returned unless asked for, so they cannot leak into an intervention set.
    """
    cks = checkpoints if checkpoints is not None else load()
    kw.setdefault("role", "trained")
    out = []
    for c in cks:
        if all(getattr(c, k, None) == v for k, v in kw.items() if v is not None):
            out.append(c)
    return out


def get(checkpoint_id: str, checkpoints=None) -> Checkpoint:
    for c in (checkpoints if checkpoints is not None else load()):
        if c.checkpoint_id == checkpoint_id:
            return c
    raise KeyError(checkpoint_id)


def load_train_config(checkpoint: Checkpoint):
    """Load the checkpoint's effective benchmark config through the standard config loader."""
    relative = checkpoint.provenance.get("config_path")
    if not relative:
        raise FileNotFoundError(
            f"{checkpoint.checkpoint_id} has no recorded training_params.json; "
            "standard-equivalent evaluation is impossible"
        )
    from utils import get_train_config

    config = get_train_config(str(HERE / relative))
    if config.model_id != BASE_MODEL:
        raise ValueError(
            f"{checkpoint.checkpoint_id} uses {config.model_id!r}, expected {BASE_MODEL!r}"
        )
    if config.seed != checkpoint.seed:
        raise ValueError(
            f"{checkpoint.checkpoint_id} config seed {config.seed} != manifest seed {checkpoint.seed}"
        )
    return config


def summary(cks=None) -> str:
    cks = cks if cks is not None else load()
    import collections
    lines = [f"{len(cks)} checkpoints"]
    for key in ("role", "method", "subject", "capacity"):
        c = collections.Counter(getattr(x, key) for x in cks)
        head = ", ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))[:12])
        lines.append(f"  {key:14} {len(c):3} distinct  {head}{' ...' if len(c) > 12 else ''}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--no-hash", action="store_true", help="skip content hashing (faster, not for records)")
    ap.add_argument(
        "--hash-experiment", action="append", default=[],
        help="with --no-hash, still hash this exact experiment name; repeat as needed")
    a = ap.parse_args()
    if a.build:
        cks = build(hash_weights=not a.no_hash, hash_experiments=set(a.hash_experiment))
        print(f"wrote {MANIFEST}")
    print(summary())

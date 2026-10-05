from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
WORKSPACE = REPO.parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        # JSONL is delimited by physical newlines. str.splitlines() also splits
        # valid U+0085/U+2028/U+2029 characters *inside* JSON strings.
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list or JSONL: {path}")
    return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def exclusive(directory, *, lock_name=".lock", blocking=False):
    """POSIX advisory lock, automatically released on crash; no stale lock cleanup."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if Path(lock_name).name != lock_name:
        raise ValueError("Lock name must be a filename")
    with (directory / lock_name).open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)




def source_files(include_tests=False):
    # Never walk .venv/vendor/data/results: doing so makes every cache lookup scan GBs.
    files = list(ROOT.glob("*.py"))
    directories = ["shared", "coding"] + (["tests"] if include_tests else [])
    for name in directories:
        files.extend((ROOT / name).rglob("*.py"))
    return sorted(files)


def source_identity():
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in source_files()})


def writable_output(path):
    path = Path(path).resolve()
    protected = (REPO / "experiments", WORKSPACE / "specint", WORKSPACE / "singular_rotation")
    if path == REPO or path == WORKSPACE or any(root in (path, *path.parents) for root in protected):
        raise ValueError("New task outputs must not touch the protected math tree")
    return path

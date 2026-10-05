"""Deterministic RNG keys (PROTOCOL.md section 6).

The protocol forbids process-dependent ``hash()`` for key derivation: CPython salts string
hashing per process, so a resumed or re-run cell would silently draw different random signs
from the same declared configuration. Keys here are a blake2b digest of explicit string ids,
so the same (checkpoint, operator, draw) triple gives the same numbers on any machine forever.
"""
from __future__ import annotations

import hashlib

import torch

_DIGEST_BYTES = 8


def rng_key(*ids: object) -> int:
    """Stable non-negative 63-bit key from explicit ids.

    Every component is stringified and length-prefixed, so ``("ab", "c")`` and ``("a", "bc")``
    cannot collide.
    """
    h = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    for part in ids:
        s = str(part).encode("utf-8")
        h.update(str(len(s)).encode("ascii"))
        h.update(b":")
        h.update(s)
        h.update(b"|")
    return int.from_bytes(h.digest(), "big") & ((1 << 63) - 1)


def generator(*ids: object, device="cpu") -> torch.Generator:
    """A seeded generator for the given ids."""
    g = torch.Generator(device=device)
    g.manual_seed(rng_key(*ids))
    return g


def random_signs(k: int, *ids: object, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """A +/-1 vector of length k, reproducible from ids alone."""
    g = generator(*ids, device=device)
    bits = torch.randint(0, 2, (k,), generator=g, device=device)
    return (bits.to(dtype) * 2.0 - 1.0)

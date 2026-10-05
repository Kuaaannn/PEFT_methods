"""specint - shared pure-tensor intervention library for PROTOCOL.md v1.1.

This package is the *only* code that both the diffusion and the LLM study are required to run
identically. It contains matrix algebra and statistics and nothing else: no model loading, no
checkpoint discovery, no module names, no file formats, no metric implementations. A repository
adapter locates and merges matrices and supplies evaluation banks; everything below that line
lives here.

Porting rule (PROTOCOL.md section 6): copy this directory verbatim, or install it as a package.
``library_hash()`` must agree between repositories for their results to be comparable, and it is
recorded in every output record.
"""
from __future__ import annotations

import hashlib
import pathlib

__version__ = "1.3.0"
CONTRACT_VERSION = "1.2.0"

_MODULES = ("__init__.py", "rng.py", "ops.py", "geometry.py", "contract.py", "plan.py")


def library_hash() -> str:
    """Digest of this library's source, for the identity fields of every record.

    Deliberately excludes ``conformance.py`` (tests) so that adding a fixture does not
    invalidate stored measurements, and includes everything that can change a number.
    """
    here = pathlib.Path(__file__).parent
    h = hashlib.blake2b(digest_size=16)
    for name in _MODULES:
        h.update(name.encode())
        h.update((here / name).read_bytes())
    return h.hexdigest()


from . import geometry, ops, rng  # noqa: E402,F401

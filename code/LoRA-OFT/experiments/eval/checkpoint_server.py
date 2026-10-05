"""Blocking local-socket vLLM worker for checkpoint intervention outcomes.

The process runs in the evaluation environment while checkpoint construction runs
in the training environment. Communication is local, synchronous, and has no
polling loop or scheduled timer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import struct

from eval.server import Worker


def _read_exact(connection: socket.socket, size: int) -> bytes:
    blocks = []
    remaining = size
    while remaining:
        block = connection.recv(remaining)
        if not block:
            raise ConnectionError("Client disconnected before completing its request")
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def _receive(connection: socket.socket) -> dict:
    size = struct.unpack("!Q", _read_exact(connection, 8))[0]
    if size > 128 << 20:
        raise ValueError("vLLM request exceeds 128 MiB")
    return json.loads(_read_exact(connection, size))


def _send(connection: socket.socket, payload: dict) -> None:
    encoded = json.dumps(payload, allow_nan=False).encode()
    connection.sendall(struct.pack("!Q", len(encoded)) + encoded)


def _handle(worker: Worker, request: dict) -> dict:
    action = request.get("action")
    if action == "load_patch":
        worker.load_weight_patch(request["patch_path"])
        return {"status": "ok", "swap_tier": worker.last_swap}
    if action == "load_checkpoint":
        # Validation-only route: exactly the ordinary standard-eval loader.
        worker.load_weights(request["model_path"])
        return {"status": "ok", "swap_tier": worker.last_swap}
    if action == "generate":
        rows = worker.generate(
            request["prompts"],
            max_new_tokens=request["max_new_tokens"],
            max_length=request["max_length"],
        )
        return {"status": "ok", "rows": rows, "engine": "vllm"}
    raise ValueError(f"Unsupported checkpoint-server action: {action!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--ready-fd", required=True, type=int)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-fraction", type=float, default=0.40)
    args = parser.parse_args()
    if not 0 < args.gpu_fraction < 1:
        parser.error("--gpu-fraction must be between zero and one")

    # The server keeps this descriptor, but vLLM's child processes must not keep
    # the parent runner blocked if server initialization itself fails.
    os.set_inheritable(args.ready_fd, False)
    socket_path = Path(args.socket)
    if socket_path.exists():
        socket_path.unlink()
    worker = Worker(args.base_model, max_model_len=args.max_model_len,
                    gpu_fraction=args.gpu_fraction)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        server.listen(1)
        os.write(args.ready_fd, b"1")
        os.close(args.ready_fd)
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    response = _handle(worker, _receive(connection))
                except Exception as exc:  # noqa: BLE001
                    response = {"status": "error",
                                "error": f"{type(exc).__name__}: {exc}"}
                _send(connection, response)
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

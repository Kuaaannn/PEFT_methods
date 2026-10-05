"""Persistent vLLM evaluation worker.

Runs in the EVAL venv (vLLM + torch), never the training venv: vLLM and PEFT/accelerate
have conflicting dependency sets. Communication is over the filesystem, one JSON job at a
time, so the two environments never have to import each other.

The persistent worker reuses one engine and swaps weights in place. Merged models
are written to /dev/shm and deleted after evaluation to limit temporary storage.

Deliberately NOT used: vLLM's native multi-LoRA serving. It is faster, but it computes
W_0 x + BA x unmerged while OFT must be merged -- different numerics for different
methods inside the headline comparison. Fairness beats speed here; every method is
merged and treated identically.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from loraoft.eval_settings import (GSM8K_ENGINE_MAX_MODEL_LEN, GSM8K_MAX_LENGTH,
                                    GSM8K_MAX_NEW_TOKENS, GSM8K_SEED,
                                    GSM8K_TEMPERATURE)

# collective_rpc refuses to ship a raw callable to the worker process unless this is set:
#   TypeError: Object of type <class 'function'> is not serializable
# Without it the in-place weight swap fails and every checkpoint pays a ~40 s full engine
# rebuild -- ~13 h across a full sweep, with correct answers throughout, so nothing fails
# and nothing looks wrong. "Insecure" here means cloudpickle between our own two trusted
# local processes; no untrusted input reaches it. Must be set BEFORE vllm is imported.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def require_merged(model_path: str) -> str:
    """Fail loudly on an unmerged adapter directory.

    This is the one failure in the harness that produces no error and no warning: handed
    a PEFT checkpoint, vLLM finds the base model named in adapter_config.json, loads THAT,
    ignores the adapter, and returns a complete set of perfectly plausible numbers for the
    untuned model. Every downstream stage then looks healthy. So it is checked at every
    point a model path is consumed.
    """
    p = Path(model_path)
    if not p.is_dir():
        return model_path                       # a hub id: nothing local to inspect
    if (p / "adapter_config.json").exists():
        raise SystemExit(
            f"\n{p}\n  is a PEFT adapter, not a merged model. Evaluating it would "
            "silently score the untuned base model.\n  Merge it first in the TRAINING "
            "venv:  python -m eval.merge --run <run_dir>\n")
    if not any(p.glob("*.safetensors")) and not any(p.glob("*.bin")):
        raise SystemExit(f"{p} contains no model weights")
    return model_path


def _iter_weights(model_path: str):
    """(name, tensor) for every shard, materialised once."""
    from safetensors.torch import load_file

    weights: dict = {}
    for shard in sorted(Path(model_path).glob("*.safetensors")):
        weights.update(load_file(str(shard)))
    return list(weights.items())


def _load_weights_into_worker(worker, model_path: str) -> None:
    """Runs inside each vLLM worker process; `worker` is injected by collective_rpc.

    `worker.get_model()` is the supported accessor in vLLM's V1 worker. The obvious
    `worker.model_runner.model` chain is a V0 shape and does not exist there -- reaching
    for it silently fell through to the 40 s full-engine rebuild on every checkpoint.
    """
    worker.get_model().load_weights(_iter_weights(model_path))


def _iter_patch_weights(patch_path: str):
    """Yield one GPU tensor at a time from an ephemeral analysis patch."""
    import torch

    root = Path(patch_path).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for row in manifest["weights"]:
        source = (root / row["file"]).resolve()
        if source.parent != root:
            raise ValueError("Weight patch escapes its declared directory")
        yield row["name"], torch.load(
            source, map_location="cuda:0", weights_only=True)


def _load_weight_patch_into_worker(worker, patch_path: str) -> None:
    """Runs in each vLLM worker and updates only tensors named by the patch."""
    worker.get_model().load_weights(_iter_patch_weights(patch_path))


class Worker:
    def __init__(self, base_model: str, max_model_len: int = 1024,
                 gpu_fraction: float = 0.85):
        from vllm import LLM

        self.llm = LLM(model=base_model, dtype="bfloat16",
                       max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_fraction,
                       enable_prefix_caching=True)   # few-shot suites share long prefixes
        self.current = base_model
        self.last_swap = "init"
        self.swap_errors: list[str] = []

    def load_weights(self, model_path: str) -> None:
        """Swap weights in place rather than rebuilding the engine.

        Rebuilding costs 40-60 s. Across the full sweep -- 237 runs x ~5 checkpoints --
        that is ~20 h of pure engine startup, so the in-place path is not an optimisation
        but the difference between a feasible and an infeasible evaluation pass.

        Three tiers, most specific first:
          1. `collective_rpc` -- the supported V1 route; runs on each worker process.
          2. the V0 `driver_worker.model_runner` attribute chain, for older engines.
          3. full rebuild, so correctness never depends on either being available.
        """
        require_merged(model_path)
        if model_path == self.current:
            self.last_swap = "cached"
            return

        errors = []
        for tier, fn in (("collective_rpc", self._swap_v1), ("v0-attr", self._swap_v0)):
            try:
                fn(model_path)
                if self.llm.reset_prefix_cache() is False:
                    raise RuntimeError("vLLM refused to clear prefix cache after a weight swap")
                self.current = model_path
                self.last_swap = tier
                return
            except Exception as e:                              # noqa: BLE001
                errors.append(f"{tier}: {type(e).__name__}: {e}")

        # Reported back through the job result, not printed. vLLM reconfigures stdout
        # during engine init, so this process's stdout does not reliably reach a
        # redirected log -- which is how a silent fall-through to the 40 s rebuild path
        # went unnoticed: correct answers, 13 h of avoidable engine startup across a
        # full sweep.
        self.last_swap = "rebuild"
        self.swap_errors = errors
        from vllm import LLM
        del self.llm
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()
        self.llm = LLM(model=model_path, dtype="bfloat16",
                       enable_prefix_caching=True)
        self.current = model_path

    def load_weight_patch(self, patch_path: str) -> None:
        """Apply an ephemeral partial checkpoint without rebuilding the engine."""
        errors = []
        for tier, fn in (("collective_rpc", self._patch_v1), ("v0-attr", self._patch_v0)):
            try:
                fn(patch_path)
                if self.llm.reset_prefix_cache() is False:
                    raise RuntimeError("vLLM refused to clear prefix cache after a weight patch")
                self.last_swap = tier + "-patch"
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{tier}: {type(exc).__name__}: {exc}")
        raise RuntimeError("Partial vLLM weight update failed: " + "; ".join(errors))

    def _swap_v1(self, model_path: str) -> None:
        self.llm.collective_rpc(_load_weights_into_worker, args=(model_path,))

    def _swap_v0(self, model_path: str) -> None:
        runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
        runner.model.load_weights(_iter_weights(model_path))

    def _patch_v1(self, patch_path: str) -> None:
        self.llm.collective_rpc(_load_weight_patch_into_worker, args=(patch_path,))

    def _patch_v0(self, patch_path: str) -> None:
        runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
        runner.model.load_weights(_iter_patch_weights(patch_path))

    def swap_tier_probe(self) -> str:
        """Which tier a swap would use, without performing one. For diagnostics."""
        try:
            self.llm.collective_rpc(lambda w: hasattr(w, "get_model"))
            return "collective_rpc"
        except Exception as e:                                  # noqa: BLE001
            return f"unavailable: {type(e).__name__}: {e}"

    def generate(self, prompts: list[str], max_new_tokens: int = GSM8K_MAX_NEW_TOKENS,
                 max_length: int = GSM8K_MAX_LENGTH,
                 temperature: float = GSM8K_TEMPERATURE) -> list[dict]:
        """Greedy generation with PEFT's dual cap.

        `min(max_length - len(prompt), max_new_tokens)`: PEFT found that a handful of
        runaway generations inflated eval time 3x while contributing zero accuracy,
        because they were truncated mid-sentence anyway. Truncation is recorded per
        sample so token-budget effects can be distinguished from model errors.
        """
        from vllm import SamplingParams

        # The dual cap is PER PROMPT: min(max_new_tokens, max_length - len(prompt)).
        # This used to pass a single `max_tokens=max_new_tokens`, which is not the cap
        # the docstring describes and not the cap training used -- so a long prompt could
        # generate past `max_length` and be scored under a budget the in-process path
        # would have refused. vLLM takes one SamplingParams per prompt, so the real cap
        # is expressible directly.
        tok = self.llm.get_tokenizer()
        params = []
        for pr in prompts:
            n_prompt = len(tok(pr)["input_ids"])
            budget = max(1, min(max_new_tokens, max_length - n_prompt))
            params.append(SamplingParams(temperature=temperature, max_tokens=budget,
                                         n=1, seed=GSM8K_SEED))
        outs = self.llm.generate(prompts, params)
        rows = []
        for o in outs:
            c = o.outputs[0]
            rows.append({
                "text": c.text,
                "n_prompt_tokens": len(o.prompt_token_ids),
                "n_generated_tokens": len(c.token_ids),
                "truncated": c.finish_reason == "length",
                "finish_reason": c.finish_reason,
            })
        return rows

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--queue", required=True, help="directory watched for job JSON files")
    p.add_argument("--max-model-len", type=int, default=GSM8K_ENGINE_MAX_MODEL_LEN)
    args = p.parse_args()

    q = Path(args.queue)
    q.mkdir(parents=True, exist_ok=True)
    worker = Worker(args.base_model, max_model_len=args.max_model_len)

    # Readiness is signalled by a FILE, not by a log line. vLLM reconfigures stdout
    # during engine init, so this process's `print` does not reliably reach a redirected
    # log -- a reader grepping for it blocks forever even though the worker is up and
    # answering jobs. The queue is already a filesystem protocol; readiness joins it.
    ready = q / "READY"
    ready.write_text(f"{args.base_model}\n")
    print(f"worker ready on {args.base_model}; watching {q}", flush=True)

    while True:
        jobs = sorted(q.glob("*.job.json"))
        if not jobs:
            time.sleep(2)
            continue
        job_path = jobs[0]
        job = json.loads(job_path.read_text())
        try:
            worker.load_weights(job["model_path"])
            rows = worker.generate(
                job["prompts"],
                max_new_tokens=job.get("max_new_tokens", GSM8K_MAX_NEW_TOKENS),
                max_length=job.get("max_length", GSM8K_MAX_LENGTH))
            out = {"status": "ok", "rows": rows,
                   "swap_tier": worker.last_swap,
                   "swap_errors": worker.swap_errors}
        except Exception as e:                                  # noqa: BLE001
            out = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        # Atomic handoff. The client polls for result_path.exists() and reads
        # immediately, so a direct write_text() exposes a window where the file
        # exists but is empty -- the client then hits JSONDecodeError and the whole
        # sweep dies, losing every unsaved row. Write beside it, then rename.
        rp = Path(job["result_path"])
        tmp = rp.with_suffix(".partial")
        tmp.write_text(json.dumps(out))
        tmp.rename(rp)
        job_path.unlink()


if __name__ == "__main__":  # pragma: no cover
    main()

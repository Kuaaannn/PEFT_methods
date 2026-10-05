"""In-process validation during training.

This is the CORRECTNESS path, not the fast path. It uses `transformers.generate` so a run
is self-contained in the training venv; the fast path is the persistent vLLM worker in
`eval/`, which is 10-30x quicker and is what the full sweep should use.

Two details carried over from PEFT's method_comparison, both of which matter:

  * **Dual generation cap.** `min(max_length - prompt, max_new_tokens)`. A handful of
    runaway generations inflated their eval time 3x while contributing zero accuracy,
    because they were truncated mid-sentence anyway.
  * **Truncation is recorded per sample**, not just accuracy, so token-budget
    effects can be distinguished from model errors.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


class merged_for_eval:
    """Merge the adapter for evaluation, then restore the base weights EXACTLY.

    Generation runs one forward per token, and an unmerged OFT forward rebuilds 196
    Cayley rotations (`torch.linalg.solve`) every time. Measured: 28.58 ms unmerged
    versus 0.38 ms merged -- a **75x** difference, which showed up as OFT evaluation
    taking 52-67 min per run against LoRA's 5.

    Restoration is by COPY from a cached snapshot, not by `unmerge_adapter()`. In bf16 a
    single merge/unmerge round trip was measured to leave **7.2e-3 relative error** in
    the base weights -- a 1% corruption of a frozen tensor, mid-run. Adapter methods keep
    the base frozen, so a snapshot taken once is always the exact truth to restore.

    Costs one bf16 CPU copy of the adapted matrices (~2.6 GB at 1.5B).
    """

    def __init__(self, model, cached: dict[str, torch.Tensor] | None):
        self.model = model
        self.cached = cached

    def __enter__(self):
        if self.cached is not None:
            self.model.merge_adapter()
        return self.model

    def __exit__(self, *exc):
        if self.cached is None:
            return False
        for name, mod in self.model.named_modules():
            base = getattr(mod, "get_base_layer", None)
            if base is None:
                continue
            w = base().weight
            src = self.cached.get(name)
            if src is not None:
                w.data.copy_(src.to(w.device, w.dtype))
        # merged_adapters must be cleared or PEFT thinks the merge is still applied.
        for mod in self.model.modules():
            if hasattr(mod, "merged_adapters"):
                mod.merged_adapters.clear()
        return False


@torch.no_grad()
def snapshot_base_weights(model) -> dict[str, torch.Tensor]:
    """CPU copy of every adapted base weight, keyed by module name.

    Only valid while the base is frozen -- i.e. for adapter methods, never full FT.
    """
    out = {}
    for name, mod in model.named_modules():
        base = getattr(mod, "get_base_layer", None)
        if base is not None:
            out[name] = base().weight.detach().clone().cpu()
    return out


@torch.no_grad()
def generate_greedy(model, tokenizer, prompts: list[str], *, max_new_tokens: int = 300,
                    max_length: int = 800, batch_size: int = 32,
                    device: str = "cuda") -> list[dict[str, Any]]:
    """Deterministic generation with the dual cap. Left-padded so batching is valid."""
    was_training = model.training
    model.eval()
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"          # right padding corrupts batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[dict[str, Any]] = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(device)
            n_prompt = enc["input_ids"].shape[1]
            budget = max(1, min(max_new_tokens, max_length - n_prompt))
            out = model.generate(**enc, max_new_tokens=budget, do_sample=False,
                                 num_beams=1, pad_token_id=tokenizer.pad_token_id)
            gen = out[:, n_prompt:]
            for j in range(gen.shape[0]):
                ids = gen[j]
                keep = ids[ids != tokenizer.pad_token_id]
                text = tokenizer.decode(keep, skip_special_tokens=True)
                rows.append({
                    "text": text,
                    "n_prompt_tokens": int(n_prompt),
                    "n_generated_tokens": int(keep.numel()),
                    # Hit the cap without emitting EOS: the answer may simply be cut off.
                    "truncated": bool(keep.numel() >= budget),
                })
    finally:
        tokenizer.padding_side = prev_side
        if was_training:
            model.train()
    return rows


def gsm8k_evaluator(tokenizer, queries: list[str], responses: list[str], *,
                    max_new_tokens: int = 300, max_length: int = 800,
                    batch_size: int = 32) -> Callable:
    """Build the selection-metric evaluator: accuracy on the GSM8K validation sample.

    Returns a callable `(model, step) -> metrics dict`, matching `train.train`'s hook.

    The validation set is 500 examples, not PEFT's 50. At p ~ 0.5, n=50 gives a 7.1pp
    binomial SD -- larger than the differences this project must resolve, so selecting a
    learning rate on it would select noise.
    """
    from .data.metamath import QUERY_TEMPLATE, is_correct

    prompts = [QUERY_TEMPLATE.format(query=q) for q in queries]

    def evaluate(model, step: int) -> dict[str, Any]:
        rows = generate_greedy(model, tokenizer, prompts,
                               max_new_tokens=max_new_tokens, max_length=max_length,
                               batch_size=batch_size)
        correct = [is_correct(r["text"], gold) for r, gold in zip(rows, responses)]
        n = len(correct)
        n_trunc = sum(r["truncated"] for r in rows)
        n_ok = sum(correct)
        # Accuracy conditional on non-truncation, so a truncation artefact cannot be
        # mistaken for a capability difference.
        untrunc = [c for c, r in zip(correct, rows) if not r["truncated"]]
        return {
            "step": step,
            "valid_accuracy": n_ok / n if n else float("nan"),
            "valid_n": n,
            "valid_n_correct": n_ok,
            "valid_truncation_rate": n_trunc / n if n else float("nan"),
            "valid_accuracy_untruncated": (sum(untrunc) / len(untrunc)
                                           if untrunc else float("nan")),
            "valid_mean_gen_tokens": (sum(r["n_generated_tokens"] for r in rows) / n
                                      if n else float("nan")),
        }

    return evaluate


@torch.no_grad()
def token_nll(model, tokenizer, texts: list[str], *, max_length: int = 768,
              batch_size: int = 8, device: str = "cuda") -> float:
    """Mean per-token NLL on held-out text.

    Used both as a selection metric with a tighter practical margin than accuracy
    (0.005 nats/token) and as the retention/forgetting axis: the increase in this
    quantity on general text before versus after training.
    """
    was_training = model.training
    model.eval()
    total_nll, total_tok = 0.0, 0
    try:
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(device)
            logits = model(**enc).logits.float()
            labels = enc["input_ids"]
            mask = enc["attention_mask"][:, 1:].bool()
            lp = torch.nn.functional.cross_entropy(
                logits[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none")
            total_nll += float((lp * mask).sum())
            total_tok += int(mask.sum())
    finally:
        if was_training:
            model.train()
    return total_nll / total_tok if total_tok else float("nan")

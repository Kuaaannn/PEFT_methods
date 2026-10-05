"""Identical numeric path for standard and modified-checkpoint evaluations."""
from pathlib import Path
import time

from .data import encode, pad_features
from .io import atomic_json, digest, file_hash, source_identity, write_rows
from .runtime import require_cuda

EVAL_PROTOCOL = {"version": 1, "dtype": "bfloat16", "attention": "sdpa",
                 "retention_documents": 200, "retention_max_length": 768,
                 "retention_batch_size": 2, "kl_direction": "base||evaluated",
                 "reduction": "pooled_valid_next_tokens", "nll_logits": "float32",
                 "generation_batch_size": 4, "coding_new_tokens": 1024,
                 "coding_beams": 1, "coding_generation_batch_size": 1,
                 "response_max_length": 4096, "response_batch_size": 1,
                 "coding_prompt": "evalplus_force_base"}


def generate(model, tokenizer, prompts, *, beams, new_tokens, batch_size=4, stop_strings=None,
             progress_label=None):
    torch = require_cuda()
    from transformers import GenerationConfig
    model.eval()
    previous = tokenizer.padding_side
    tokenizer.padding_side = "left"
    output = []
    if progress_label:
        progress_started = time.monotonic()
        progress_tokens = 0
        print(f"GENERATION {progress_label} completed=0/{len(prompts)} elapsed_s=0.0 tokens=0", flush=True)
    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start:start + batch_size]
                inputs = tokenizer(batch, padding=True, truncation=False, return_tensors="pt").to("cuda:0")
                width = inputs["input_ids"].shape[1]
                if width + new_tokens > model.config.max_position_embeddings:
                    raise ValueError("Prompt exceeds context window; no silent benchmark truncation")
                generation_config = GenerationConfig(
                    do_sample=False, num_beams=beams, max_new_tokens=new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id, use_cache=True)
                extra = {"stop_strings": stop_strings, "tokenizer": tokenizer} if stop_strings else {}
                tokens = model.generate(**inputs, generation_config=generation_config, **extra)[:, width:]
                for index, generated in enumerate(tokens.tolist()):
                    stopped = tokenizer.eos_token_id in generated
                    if stopped:
                        generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
                    text = tokenizer.decode(generated, skip_special_tokens=True)
                    stop_at = min([text.find(s) for s in (stop_strings or []) if s in text] or [len(text)])
                    output.append({"text": text[:stop_at], "raw_text": text,
                                   "token_ids": generated, "stopped_by_eos": stopped,
                                   "stopped_by_string": stop_at < len(text),
                                   "truncated": not stopped and stop_at == len(text) and len(generated) == new_tokens,
                                   "prompt_tokens": int(inputs["attention_mask"][index].sum())})
                if progress_label:
                    progress_tokens += sum(len(row["token_ids"]) for row in output[start:])
                    progress_elapsed = time.monotonic() - progress_started
                    progress_done = len(output)
                    progress_eta = progress_elapsed * (len(prompts) - progress_done) / progress_done
                    print(f"GENERATION {progress_label} completed={progress_done}/{len(prompts)} "
                          f"elapsed_s={progress_elapsed:.1f} tokens={progress_tokens} "
                          f"tokens_per_s={progress_tokens / max(progress_elapsed, 1e-9):.2f} "
                          f"eta_s_estimate={progress_eta:.1f}", flush=True)
    finally:
        tokenizer.padding_side = previous
    return output


def retention(model, tokenizer, documents, *, reference=None):
    """Stream KL against a resident base model; never persist full-vocabulary logits."""
    torch = require_cuda()
    if len(documents) != 200:
        raise ValueError("Retention/KL require exactly 200 frozen documents")
    model.eval()
    if reference is not None:
        reference.eval()
    previous = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total_nll = total_kl = 0.0
    total_tokens, rows = 0, []
    try:
        with torch.inference_mode():
            for start in range(0, 200, 2):
                batch = documents[start:start + 2]
                inputs = tokenizer([row["text"] for row in batch], padding=True, truncation=True,
                                   max_length=768, return_tensors="pt").to("cuda:0")
                logits = model(**inputs, use_cache=False).logits
                target = inputs["input_ids"][:, 1:]
                mask = inputs["attention_mask"][:, 1:].bool()
                # Chunk the vocabulary logits by sequence to bound temporary FP32 memory.
                ref_logits = reference(**inputs, use_cache=False).logits if reference is not None else None
                nll_doc = torch.zeros(len(batch), device="cuda:0", dtype=torch.float64)
                kl_doc = torch.zeros_like(nll_doc)
                for offset in range(0, target.shape[1], 64):
                    sl = slice(offset, offset + 64)
                    logq = logits[:, :-1][:, sl].float().log_softmax(-1)
                    losses = -logq.gather(-1, target[:, sl, None]).squeeze(-1)
                    nll_doc += (losses * mask[:, sl]).double().sum(-1)
                    if ref_logits is not None:
                        logp = ref_logits[:, :-1][:, sl].float().log_softmax(-1)
                        kl = (logp.exp() * (logp - logq)).sum(-1)
                        kl_doc += (kl * mask[:, sl]).double().sum(-1)
                for j, row in enumerate(batch):
                    count = int(mask[j].sum())
                    if count == 0:
                        raise ValueError(f"No predictable retention tokens: {row['id']}")
                    rows.append({"id": row["id"], "tokens": count, "nll_sum": float(nll_doc[j]),
                                 "kl_sum": float(kl_doc[j]) if reference is not None else None})
                total_nll += float(nll_doc.sum())
                total_kl += float(kl_doc.sum())
                total_tokens += int(mask.sum())
                del logits, ref_logits
    finally:
        tokenizer.padding_side = previous
    return {"retention_nll": total_nll / total_tokens,
            "retention_kl": total_kl / total_tokens if reference is not None else None,
            "retention_tokens": total_tokens, "retention_documents": 200}, rows


def response_nll(model, tokenizer, rows, config):
    torch = require_cuda()
    total, count = 0.0, 0
    details = []
    with torch.inference_mode():
        for start in range(0, len(rows)):
            batch = rows[start:start + 1]
            features = [encode(row, tokenizer, config.task, 4096, response_only=True) for row in batch]
            padded = pad_features(features, tokenizer.pad_token_id)
            tensors = {k: torch.tensor(v, device="cuda:0") for k, v in padded.items()}
            labels = tensors.pop("labels")[:, 1:]
            logits = model(**tensors, use_cache=False).logits[:, :-1]
            sums = torch.zeros(len(batch), device="cuda:0", dtype=torch.float64)
            for offset in range(0, labels.shape[1], 64):
                sl = slice(offset, offset + 64)
                losses = torch.nn.functional.cross_entropy(logits[:, sl].float().transpose(1, 2), labels[:, sl],
                                                           ignore_index=-100, reduction="none")
                sums += losses.double().sum(-1)
            for i, row in enumerate(batch):
                n = int((labels[i] != -100).sum())
                if n == 0:
                    raise ValueError("No response tokens")
                value = float(sums[i])
                total += value
                count += n
                details.append({"id": row["id"], "nll_sum": value, "tokens": n})
    return {"response_nll": total / count, "response_tokens": count}, details


def result_identity(config, manifest, split, sandbox=None):
    return {"config_hash": config.identity, "adapter_files": manifest["adapter_files"],
            "bank_hash": file_hash(config.data_manifest), "source_hash": source_identity(),
            "protocol": EVAL_PROTOCOL, "split": split,
            "sandbox_manifest_sha256": file_hash(sandbox) if sandbox else None}


def base_control(config, tokenizer, reference, bank, split, evaluator, sandbox=None):
    """One small base-result cache per protocol/model/bank; no checkpoint/logit cache."""
    from .io import exclusive, read_json, writable_output
    identity = {"task": config.task, "model": config.model_id, "revision": config.model_revision,
                "bank": file_hash(config.data_manifest), "source": source_identity(),
                "protocol": EVAL_PROTOCOL, "split": split,
                "sandbox": file_hash(sandbox) if sandbox else None}
    output = writable_output(Path(config.output).parent / "_base_evaluations" / digest(identity))
    with exclusive(output, blocking=True):
        result = output / "metrics.json"
        if result.exists():
            cached = read_json(result)
            if cached["identity_hash"] != digest(identity):
                raise ValueError("Base evaluation identity mismatch")
            return cached["metrics"]
        metrics, records = evaluator(reference, tokenizer, config, bank, split,
                                     reference=None, output=output, sandbox=sandbox)
        save_evaluation(output, metrics, records, identity)
        return metrics


def with_base(metrics, baseline):
    result = {**metrics, "base_retention_nll": baseline["retention_nll"],
              "forgetting_nll_delta": metrics["retention_nll"] - baseline["retention_nll"]}
    if "task_accuracy" in metrics:
        result.update(base_task_accuracy=baseline["task_accuracy"],
                      task_gain=metrics["task_accuracy"] - baseline["task_accuracy"])
    for benchmark in ("humaneval", "mbpp"):
        if benchmark in metrics:
            result[benchmark] = {**metrics[benchmark], "base": baseline[benchmark],
                "gain_pass@1": metrics[benchmark]["pass@1"] - baseline[benchmark]["pass@1"],
                "gain_plus_pass@1": metrics[benchmark]["plus_pass@1"] - baseline[benchmark]["plus_pass@1"]}
    return result


def save_evaluation(output, metrics, records, identity):
    output = Path(output)
    for name, rows in records.items():
        write_rows(output / f"{name}.jsonl", rows)
    atomic_json(output / "metrics.json", {"status": "complete", "identity": identity,
                                           "identity_hash": digest(identity), "metrics": metrics})

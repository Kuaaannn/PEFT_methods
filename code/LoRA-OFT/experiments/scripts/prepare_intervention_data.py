"""Build frozen token banks for checkpoint interventions on a CUDA machine."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from loraoft.checkpoint_analysis import atomic_json, require_gpu
from loraoft.data.metamath import QUERY_TEMPLATE
from loraoft.eval_settings import (GSM8K_DEV_N, GSM8K_DO_SAMPLE,
                                    GSM8K_ENGINE_MAX_MODEL_LEN, GSM8K_MAX_LENGTH,
                                    GSM8K_MAX_NEW_TOKENS, GSM8K_NUM_BEAMS, GSM8K_SEED,
                                    GSM8K_TEMPERATURE)


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = {"qwen": "Qwen/Qwen2.5-7B", "llama": "meta-llama/Meta-Llama-3.1-8B"}
BASE_REFERENCES = {
    "qwen": ROOT / "results/base_reference_qwen25_7b.json",
    "llama": ROOT / "results/base_reference_llama31_8b.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="append", required=True, help="Training run directory; repeatable")
    parser.add_argument("--diagnostic-n", type=int, default=16)
    parser.add_argument("--mmlu-n", type=int, default=64)
    parser.add_argument("--snapshot", help="Prepared local model snapshot used for training")
    args = parser.parse_args()
    require_gpu()
    if args.diagnostic_n < 1 or args.mmlu_n < 1:
        parser.error("bank sizes must be positive")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    model_id = args.snapshot or MODEL_IDS[args.model_key]
    snapshot = Path(model_id) if args.snapshot else Path(snapshot_download(model_id))
    revision = snapshot.name
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    train = load_dataset("openai/gsm8k", "main", split="train")
    mmlu = load_dataset("cais/mmlu", "all", split="validation")
    retention_path = ROOT / "results" / "retention_bank.json"
    retention = json.loads(retention_path.read_text())
    base_reference = json.loads(BASE_REFERENCES[args.model_key].read_text())
    if (retention.get("n") != 200 or len(retention.get("rows", [])) != 200
            or base_reference.get("model_id") != model_id
            or base_reference.get("retention_bank_sha") != retention.get("sha256")
            or base_reference.get("n_rows") != 200
            or base_reference.get("max_length") != 768
            or base_reference.get("batch_size") != 2):
        raise RuntimeError("Standard retention bank/base-reference contract changed")
    selected = [{"run": run} for run in args.run]
    split_lists = []
    for task in selected:
        split = json.loads((Path(task["run"]) / "split_ids.json").read_text())
        if (split.get("group_holdout") != GSM8K_DEV_N
                or len(split["valid_indices"]) != GSM8K_DEV_N):
            raise RuntimeError(f"Run does not have the standard full dev split: {task['run']}")
        split_lists.append(split["valid_indices"])
    if not split_lists or any(indices != split_lists[0] for indices in split_lists[1:]):
        raise RuntimeError("All selected runs must share one frozen dev split")
    dev_indices = split_lists[0]
    dev = train.select(dev_indices)

    def answer_row(example, item_id, limit=768):
        prompt = QUERY_TEMPLATE.format(query=example["question"])
        encoded = tokenizer(prompt + example["answer"], add_special_tokens=True,
                            return_offsets_mapping=True, truncation=True, max_length=limit)
        mask = [int(start >= len(prompt) and end > start)
                for start, end in encoded["offset_mapping"]]
        if sum(mask) == 0:
            raise RuntimeError(f"No answer tokens in {item_id}")
        return {"id": item_id, "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"], "score_mask": mask}

    target_rows = [answer_row(dev[i], f"target-dev/{i}")
                   for i in range(min(args.diagnostic_n, len(dev)))]
    dev_set = set(dev_indices)
    selection_indices = [i for i in range(len(train)) if i not in dev_set][:args.diagnostic_n]
    selection_rows = [answer_row(train[i], f"selection-train/{i}")
                      for i in selection_indices]
    general_rows = []
    for i, text in enumerate(retention["rows"][:args.diagnostic_n]):
        encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=192)
        general_rows.append({"id": f"general-finewiki/{i}",
                             "input_ids": encoded["input_ids"],
                             "attention_mask": encoded["attention_mask"],
                             "score_mask": list(encoded["attention_mask"])})

    target_outcomes = []
    for position, example in enumerate(dev):
        prompt = QUERY_TEMPLATE.format(query=example["question"])
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        budget = max(1, min(GSM8K_MAX_NEW_TOKENS, GSM8K_MAX_LENGTH - len(prompt_ids)))
        target_outcomes.append({"id": f"gsm8k-dev/{position}", "type": "gsm8k",
                                "task": "gsm8k_dev", "prompt_ids": prompt_ids,
                                "gold": example["answer"], "max_new_tokens": budget,
                                "source_train_index": int(dev_indices[position])})

    letters = "ABCD"
    general_outcomes = []
    for i in range(min(args.mmlu_n, len(mmlu))):
        example = mmlu[i]
        prompt = (example["question"] + "\n" +
                  "\n".join(f"{letters[j]}. {choice}"
                              for j, choice in enumerate(example["choices"])) + "\nAnswer:")
        choices = []
        for letter in letters:
            encoded = tokenizer(prompt + " " + letter, add_special_tokens=True,
                                return_offsets_mapping=True, truncation=True, max_length=768)
            mask = [int(start >= len(prompt) and end > start)
                    for start, end in encoded["offset_mapping"]]
            choices.append({"id": f"mmlu-validation/{i}/{letter}",
                            "input_ids": encoded["input_ids"],
                            "attention_mask": encoded["attention_mask"],
                            "score_mask": mask})
        general_outcomes.append({"id": f"mmlu-validation/{i}", "type": "choice",
                                 "task": "mmlu_validation", "gold_index": int(example["answer"]),
                                 "scoring": "sum_logprob", "choices": choices,
                                 "subject": example.get("subject")})

    retention_hash = hashlib.sha256(retention_path.read_bytes()).hexdigest()
    bank = {
        "model_id": model_id, "tokenizer_revision": revision,
        "purpose": "Checkpoint interventions; full standard GSM8K dev outcomes",
        "eos_policy": "no inserted EOS; frozen truncation occurs only while building diagnostic contexts",
        "mask_policy": "label_indexed_answer_tokens/general_all_valid",
        "generation": {"prompt": QUERY_TEMPLATE, "do_sample": GSM8K_DO_SAMPLE,
                       "temperature": GSM8K_TEMPERATURE, "seed": GSM8K_SEED,
                       "num_beams": GSM8K_NUM_BEAMS,
                       "max_new_tokens": GSM8K_MAX_NEW_TOKENS,
                       "max_total_tokens": GSM8K_MAX_LENGTH,
                       "standard_engine_max_model_len": GSM8K_ENGINE_MAX_MODEL_LEN,
                       "scorer": "loraoft.data.metamath.is_correct",
                       "standard_eval_scripts": ["eval/run_eval.py"]},
        "execution_numerics": {
            "standard_eval": "merged BF16 weights in vLLM",
            "checkpoint_protocol": "merged BF16 weights in vLLM for GSM8K generation; BF16 Transformers SDPA forward compute for retention NLL, diagnostic NLL/KL, choice scoring, and activation geometry; FP32 intervention factors/SVD and full-vocabulary log-softmax; reference logits reconstructed from native-dtype final hidden states through the unchanged output embedding",
            "reason": "GSM8K uses the standard vLLM engine while protocol measurements retain the logits and hooks they require",
            "semantic_settings_match": True},
        "standard_retention": {
            "source": retention["source"],
            "source_file_sha256": retention_hash,
            "content_sha256": retention["sha256"],
            "n_rows": 200,
            "max_length": 768,
            "batch_size": 2,
            "aggregation": "token_pooled_mean_nll",
            "base_retention_nll": base_reference["base_retention_nll"],
            "kl_n_rows": 200,
            "kl_max_length": 768,
            "kl_vocabulary": "full",
            "kl_aggregation": "example_mean_and_token_pooled",
        },
        "selection": {"source": "cached GSM8K train examples outside the 1,000-example holdout; numerical-floor contexts only",
                      "rows": selection_rows},
        "target_eval": {"source": "the exact 1,000-example group-held-out GSM8K train-derived dev split used by standard post-hoc evaluation",
                        "rows": target_rows, "outcomes": target_outcomes},
        "general_eval": {"source": f"frozen FineWiki retention bank sha256={retention_hash}; first {args.mmlu_n} cached MMLU validation rows",
                         "rows": general_rows, "outcomes": general_outcomes},
        "public_gsm8k_test_status": "locked and unused",
        "target_kl_context_n": args.diagnostic_n,
        "activation_context_n_per_role": args.diagnostic_n,
        "target_outcome_n": len(target_outcomes),
        "general_outcome_n": len(general_outcomes),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, bank)
    print(json.dumps({"bank": str(destination), "model_id": model_id,
                      "target_outcomes": len(target_outcomes),
                      "target_rows": len(target_rows), "general_rows": len(general_rows),
                      "general_outcomes": len(general_outcomes)}))


if __name__ == "__main__":
    main()

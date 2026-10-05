from __future__ import annotations

import re
from pathlib import Path

from .io import digest, file_hash, read_json, read_rows


def group_key(row):
    # Exclude labels so differently-labelled copies cannot straddle train/dev/test.
    instruction = row.get("instruction", row.get("prompt", ""))
    return digest([re.sub(r"\s+", " ", instruction).strip(),
                   re.sub(r"\s+", " ", row.get("input", "")).strip()])


def split_groups(rows, tests, *, train_n=20000, dev_n=2000, seed=2027, stratified=False,
                 response_policy="strict"):
    if response_policy not in ("strict", "classification", "generative"):
        raise ValueError("Unknown duplicate-response policy")
    forbidden = {group_key(row) for row in tests}
    groups, members, excluded = {}, {}, 0
    conflicts, alternatives = set(), set()
    for row in rows:
        key = group_key(row)
        if key in forbidden:
            excluded += 1
            continue
        if key in groups:
            members[key].append(row.get("id", key))
            first = groups[key]
            if first.get("task") != row.get("task"):
                raise ValueError("Conflicting labels/tasks in duplicate group")
            if response_policy == "strict" and first["output"] != row["output"]:
                raise ValueError("Conflicting labels/tasks in duplicate group")
            if response_policy == "classification":
                if not row.get("answer") or not first.get("answer"):
                    raise ValueError("Classification deduplication requires validated answer labels")
                if first["answer"] != row["answer"]:
                    conflicts.add(key)
            if first["output"] != row["output"]:
                alternatives.add(key)
            continue
        if response_policy == "classification" and not row.get("answer"):
            raise ValueError("Classification deduplication requires validated answer labels")
        groups[key] = {**row, "id": row.get("id", key), "group_id": key}
        members[key] = [row.get("id", key)]
    # Remove the whole ambiguous classification group, not an arbitrarily chosen label.
    # Generative alternatives retain the first response in the immutable source order.
    conflict_audit = [{"group_id": key, "source_ids": members[key]} for key in sorted(conflicts)]
    alternative_audit = [{"group_id": key, "kept_id": groups[key]["id"], "source_ids": members[key]}
                         for key in sorted(alternatives - conflicts)]
    for key in conflicts:
        del groups[key]
    duplicates = sum(len(members[key]) - 1 for key in groups)
    ordered = sorted(groups.values(), key=lambda row: digest([seed, row["group_id"]]))
    if stratified:
        raise ValueError("Stratified splitting is not supported for coding")
    dev, train = ordered[:dev_n], ordered[dev_n:dev_n + train_n]
    if len(train) != train_n or len(dev) != dev_n:
        raise ValueError("Not enough deduplicated, uncontaminated source examples")
    return train, dev, {"source_rows": len(rows), "exact_test_overlap_removed": excluded,
                        "duplicates_removed": duplicates, "eligible_unique_prompts": len(groups),
                        "response_policy": response_policy,
                        "representative_rule": "first_row_in_pinned_source_order",
                        "conflicting_prompt_groups_removed": len(conflicts),
                        "conflicting_rows_removed": sum(len(members[key]) for key in conflicts),
                        "conflicting_groups": conflict_audit,
                        "alternative_response_groups": len(alternative_audit),
                        "alternative_response_choices": alternative_audit,
                        "near_duplicate_audit": "not_established_by_exact_normalized_matching"}


def load_bank(path, task):
    path = Path(path)
    manifest = read_json(path)
    if manifest.get("schema") != 1 or manifest.get("task") != task:
        raise ValueError("Wrong data bank schema/task")
    rows = {}
    for name in ("train", "dev", "test", "retention"):
        item = manifest["files"][name]
        file = path.parent / item["path"]
        if file_hash(file) != item["sha256"]:
            raise ValueError(f"Data bank changed: {name}")
        rows[name] = read_rows(file)
        if len(rows[name]) != item["n"] or not rows[name]:
            raise ValueError(f"Wrong count in {name}")
        ids = [row["id"] for row in rows[name]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate IDs in {name}")
    if len(rows["retention"]) != 200 or any(not r.get("text", "").strip() for r in rows["retention"]):
        raise ValueError("Retention bank must contain 200 nonempty documents")
    train_keys, dev_keys = ({group_key(r) for r in rows[k]} for k in ("train", "dev"))
    test_keys = {group_key(r) for r in rows["test"]}
    if train_keys & dev_keys or (train_keys | dev_keys) & test_keys:
        raise ValueError("Train/dev/test overlap")
    return manifest, rows


class ResponseTruncatedError(ValueError):
    """No response targets remain; only coding training may exclude this row."""
    def __init__(self, row_id, raw_length):
        super().__init__(f"Response entirely truncated for {row_id}; increase max_length")
        self.raw_length = raw_length


def encode(row, tokenizer, task, max_length, *, response_only=None):
    if task == "coding":
        from ..coding.protocol import prompt
        source, target = prompt(row), row["output"] + "\n"
        mask_source = True
    else:
        raise ValueError(task)
    # Offsets handle a BPE token spanning the source/response boundary safely.
    encoded = tokenizer(source + target, add_special_tokens=True, truncation=False,
                        return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    response = [start >= len(source) and end > start for start, end in offsets]
    raw_length = len(ids)
    ids, response = ids[:max_length], response[:max_length]
    # Match DoRA: truncate first; append EOS only when there is room. Never
    # replace the last content token or count an unappended EOS as truncation.
    if ids and ids[-1] != tokenizer.eos_token_id and len(ids) < max_length:
        ids.append(tokenizer.eos_token_id)
        response.append(True)
    response_tokens = sum(response[1:])
    # DoRA trains on the entire retained prefix, including prompt-only rows.
    # Response-only coding and dev NLL must still have supervised answer tokens.
    if mask_source and not response_tokens:
        raise ResponseTruncatedError(row.get("id"), raw_length)
    labels = [token if not mask_source or keep else -100 for token, keep in zip(ids, response)]
    if not any(label != -100 for label in labels[1:]):
        raise ValueError(f"No supervised next-token targets for {row.get('id')}")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels,
            "truncated": raw_length > max_length, "raw_length": raw_length,
            "response_tokens": response_tokens,
            "response_entirely_truncated": response_tokens == 0}


def pad_features(features, pad_id, *, multiple=8):
    if not features or any(not f["input_ids"] for f in features):
        raise ValueError("Cannot collate empty examples")
    width = max(len(f["input_ids"]) for f in features)
    width = ((width + multiple - 1) // multiple) * multiple
    output = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for feature in features:
        n = len(feature["input_ids"])
        if len(feature["labels"]) != n:
            raise ValueError("Label alignment mismatch")
        output["input_ids"].append(feature["input_ids"] + [pad_id] * (width - n))
        output["attention_mask"].append([1] * n + [0] * (width - n))
        output["labels"].append(feature["labels"] + [-100] * (width - n))
    return output


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in pad_features(features, self.pad_id).items()}


def training_examples(rows, tokenizer, config):
    examples, excluded, lengths = [], [], []
    for row in rows:
        try:
            encoded = encode(row, tokenizer, config.task, config.max_length)
        except ResponseTruncatedError as exc:
            if config.task != "coding":
                raise
            # Zero-target rows cannot contribute to response-only SFT. Do not
            # train on their prompts, invent targets, or filter any eval rows.
            excluded.append({"id": row["id"], "reason": str(exc), "raw_length": exc.raw_length})
            lengths.append(exc.raw_length)
            continue
        lengths.append(encoded["raw_length"])
        examples.append(encoded)
    lengths.sort()
    audit = {"source_n": len(rows), "n": len(examples), "excluded_rows": len(excluded), "excluded": excluded,
             "truncated": sum(r["truncated"] for r in examples),
             "source_truncated": sum(n > config.max_length for n in lengths),
             "response_entirely_truncated": len(excluded) + sum(r["response_entirely_truncated"] for r in examples),
             "lengths_measured": len(lengths),
             "p95_raw_tokens": lengths[int(.95 * (len(lengths) - 1))] if lengths else None,
             "max_raw_tokens": max(lengths, default=0),
             "supervised_tokens": sum(sum(t != -100 for t in r["labels"][1:]) for r in examples)}
    return [{key: row[key] for key in ("input_ids", "attention_mask", "labels")} for row in examples], audit


def training_audit_errors(task, audit):
    """The same bounded exclusion/truncation checks before smoke and training."""
    errors = []
    if not audit["n"]:
        errors.append("No supervised training examples remain")
    if task == "coding":
        if audit["excluded_rows"] > .01 * audit["source_n"]:
            errors.append("More than 1% of coding training examples have no response targets; review context")
        if audit["truncated"] > .05 * audit["n"]:
            errors.append("More than 5% of retained examples truncate; review context")
    return errors

"""PiSSA instruction SFT, separate from the frozen EvalPlus base prompts."""


def prompt(row):
    if row.get("input", "").strip():
        raise ValueError("PiSSA Python protocol expects instruction-only input; do not discard context")
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{row['instruction']}\n\n### Response:")


def validate_benchmarks(rows):
    ids = [(row["benchmark"], row["task_id"]) for row in rows]
    if len(set(ids)) != len(ids) or {x[0] for x in ids} != {"humaneval", "mbpp"}:
        raise ValueError("Expected unique HumanEval and MBPP task IDs")
    for row in rows:
        if not row.get("prompt") or not row.get("entry_point"):
            raise ValueError("Missing official EvalPlus prompt/entry point")

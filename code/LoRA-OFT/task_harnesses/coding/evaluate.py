from pathlib import Path

from .protocol import validate_benchmarks
from .sandbox import score_samples, validate_sandbox
from ..shared.evaluation import generate, response_nll, retention
from ..shared.io import read_json, write_rows


def evaluate(model, tokenizer, config, bank, split, *, reference, output, sandbox=None):
    metrics, records = {}, {}
    progress_role = f"model={config.model_key} role={'base' if reference is None else 'adapted'}"
    if split == "dev":
        metrics, records["response_nll"] = response_nll(model, tokenizer, bank["dev"], config)
    elif split == "test":
        if sandbox is None:
            raise ValueError("Coding test evaluation requires a pinned, isolated EvalPlus image")
        validate_sandbox(sandbox)
        from evalplus.provider.utility import EOS, extra_eos_for_direct_completion
        from evalplus.sanitize import sanitize
        validate_benchmarks(bank["test"])
        information = read_json(config.data_manifest)["evalplus"]
        for benchmark in ("humaneval", "mbpp"):
            rows = [r for r in bank["test"] if r["benchmark"] == benchmark]
            prompts = [r["prompt"].strip() + "\n" for r in rows]
            # Single-example greedy generation matches EvalPlus's reference HF path.
            generated = generate(model, tokenizer, prompts, beams=1, new_tokens=1024, batch_size=1,
                                 stop_strings=EOS + extra_eos_for_direct_completion(benchmark),
                                 progress_label=f"{progress_role} benchmark={benchmark}")
            raw, samples = [], []
            for row, prompt, generation in zip(rows, prompts, generated):
                solution = prompt + generation["text"].replace("\t", "    ")
                raw.append({**generation, "task_id": row["task_id"], "solution": solution})
                samples.append({"task_id": row["task_id"],
                                "solution": sanitize(solution, entrypoint=row["entry_point"])})
            write_rows(Path(output) / f"{benchmark}.raw.jsonl", raw)
            write_rows(Path(output) / f"{benchmark}.samples.jsonl", samples)
            records[benchmark + "_predictions"] = raw
            print(f"EVAL_STAGE {progress_role} benchmark={benchmark} grading=start", flush=True)
            metrics[benchmark] = score_samples(samples, benchmark, information[benchmark], sandbox, output)
            print(f"EVAL_STAGE {progress_role} benchmark={benchmark} grading=done scores={metrics[benchmark]}", flush=True)
    else:
        raise ValueError(split)
    print(f"EVAL_STAGE {progress_role} retention=start documents=200", flush=True)
    general, general_records = retention(model, tokenizer, bank["retention"], reference=reference)
    print(f"EVAL_STAGE {progress_role} retention=done", flush=True)
    return {**metrics, **general}, {**records, "retention": general_records}


def main():
    from ..shared.evaluate_cli import main as entry
    entry("coding", evaluate)


if __name__ == "__main__":
    main()

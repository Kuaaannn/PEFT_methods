import argparse
from pathlib import Path

from .checkpoint import load_run
from .data import load_bank
from .evaluation import base_control, result_identity, save_evaluation, with_base
from .io import digest, exclusive, read_json, writable_output
from .runtime import base_model, require_cuda


def main(task, evaluator):
    parser = argparse.ArgumentParser(description=f"Standard {task} evaluation")
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sandbox", help="Pinned Apptainer image manifest, required for coding test")
    args = parser.parse_args()
    require_cuda()
    if task == "coding" and args.split == "test":
        if not args.sandbox:
            parser.error("--sandbox is required for coding test")
        from ..coding.sandbox import validate_sandbox
        validate_sandbox(args.sandbox)
    output = writable_output(args.output)
    with exclusive(output):
        config, manifest, model, tokenizer = load_run(args.run)
        if config.task != task:
            raise ValueError("Wrong task evaluator")
        identity = result_identity(config, manifest, args.split, args.sandbox)
        existing = output / "metrics.json"
        if existing.exists():
            if read_json(existing)["identity_hash"] != digest(identity):
                raise ValueError("Output belongs to another evaluation")
            return
        _, bank = load_bank(config.data_manifest, task)
        reference = base_model(config).eval()
        baseline = base_control(config, tokenizer, reference, bank, args.split, evaluator, args.sandbox)
        metrics, records = evaluator(model, tokenizer, config, bank, args.split,
                                     reference=reference, output=output, sandbox=args.sandbox)
        save_evaluation(output, with_base(metrics, baseline), records, identity)

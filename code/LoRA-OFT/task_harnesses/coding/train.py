"""Independent PiSSA Python-protocol Trainer entry point, response-only loss."""
import argparse
import os
import time
from pathlib import Path

from ..shared.config import RunConfig
from ..shared.data import Collator
from ..shared.io import exclusive
from ..shared.runtime import require_cuda
from ..shared.training import arguments, begin, finish


def run(config):
    require_cuda()
    from transformers import Trainer
    with exclusive(config.output):
        spectral = None
        succeeded = False
        try:
            model, tokenizer, spectral, examples, manifest = begin(config)
            train_args = arguments(config, len(examples))
            train_args.report_to = []
            train_args.run_name = Path(config.output).name
            trainer = Trainer(model=model, args=train_args,
                              train_dataset=examples, processing_class=tokenizer,
                              data_collator=Collator(tokenizer.pad_token_id))
            start = time.perf_counter()
            trainer.train()
            finish(config, model, tokenizer, spectral, trainer, manifest, time.perf_counter() - start)
            succeeded = True
        finally:
            if spectral is not None:
                spectral.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(RunConfig.load(args.config, "coding"))


if __name__ == "__main__":
    main()

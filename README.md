# Are Parameter-Efficient Fine-tuning Methods Really Different?

Experiment code by Yikuan Li, Pinyan Lu and Fanghui Liu.

This repository provides experiments for LoRA, DoRA, PiSSA, MiLoRA, OFT and HRA. It covers mathematics and coding adaptation, FLUX cat personalization, the 28-object study, geometric measurements, spectral restoration and component interventions, and synthetic experiments.

## Requirements

Use Linux, Python 3.12 and CUDA GPUs with BF16 support for model experiments. Training uses one GPU per run. The experiments use A100 GPUs. Mathematics interventions require two GPUs, one for analysis and one for vLLM evaluation. Synthetic experiments run on a CPU.

Obtain access to the gated Hugging Face models using your own account. After installing the training environment, authenticate with `.venvs/train/bin/hf auth login`. Coding evaluation also requires Apptainer or Singularity.

## Quick start

Run these commands from the repository root. On a machine with multiple GPUs, expose the intended device with `CUDA_VISIBLE_DEVICES=0`.

```bash
python3 scripts/setup.py train
python3 scripts/setup.py eval
python3 scripts/prepare.py math --model qwen
python3 scripts/prepare.py math --model qwen --stage data

# Train all five learning rates and three seeds for this configuration
python3 scripts/train.py math --model qwen --method lora --level 0 --keep-going
python3 scripts/evaluate_runs.py math dev --model qwen --method lora --level 0
python3 scripts/choose_lr.py math --model qwen --method lora --level 0
python3 scripts/evaluate_runs.py math test
python3 scripts/report.py math
```

See [the experiment guide](docs/experiments.md) for the full grids, coding, images, baselines and interventions. [Environment and data notes](docs/environment.md) describe dependencies and storage.

## Running selected configurations

- Add `--dry-run` to preview commands without downloads or model execution.
- Filter training with `--model`, `--method`, `--level`, `--lr` and `--seed`. Capacity levels start at zero.
- List jobs with `python3 scripts/grid.py math`, then use `--index N` to run an individual configuration.
- Pass the same `--output /path/to/runs` to each stage. The default is `runs/` in the repository. For a custom output directory inside the repository, add that directory to your local Git ignore rules.
- Remove the filters to run the complete grid. LR selection requires all three seeds at a candidate LR and every subject for the object study.

The full grids contain 540 mathematics runs, 180 coding runs, 576 cat runs and 2,016 object runs. The HRA extension adds 18 runs. Selection uses development scores only. Image training records both splits, and selection reads validation scores at step 700. Reports average objects within each seed before computing the mean and sample standard deviation over seeds.

## Synthetic experiments

```bash
python3 scripts/setup.py analysis
python3 scripts/theory.py --plots
```

Plotting requires a complete TeX installation with `latex`, `dvipng` and the Times font packages. Outputs are written under the chosen output directory. Omit `--plots` to run the numerical experiments without TeX.

The `data/` directory contains the frozen 200-document retention sample used by the language experiments.

## Acknowledgments

Our code is based on the [Hugging Face PEFT library](https://github.com/huggingface/peft) and its [MetaMathQA benchmark for LLM mathematics](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA) and [image generation benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/image-gen). We thank the PEFT authors and contributors for these open-source implementations.

Please also credit these upstream projects when building on this code. Source links and attribution are provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), with references in [CITATION.cff](CITATION.cff). Upstream licenses and copyright notices are retained.

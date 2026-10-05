# Environments and data

The launchers use separate environments for language training, language generation and image experiments. Language training and merging use the bundled language PEFT build. GSM8K generation uses vLLM 0.10.2 and Transformers 4.56.2 in a separate process. Images use the bundled image PEFT snapshot with Diffusers 0.40.0 and Transformers 5.17.0. The installation commands verify the active LoRA, OFT and HRA source files.

Do not replace the two PEFT builds with one stock installation. OFT uses unshared blocks and five Cayley–Neumann terms. HRA uses compact-WY execution. PiSSA and MiLoRA use the original spectral initialization and portable difference-adapter export.

`scripts/setup.py` creates environments under `.venvs/`. Models use exact revisions recorded in `scripts/common.py`. Prepared assets are recorded under the chosen output directory. Hugging Face downloads default to `.cache/huggingface` and respect an existing `HF_HOME`. Keep sufficient host RAM and disk for model snapshots, merged language checkpoints and temporary SVD factors. The launchers do not reduce precision, context, batch size or matrix coverage automatically.

The frozen 200-document FineWiki sample is in `data/retention_bank.json`, with its original content checksum and source identifier. Both language tasks use the same documents. Math training uses the original MetaMath/GSM8K loading and deterministic splitting code, which records the selected indices in every run. MetaMath and the GSM8K training split use the dataset loader defaults. The official GSM8K test, coding data, object images and COCO retention shard use their saved immutable revisions.

Public inputs are obtained from the original repositories.

On a machine with several GPUs, set `CUDA_VISIBLE_DEVICES=0` for a single-GPU job. Coding requires exactly one visible GPU. The mathematics intervention command explicitly assigns its two workers with `--gpus`.

To use an existing compatible environment, set `PAPER_TRAIN_PYTHON`, `PAPER_EVAL_PYTHON`, `PAPER_FLUX_PYTHON` or `PAPER_ANALYSIS_PYTHON` to its Python executable. Keep the executable path inside the virtual environment. Model environments still need the bundled PEFT runtime checks.

- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- [Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- [FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) and [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki)
- [PiSSA data](https://huggingface.co/datasets/fxmeng/pissa-dataset) and [EvalPlus](https://github.com/evalplus/evalplus)
- [Cat images](https://huggingface.co/datasets/peft-internal-testing/cat-image-dataset), [DreamBooth objects](https://huggingface.co/datasets/google/dreambooth) and [COCO mirror](https://huggingface.co/datasets/sayakpaul/coco-30-val-2014)

Model and dataset licenses remain with their respective providers.

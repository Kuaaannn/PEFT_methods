# Running the experiments

Run commands from the release root. All stages accept `--output` and `--dry-run`. Full commands below run sequentially. For parallel execution, assign distinct indices from `scripts/grid.py` to independent GPU jobs. Keep the same output root and expose only the intended GPU with `CUDA_VISIBLE_DEVICES`.

## Mathematics

```bash
python3 scripts/setup.py train
python3 scripts/setup.py eval
python3 scripts/prepare.py math
python3 scripts/prepare.py math --stage data
python3 scripts/baselines.py math
python3 scripts/train.py math --keep-going
python3 scripts/evaluate_runs.py math dev
python3 scripts/choose_lr.py math
python3 scripts/evaluate_runs.py math test
python3 scripts/report.py math
```

This uses Qwen2.5-7B and Llama-3.1-8B, 20,000 MetaMath examples, 625 updates, sequence length 768, batch size 2 and accumulation 16. The fixed 1,000-example group holdout prevents MetaMath rewrites of development questions from entering training. Final accuracy uses all 1,319 official GSM8K test examples with the original merged-weight vLLM evaluator.

The three capacity levels use additive ranks 7/14/28 for Qwen and 7/15/30 for Llama, OFT blocks 32/64/128, and HRA ranks 16/32/64. Learning rates are 1e-5, 3e-5, 1e-4, 3e-4 and 1e-3. Seeds are 13, 37 and 73. Selection maximizes mean development accuracy, with lower LR breaking ties.

Run the Llama HRA extension after the main grid.

```bash
python3 scripts/train.py hra-extension --keep-going
python3 scripts/evaluate_runs.py hra-extension dev
python3 scripts/choose_lr.py hra-extension
python3 scripts/evaluate_runs.py hra-extension test
python3 scripts/report.py hra-extension
```

The extension adds 3e-3 and 1e-2 and compares them with the main-grid HRA candidates. Its selection and reports stay separate.

## Coding

```bash
python3 scripts/prepare.py coding
python3 scripts/prepare.py coding --stage data
python3 scripts/prepare.py coding --stage sandbox
python3 scripts/train.py coding --keep-going
python3 scripts/evaluate_runs.py coding dev
python3 scripts/choose_lr.py coding
python3 scripts/evaluate_runs.py coding test
python3 scripts/report.py coding
```

Use the language training environment installed above. Preparation downloads the pinned PiSSA Python data and EvalPlus source. Data preparation creates the original 20,000/2,000 train/development split and checks overlap with evaluation prompts. The coding configuration uses one epoch, effective batch size 32, microbatch size 1 and maximum training length 1,024. Selection minimizes token-pooled development response NLL. Seeds and LR candidates match mathematics.

All additive methods train at rank 7, OFT uses block size 32 and HRA uses rank 16. The exported rank-14 PiSSA/MiLoRA adapter represents the trained rank-7 difference and is not a larger training budget.

The coding evaluator also computes and caches the pretrained baseline. Mathematics and images expose the same comparison through `baselines.py`.

HumanEval and MBPP evaluation requires Apptainer or Singularity. The preparation command checks the pinned evaluator and its isolation before generated programs are executed. Test evaluation uses one greedy completion per problem and evaluates the original and extended tests on the same completions. Do not bypass the sandbox.

## Image generation

```bash
python3 scripts/setup.py flux
python3 scripts/prepare.py cat
python3 scripts/prepare.py cat --stage data
python3 scripts/baselines.py cat
python3 scripts/train.py cat --keep-going
python3 scripts/evaluate_runs.py cat dev
python3 scripts/choose_lr.py cat
python3 scripts/evaluate_runs.py cat test
python3 scripts/report.py cat

python3 scripts/prepare.py objects
python3 scripts/prepare.py objects --stage data
python3 scripts/baselines.py objects
python3 scripts/train.py objects --keep-going
python3 scripts/evaluate_runs.py objects dev
python3 scripts/choose_lr.py objects
python3 scripts/evaluate_runs.py objects test
python3 scripts/report.py objects
```

Both studies use FLUX.2-klein-base-4B, 750 updates, batch size 2, resolution 512, 20 inference steps and guidance 3.5. Cat personalization runs all six methods. The 28-object study runs LoRA and OFT and verifies the original image bytes before training.

Cat LR candidates are 5e-6, 1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4 and 1e-3. Object candidates are 3e-5, 5e-5 and 1e-4. Both use seeds 0, 1 and 2. Capacity levels use additive ranks 4/8/16/32, OFT blocks 32/64/128/256 and HRA ranks 16/32/62/124.

The selector maximizes mean validation DINO at step 700. For objects it chooses one LR per method/capacity across all 28 subjects and seeds. Test and retention use step 750. The original trainer already performs these evaluations, so `evaluate_runs.py` extracts the corresponding recorded results without generating a different sample. COCO retention caches contain 200 pairs. Incompatible subject-specific drift caches are rejected by the original evaluator, which generates the matching base reference.

## Geometry and interventions

Run these after development selection and base-reference preparation. Language geometry uses the extended-grid Llama HRA selections, so complete and select the HRA extension first.

```bash
python3 scripts/geometry.py math banks
python3 scripts/geometry.py math intervene --gpus 0,1
python3 scripts/geometry.py math rotations
python3 scripts/geometry.py math he

python3 scripts/geometry.py coding intervene

python3 scripts/geometry.py cat intervene
python3 scripts/geometry.py cat rotations
python3 scripts/geometry.py cat he
```

Mathematics interventions preserve the original two-process evaluator and operate on every adapted matrix. Use two CUDA GPUs and allow substantial local scratch space for SVD factors, typically at least 160 GiB. The launcher starts and stops its own evaluation worker. Other commands use the visible CUDA device.

The active 12-cell intervention panel includes pretrained-spectrum restoration, the trained and reconstruction controls, spectrum-only and orientation-only edits, and restoration/preservation of dominant, intermediate and trailing components. Coding uses its original nine-cell panel.

Cat interventions and hyperspherical-energy measurements use the largest selected capacity. Mathematics HE uses the largest capacity for both backbones. Rotation measurements reuse fixed base-SVD references. Existing operators, merge conventions, precision and all-matrix coverage are preserved.

HE writes matrix measurements and seed summaries under `he/<group>/report`. Each analysis subset has its own directory. Rotation outputs contain both the fixed spectral-block measurements and the sensitivity measurements using the matched OFT block width.

## Results and reruns

`selected.json` freezes the chosen checkpoints and development files. `summary.csv` contains selected test means and sample standard deviations. Raw outputs remain available beneath each run. Checkpoint and evaluation identities are verified before selection and reporting. A different partial selection needs a separate output root.

`--keep-going` retains a failure record and continues independent training cells. Inspect failures before selection. Missing evaluations without a recorded training failure stop selection.

Completed training is reused. Partial or failed training directories are not silently overwritten. Use a new output directory when rerunning a failed cell.

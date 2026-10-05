"""MetaMathQA -> GSM8K, in PEFT's method_comparison setting with one deliberate change.

Splits, filter and prompt template match PEFT's `method_comparison/MetaMathQA/data.py`
so our numbers are comparable to their 63 published results, EXCEPT:

    VALID_SIZE 50 -> 500

PEFT uses 50 because it evaluates with `transformers.generate`, which is slow. At p ~ 0.5
that is a binomial SD of 7.1 percentage points on a single evaluation -- their own logged
curve moves 0.32 -> 0.36 -> 0.50, barely outside that noise. Selecting a learning rate
across 5 candidates x 3 seeds on a +-7pp metric selects noise, not a learning rate. We
evaluate with vLLM, where 500 costs seconds and gives SD ~ 2.2pp.

The 500 are the FIRST 500 indices of the SAME seeded shuffle, so our validation set is a
strict superset of theirs and the harness-validation cell stays comparable. The GSM8K
test split is untouched and stays locked until selection completes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation

import numpy as np

# Identical to PEFT's data.py: with a 768-token limit on query+response, excluding texts
# longer than 1300 characters retains 93.8% of the dataset.
CHAR_LIMIT = 1300
VALID_SIZE = 500                 # PEFT uses 50; see module docstring
VALID_SHUFFLE_SEED = 0           # same seed as PEFT, so the sets nest
QUERY_TEMPLATE = "Question: {query} Think step by step.\nAnswer:"


@dataclass
class SplitIds:
    """Frozen example identities, so a split can be audited rather than trusted.

    Recording example indices lets later runs verify that they used the same data
    even if the dataset revision moves.
    """

    valid_indices: list[int]
    n_train: int
    n_test: int
    train_revision: str | None = None
    test_revision: str | None = None
    train_truncated_to: int | None = None   # test-only cap; None for scientific runs
    group_holdout: int | None = None        # dev problems whose rewrites left training


def filter_by_char_limit(ds, print_fn=print):
    lengths = [len(f"{q} {r}") for q, r in zip(ds["query"], ds["response"])]
    keep = [i for i, n in enumerate(lengths) if n <= CHAR_LIMIT]
    print_fn(f"MetaMathQA filter: kept {100 * len(keep) / len(ds):.1f}%")
    return ds.select(keep)


def build_group_holdout(dev_size: int = 1000, seed: int = VALID_SHUFFLE_SEED):
    """Dev problems from GSM8K train, plus the set of MetaMathQA rows to REMOVE.

    MetaMathQA is ~28 rewrites each of 13,929 source problems, and `original_question`
    is the group key. Splitting by ROW leaks: 98.8% of a GSM8K-train sample appears
    verbatim as a source problem, so the model has already seen the problem and a worked
    solution. Splitting by GROUP -- holding out problems and deleting every rewrite of
    them -- makes dev as clean as GSM8K test (measured max-similarity 0.189 vs 0.183).

    Costs 8.28% of a pool we consume 5.5% of, so nothing real.
    """
    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main")["train"]
    rng = np.random.RandomState(seed)
    idx = np.arange(len(gsm))
    rng.shuffle(idx)
    dev_idx = idx[:dev_size].tolist()
    dev_questions = {gsm[int(i)]["question"] for i in dev_idx}
    return dev_idx, dev_questions


def load_splits(tokenizer, max_seq_length: int = 768, valid_size: int = VALID_SIZE,
                print_fn=print, max_train_examples: int | None = None,
                group_holdout: int | None = None):
    """Return (train, valid, test, SplitIds).

    train  MetaMathQA, char-filtered, tokenized WITH the answer (supervised)
    valid  GSM8K train sample, tokenized WITHOUT the answer (generated at eval)
    test   GSM8K test, full, tokenized WITHOUT the answer -- LOCKED
    """
    from datasets import load_dataset

    metamath = load_dataset("meta-math/MetaMathQA")["train"]
    metamath = filter_by_char_limit(metamath, print_fn)

    dev_questions = None
    if group_holdout:
        _, dev_questions = build_group_holdout(group_holdout)
        before = len(metamath)
        keep = [i for i, o in enumerate(metamath["original_question"])
                if o not in dev_questions]
        metamath = metamath.select(keep)
        print_fn(f"group holdout: removed {before - len(metamath):,} rows "
                 f"({(before - len(metamath)) / before:.2%}) derived from "
                 f"{group_holdout} dev problems")

    gsm8k = load_dataset("openai/gsm8k", "main")
    gsm8k = gsm8k.rename_columns({"question": "query", "answer": "response"})

    rng = np.random.RandomState(VALID_SHUFFLE_SEED)
    idx = np.arange(len(gsm8k["train"]))
    rng.shuffle(idx)
    # With a group holdout the dev set IS the held-out set, so it must be the same
    # prefix of the same shuffle that build_group_holdout removed from training.
    # Taking `valid_size` here instead would still be clean (it is a subset), but it
    # would silently be smaller than the split we froze and reported.
    valid_idx = idx[:(group_holdout or valid_size)].tolist()

    ds_train = metamath
    if max_train_examples is not None:
        # Test-only. Never set for a scientific run: it would change the data the model
        # sees while leaving the manifest looking identical, so it is recorded below.
        ds_train = ds_train.select(range(min(max_train_examples, len(ds_train))))
    ds_valid = gsm8k["train"].select(valid_idx)
    ds_test = gsm8k["test"]

    def with_answer(batch):
        texts = [QUERY_TEMPLATE.format(query=q) + a
                 for q, a in zip(batch["query"], batch["response"])]
        tok = tokenizer(texts, truncation=True, max_length=max_seq_length)
        return tok

    def without_answer(batch):
        texts = [QUERY_TEMPLATE.format(query=q) for q in batch["query"]]
        return tokenizer(texts, truncation=True, max_length=max_seq_length)

    ds_train = ds_train.map(with_answer, batched=True,
                            remove_columns=[c for c in ds_train.column_names
                                            if c not in ("query", "response")])
    ds_valid = ds_valid.map(without_answer, batched=True)
    ds_test = ds_test.map(without_answer, batched=True)

    print_fn(f"train {len(ds_train)} | valid {len(ds_valid)} | test {len(ds_test)}")
    return ds_train, ds_valid, ds_test, SplitIds(
        valid_indices=valid_idx, n_train=len(ds_train), n_test=len(ds_test),
        train_truncated_to=max_train_examples, group_holdout=group_holdout)


# --------------------------------------------------------------------------------------
# Collation
# --------------------------------------------------------------------------------------

def collate_metamath(features: list[dict], tokenizer) -> dict:
    """Pad a batch and build labels that KEEP the first EOS as a training target.

    This is not a detail. `DataCollatorForLanguageModeling` masks every padding token to
    -100, and since pad_token == eos_token the model then never sees EOS as a target, never
    learns to stop, and generates until it hits the length cap -- measured at a **100%
    truncation rate** and a GSM8K test score 11 points below PEFT's published result, with
    an otherwise identical training loss.

    PEFT's `method_comparison` handles this explicitly, and we replicate it: mask
    everything strictly after the first pad position, so index `num_tokens` (the first
    EOS) survives as a target. Their own comment: "if we ignore all, the model will not
    learn to predict the EOS token."

    Note the sample with the most tokens in a batch gets no EOS target at all, because
    there is nothing to pad. That is PEFT's behaviour too, and harmless for batch > 1.
    """
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        raise ValueError(
            "This collator relies on pad_token == eos_token so the first pad position IS "
            "an EOS and can be used as a target."
        )
    lengths = [len(f["input_ids"]) for f in features]
    batch = tokenizer.pad(
        {"input_ids": [f["input_ids"] for f in features],
         "attention_mask": [f["attention_mask"] for f in features]},
        return_tensors="pt", padding_side="right")
    labels = batch["input_ids"].clone()
    for i, n in enumerate(lengths):
        labels[i, n + 1:] = -100          # n survives: the first EOS is a target
    batch["labels"] = labels
    return batch


# --------------------------------------------------------------------------------------
# Answer extraction and scoring -- replicated from PEFT's method_comparison
# --------------------------------------------------------------------------------------
#
# Scoring must match the reference implementation or the accuracy is not comparable.
# Taking "the last number in the generation" -- the obvious simple choice -- is wrong in
# two ways: it picks up trailing text when a model rambles, and it cannot represent
# fractional answers such as "20/14" that MetaMathQA emits and GSM8K scores as decimals.

ANSWER_DELIMITERS = (
    "The answer is: ", "The answer is ",            # MetaMathQA
    "The final answer is: ", "The final answer is ",
    "#### ",                                        # GSM8K gold
)


def parse_answer(text: str) -> str | None:
    """Text after the last answer delimiter, or None when no delimiter is present.

    None is a meaningful outcome: a generation that never produced an answer marker is
    scored incorrect rather than being mined for a stray number.
    """
    text = text.strip().rstrip(".!?")
    for delimiter in ANSWER_DELIMITERS:
        if delimiter in text:
            break
    else:
        return None
    tail = text.rpartition(delimiter)[-1].strip()
    tail = tail.split("\n", 1)[0]                    # drop any following paragraph
    # GSM8K omits the percent sign rather than dividing by 100, so stripping it matches.
    return tail.strip(" .!?$%")


def convert_to_decimal(s: str | None) -> Decimal | None:
    """Parse a number or a fraction. Returns None on anything unparseable."""
    if s is None:
        return None
    try:
        s = s.strip().replace(",", "")
        if "/" in s:
            parts = s.split("/")
            if len(parts) != 2:
                return None
            num, den = Decimal(parts[0].strip()), Decimal(parts[1].strip())
            if den == 0:
                return None
            return num / den
        return Decimal(s)
    except (DivisionByZero, InvalidOperation, ValueError):
        return None


def gold_answer(response: str) -> str | None:
    return parse_answer(response)


def is_correct(generation: str, response: str) -> bool:
    """Compare parsed answers, normalising fractions to decimals.

    Float conversion makes "20/35" and "0.5714285714285714" compare equal, which is how
    the reference implementation scores them.
    """
    pred, gold = parse_answer(generation), parse_answer(response)
    if gold is None:
        raise ValueError(f"unparseable gold response: {response[:120]!r}")
    if pred is None:
        return False
    dp, dg = convert_to_decimal(pred), convert_to_decimal(gold)
    if dp is not None and dg is not None:
        return float(dp) == float(dg)
    return pred == gold

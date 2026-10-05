"""Summarize selected test results with equal seed weights."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics

from common import shared_options, validated_metrics, write
from evaluate_runs import selected_rows


def numbers(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            result.update(numbers(item, name + "/"))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = item
    return result


def summarize(records):
    groups = defaultdict(lambda: defaultdict(dict))
    for cell, metrics in records:
        for metric, value in numbers(metrics).items():
            key = cell["model"], cell["method"], cell["level"], cell["capacity"], metric
            subjects = groups[key][cell["seed"]]
            if cell["subject"] in subjects:
                raise ValueError("Duplicate subject within a seed")
            subjects[cell["subject"]] = value
    output = []
    for key, seeds in sorted(groups.items()):
        subject_sets = [set(subjects) for subjects in seeds.values()]
        if len(seeds) != 3 or any(s != subject_sets[0] for s in subject_sets):
            raise ValueError(f"Incomplete three-seed result for {key}")
        # First average subjects within each seed, then compute sample SD over seeds.
        values = [statistics.mean(subjects.values()) for subjects in seeds.values()]
        output.append(dict(model=key[0], method=key[1], level=key[2], capacity=key[3], metric=key[4],
                           mean=statistics.mean(values), sample_sd=statistics.stdev(values),
                           seeds=3, subjects=len(subject_sets[0])))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("math", "coding", "cat", "objects", "hra-extension"))
    shared_options(p); a = p.parse_args(); root = a.output.resolve()
    if a.dry_run:
        print(f"Read selected test results and write {root / a.task / 'summary.csv'}")
        return
    records = []
    for cell in selected_rows(root, a.task):
        value = validated_metrics(root, cell, "test")
        records.append((cell, value["metrics"]))
    rows = summarize(records)
    if not rows:
        raise ValueError("No completed test results")
    path = root / a.task / "summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write(root / a.task / "summary.json", rows)
    print(f"Wrote {len(rows)} summaries to {path}")


if __name__ == "__main__":
    main()

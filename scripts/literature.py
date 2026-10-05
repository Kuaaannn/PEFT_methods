"""Recount the released PEFT annotations and optionally render Figure 2."""
import argparse
import csv
from collections import Counter

from common import CODE, ROOT, execute, python, read, shared_options, write

CRITERIA = ("replication", "lr_selection", "lr_space", "matched", "backbone", "data", "exposure", "modules", "budget")
CATEGORIES = ("documented", "explicitly_not_met", "unclear_not_reported")


def summarize(rows, criterion):
    counts = Counter(r[criterion] for r in rows)
    denominator = sum(counts[c] for c in CATEGORIES)
    if denominator + counts["not_applicable"] != len(rows):
        raise ValueError(f"Unexpected label for {criterion}")
    return dict(n=denominator, **{c: counts[c] for c in CATEGORIES},
                all=sum(r[criterion + "_all"].lower() == "true" for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    shared_options(p)
    p.add_argument("--plot", action="store_true", help="Render the original figure; requires matplotlib and pdflatex")
    a = p.parse_args()
    source = ROOT / "data/literature"
    with (source / "papers.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = {c: summarize(rows, c) for c in CRITERIA}
    if len(rows) != 64:
        raise ValueError("Annotation counts no longer match the paper")
    output = a.output.resolve() / "literature"
    if not a.dry_run:
        write(output / "summary.json", {"eligible_papers": len(rows), "summary": summary})
    print("Verified all nine criteria and their denominators across 64 eligible papers")
    if a.plot:
        execute([python("analysis"), CODE / "analysis/render_literature.py", "--paper", ROOT / "paper",
                 "--source", output / "summary.json", "--output", output],
                env_name="analysis", dry_run=a.dry_run)


if __name__ == "__main__":
    main()

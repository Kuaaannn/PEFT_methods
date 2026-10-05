"""Build a copy of the frozen arXiv manuscript without editing its sources."""
import argparse
import shutil
from common import ROOT, execute, shared_options


def main():
    p = argparse.ArgumentParser(description=__doc__)
    shared_options(p); a = p.parse_args()
    destination = a.output.resolve() / "paper-build"
    if not a.dry_run:
        shutil.copytree(ROOT / "paper", destination, dirs_exist_ok=True)
    execute(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=destination, dry_run=a.dry_run)
    print(destination / "main.pdf")


if __name__ == "__main__":
    main()

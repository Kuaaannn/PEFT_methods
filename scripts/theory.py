"""Run the original synthetic experiments in an isolated output workspace."""
import argparse
from pathlib import Path
import shutil

from common import CODE, ROOT, execute, python, shared_options


def main():
    p = argparse.ArgumentParser(description=__doc__)
    shared_options(p)
    p.add_argument("--plots", action="store_true", help="Also render the original figures using a local TeX installation")
    a = p.parse_args(); work = a.output.resolve() / "theory"
    if not a.dry_run:
        for name in ("analysis", "paper/figures", "paper/generated"):
            (work / name).mkdir(parents=True, exist_ok=True)
        for pth in (CODE / "analysis").rglob("*.py"):
            if pth.name == "render_literature.py":
                continue
            target = work / pth.relative_to(CODE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pth, target)
        shutil.copy2(ROOT / "paper/palette.tex", work / "paper/palette.tex")
    execute([python("analysis"), work / "analysis/svd_perturbation_theory/experiment.py"], cwd=work,
            env_name="analysis", dry_run=a.dry_run)
    # Import from the copied workspace so all original relative output paths resolve there.
    for file in ("validate_index_model", "present_angle_prediction"):
        source = work / "analysis/theory_integration" / f"{file}.py"
        program = ("import runpy; d=runpy.run_path(" + repr(str(source)) + "); "
                   "d['OUT'].mkdir(parents=True,exist_ok=True); d['run']()")
        if a.plots:
            program += "; d['plot']()"
        execute([python("analysis"), "-c", program], cwd=source.parent, env_name="analysis", dry_run=a.dry_run)
    if a.plots:
        execute([python("analysis"), work / "analysis/svd_perturbation_theory/plot_results.py"], cwd=work,
                env_name="analysis", dry_run=a.dry_run)


if __name__ == "__main__":
    main()

"""Create independent Python 3.12 environments for training and evaluation."""
import argparse
from pathlib import Path
import platform
import sys

from common import CODE, CONFIGS, FLUX, HARNESS, LLM, ROOT, execute, python, read


def commands(name, executable):
    env = ROOT / ".venvs" / name
    pip = [python(name), "-m", "pip"]
    result = [[executable, "-m", "venv", str(env)], pip + ["install", "--upgrade", "pip"]]
    if name in ("train", "eval"):
        versions = read(CONFIGS / "language_versions.json")[".venv-" + name]
        result.append(pip + ["install"] + [f"{p}=={v}" for p, v in versions.items()])
        if name == "train":
            result += [pip + ["install", "--no-deps", str(LLM / "third_party/peft-runtime")],
                       pip + ["install", "--no-deps", "-e", str(LLM)],
                       pip + ["install", "--no-deps", "-r", str(HARNESS / "task_harnesses/requirements.txt")]]
    elif name == "flux":
        result += [pip + ["install", "torch==2.8.0", "torchvision==0.23.0"],
                   pip + ["install", "-r", str(FLUX / "requirements-training.txt"),
                          "diffusers==0.40.0", "transformers==5.17.0", "accelerate==1.15.0",
                          "numpy==2.5.2", "safetensors==0.8.0"]]
        # The launchers expose the complete image PEFT snapshot before site-packages.
    else:
        result += [pip + ["install", "numpy", "scipy", "matplotlib", "pandas"]]
    result.append(pip + ["check"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("environment", choices=("train", "eval", "flux", "analysis"))
    p.add_argument("--python", default="python3.12")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.dry_run and a.environment != "analysis" and platform.system() != "Linux":
        p.error("The model experiments require Linux and CUDA. Use --dry-run to inspect the setup on this device.")
    for command in commands(a.environment, a.python):
        execute(command, env_name=a.environment, dry_run=a.dry_run)
    if a.environment in ("train", "flux"):
        execute([python(a.environment), ROOT / "scripts/check_environment.py", a.environment],
                env_name=a.environment, dry_run=a.dry_run)


if __name__ == "__main__":
    main()

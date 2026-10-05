"""Run the native image trainer with artifacts in the requested run directory."""
import argparse
from pathlib import Path

from common import gpu, write


def route_artifacts(native, utilities, output):
    """Redirect file output without changing training or evaluation."""
    output = Path(output)

    def log_to_file(*, log_data, save_dir, experiment_name, timestamp, print_fn):
        destination = output / "training.json"
        write(destination, log_data, immutable=True)
        print_fn(f"Saved log to: {destination}")

    utilities.log_to_file = log_to_file
    utilities.get_result_save_dir = lambda **kwargs: str(output)
    native.get_sample_image_save_dir = lambda **kwargs: str(output / "samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpu()
    import run as native
    import utils
    output = args.output.resolve()
    route_artifacts(native, utils, output)
    native.print_verbose = print
    native.main(path_experiment=args.experiment,
                experiment_name=utils.validate_experiment_path(args.experiment),
                clean=False, checkpoint_dir=str(output / "adapter"))


if __name__ == "__main__":
    main()

"""Download pinned source files only. No parsing/tokenization/models/jobs."""
import os
import urllib.request
from pathlib import Path

from .shared.io import ROOT, atomic_json, exclusive, file_hash, read_json

PISSA_REVISION = "d4746ceca8314940af8a61333bc2d395d9e259c9"
RELEASES = {"HumanEvalPlus": "v0.1.10", "MbppPlus": "v0.2.0"}


def main():
    from huggingface_hub import hf_hub_download
    root = ROOT / "data/sources"
    with exclusive(root):
        manifest = root / "downloads.json"
        if manifest.exists():
            for name, sha in read_json(manifest)["files"].items():
                if file_hash(root / name) != sha:
                    raise ValueError(f"Downloaded source changed: {name}")
            print("Pinned source downloads already present", flush=True)
            return
        hf_hub_download("fxmeng/pissa-dataset", repo_type="dataset", revision=PISSA_REVISION,
                        filename="python/train.json", local_dir=root / "pissa")
        paths = [root / "pissa/python/train.json"]
        for name, version in RELEASES.items():
            path = root / f"{name}-{version}.jsonl.gz"
            if not path.exists():
                url = f"https://github.com/evalplus/{name.lower()}_release/releases/download/{version}/{name}.jsonl.gz"
                temporary = path.with_name(path.name + ".partial")
                print(f"Downloading {name} {version}", flush=True)
                urllib.request.urlretrieve(url, temporary)
                os.replace(temporary, path)
            paths.append(path)
        atomic_json(manifest, {"pissa_revision": PISSA_REVISION, "releases": RELEASES,
                    "files": {str(p.relative_to(root)): file_hash(p) for p in paths}})
        print("Source downloads complete; data splitting/audits deferred to the GPU smoke job", flush=True)


if __name__ == "__main__":
    main()

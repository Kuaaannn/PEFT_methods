"""Verify the selected PEFT implementation before using a GPU."""
import argparse
from pathlib import Path
from common import CODE, LLM, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("environment", choices=("train", "flux"))
    a = p.parse_args()
    import peft
    import peft.tuners.hra.layer as hra
    from peft import OFTConfig
    reference = (CODE / "third_party_snapshots/flux/peft" if a.environment == "flux"
                 else LLM / "third_party/peft-runtime/peft")
    for relative in ("tuners/lora/layer.py", "tuners/oft/layer.py", "tuners/hra/layer.py"):
        if sha(Path(peft.__file__).parent / relative) != sha(reference / relative):
            raise RuntimeError(f"Wrong PEFT implementation for {relative}. Use the supplied launcher.")
    assert hasattr(hra, "_cwy_factors")
    assert {"oft_block_size", "use_cayley_neumann", "num_cayley_neumann_terms"}.issubset(OFTConfig.__dataclass_fields__)
    print(f"Verified {a.environment} PEFT runtime")


if __name__ == "__main__":
    main()

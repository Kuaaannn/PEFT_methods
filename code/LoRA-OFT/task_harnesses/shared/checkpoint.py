from pathlib import Path

from .config import RunConfig, TARGETS
from .io import file_hash, read_json
from .runtime import base_model, bridge, check_peft, tokenizer_for


def inspect_run(run):
    run = Path(run)
    manifest = read_json(run / "complete.json")
    config = RunConfig(**manifest["config"])
    if manifest["status"] != "complete" or manifest["config_hash"] != config.identity:
        raise ValueError("Incomplete or inconsistent training manifest")
    for name, sha in manifest["adapter_files"].items():
        if Path(name).name != name or file_hash(run / "adapter" / name) != sha:
            raise ValueError(f"Adapter changed: {name}")
    adapter = read_json(run / "adapter/adapter_config.json")
    expected = "OFT" if config.method == "oft" else "HRA" if config.method == "hra" else "LORA"
    if adapter["peft_type"] != expected or set(adapter["target_modules"]) != set(TARGETS):
        raise ValueError("Adapter type/targets disagree with run")
    if expected == "LORA":
        multiplier = 2 if config.method in ("pissa", "milora") else 1
        if adapter["r"] != multiplier * config.rank or adapter["lora_alpha"] != multiplier * config.alpha:
            raise ValueError("Export rank/alpha mismatch (training rank is not export rank)")
        if bool(adapter.get("use_dora")) != (config.method == "dora"):
            raise ValueError("DoRA magnitude flag mismatch")
    if file_hash(config.data_manifest) != manifest["bank_hash"]:
        raise ValueError("Training data manifest changed")
    return config, manifest


def adapted_layers(model, manifest):
    methods = bridge()
    layers = {methods.canonical_name(name + ".weight"): module
              for name, module in model.named_modules()
              if hasattr(module, "get_base_layer") and hasattr(module, "merge")}
    if set(layers) != set(manifest["capacity"]["target_matrices"]):
        raise ValueError("Adapted matrix coverage mismatch")
    return dict(sorted(layers.items()))


def load_run(run, *, keep_adapter=False):
    config, manifest = inspect_run(run)
    bridge()
    check_peft()
    from peft import PeftModel
    wrapped = PeftModel.from_pretrained(base_model(config), str(Path(run) / "adapter"),
                                       is_trainable=False).eval()
    adapted_layers(wrapped, manifest)
    tokenizer = tokenizer_for(config)
    if keep_adapter:
        return config, manifest, wrapped, tokenizer
    # All methods use the actual PEFT merge; a LoRA delta shortcut would break DoRA.
    model = wrapped.merge_and_unload(safe_merge=True).eval()
    model.config.use_cache = True
    return config, manifest, model, tokenizer

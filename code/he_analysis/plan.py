"""Login-safe frozen selections and artifact identities. No torch imports."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import struct

WORKSPACE = Path(__file__).resolve().parents[1]
ROOTS = {"llm": WORKSPACE / "LoRA-OFT/experiments",
         "flux": WORKSPACE / "LoRA_OFT_image_generation"}
SELECTIONS = {"llm": ROOTS["llm"] / "checkpoint_protocol/joint_108/selection_manifest.json",
              "flux": ROOTS["flux"] / "checkpoint_protocol/cat_causal22/selection.json"}
PLANS = {key: root / "checkpoint_protocol/hyperspherical/plan.json" for key, root in ROOTS.items()}
OUTPUTS = {"llm": ROOTS["llm"] / "results/hyperspherical/causal22_largest36_v1",
           "flux": ROOTS["flux"] / "runs/hyperspherical/causal22_cat18_v1"}
SETTINGS = {"block_size": 1024, "matmul_dtype": "float32", "repair_below": 1e-5,
            "sample_count": 4096, "audit_rows": 128,
            "weight_representation": "unchanged_standard_merged_bfloat16",
            "aggregation": "equal_matrix_then_mean_and_sample_sd_across_seeds"}
METHODS = {"lora", "oft", "dora", "pissa", "milora", "hra"}
PILOT_SEEDS = {"llm": 13, "flux": 0}  # First saved seed; LLM has no literal seed 0.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def tensor_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        require(0 < length < 32 << 20, f"Invalid safetensors header: {path}")
        raw = stream.read(length)
    header = {k: v for k, v in json.loads(raw).items() if k != "__metadata__"}
    stat = path.stat()
    require(header and max(v["data_offsets"][1] for v in header.values()) + length + 8 == stat.st_size,
            f"Truncated tensor file: {path}")
    return header, {"bytes": stat.st_size, "mtime_ns": str(stat.st_mtime_ns),
                    "header_sha256": hashlib.sha256(raw).hexdigest()}


def base_snapshot(domain, selection, model):
    if domain == "llm":
        return Path(selection["base_snapshots"][model]["path"])
    cache = WORKSPACE / ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-4B"
    revision = (cache / "refs/main").read_text().strip()
    return cache / "snapshots" / revision


def base_metadata(domain, directory):
    weights = directory / "transformer" if domain == "flux" else directory
    files = sorted(weights.glob("*.safetensors"))
    require(files, f"No cached base weights at {weights}")
    shapes, identities = {}, {}
    for path in files:
        header, identity = tensor_header(path)
        require(not (shapes.keys() & header.keys()), "Repeated base weight names")
        shapes.update({name: value["shape"] for name, value in header.items()})
        identities[str(path)] = identity
    return {"path": str(directory), "revision": directory.name, "weights": identities,
            "config_sha256": file_hash(weights / "config.json")}, shapes


def canonical_name(name):
    return name.removeprefix("base_model.model.").replace(".base_layer.", ".")


def entries_from_shapes(shapes):
    entries = []
    for name, shape in sorted(shapes.items()):
        match = re.fullmatch(r"model.layers.(\d+).(self_attn|mlp).(.+)\.weight", name)
        if match:
            depth, block, role = int(match[1]), match[2], match[3]
        else:
            match = re.fullmatch(r"(single_transformer_blocks|transformer_blocks)\.(\d+)\.(.+)\.weight", name)
            require(match is not None, f"Unsupported matrix name: {name}")
            block, depth, role = match[1], int(match[2]), match[3]
        entries.append({"name": name, "shape": shape, "depth": depth,
                        "block_type": block, "projection_type": role})
    return entries


def validate_adapter(directory, method, capacity, shapes):
    config = read(directory / "adapter_config.json")
    header, identity = tensor_header(directory / "adapter_model.safetensors")
    require(not config.get("fan_in_fan_out") and config.get("bias", "none") == "none"
            and not config.get("modules_to_save"), "Unsupported transposed/bias/non-target adaptation")
    expected = {}
    for name, (m, n) in shapes.items():
        stem = "base_model.model." + name.removesuffix(".weight")
        if method in {"lora", "dora", "pissa", "milora"}:
            rank = capacity * (2 if method in {"pissa", "milora"} else 1)
            require(config["peft_type"] == "LORA" and config["r"] == rank
                    and bool(config.get("use_dora")) == (method == "dora"), "Wrong LoRA export/magnitude")
            expected[stem + ".lora_A.weight"] = [rank, n]
            expected[stem + ".lora_B.weight"] = [m, rank]
            if method == "dora":
                expected[stem + ".lora_magnitude_vector"] = [m]
        elif method == "oft":
            require(config["peft_type"] == "OFT" and config["oft_block_size"] == capacity
                    and not config.get("block_share") and not config.get("coft")
                    and n % capacity == 0, "Wrong OFT convention")
            expected[stem + ".oft_R.weight"] = [n // capacity, capacity * (capacity - 1) // 2]
        elif method == "hra":
            require(config["peft_type"] == "HRA" and config["r"] == capacity
                    and config.get("apply_GS") is False, "Wrong HRA convention")
            expected[stem + ".hra_u"] = [n, capacity]
        else:
            raise ValueError(f"Unknown method {method}")
    require({k: v["shape"] for k, v in header.items()} == expected,
            f"Incomplete/unexpected target tensors: {directory}")
    if method in {"pissa", "milora"}:
        spectral = read(directory / "spectral_training.json")
        require(spectral["checkpoint_representation"] == "standard_lora_difference"
                and spectral["method"] == method and spectral["training_rank"] == capacity
                and spectral["export_rank"] == config["r"] == 2 * capacity
                and spectral["layer_count"] == len(shapes)
                and config["lora_alpha"] == spectral["export_alpha"] == 2 * spectral["training_alpha"]
                and config.get("init_lora_weights") is True,
                "PiSSA/MiLoRA must use the standard difference export, never residualize twice")
    return identity


def source_files(domain):
    return sorted(Path(__file__).parent.glob("*.py"))


def build(domain):
    root, selection_path = ROOTS[domain], SELECTIONS[domain]
    selection = read(selection_path)
    require(selection["status"] == "frozen", "Use the frozen causal-22 selection")
    require(not selection.get("selection_uses_test", selection.get("test_used_for_selection", False)),
            "Selection must not use held-out test scores")
    tasks, bases, modules, config_sources = [], {}, {}, {}
    selected = [task for task in selection["tasks"]
                if domain == "flux" or task["budget"] == "large"]
    for index, old in enumerate(selected):
        model = old["model_key"] if domain == "llm" else "flux_cat"
        method = old["method"].replace("flashoft", "oft")
        require(method in METHODS, "Unexpected method")
        if model not in bases:
            bases[model], all_shapes = base_metadata(domain, base_snapshot(domain, selection, model))
            names = (read(Path(old["run"]) / "target_matrices.json") if domain == "llm"
                     else selection["matrix_shapes"])
            shapes = {name: all_shapes[name] for name in names}
            if domain == "flux":
                require(shapes == selection["matrix_shapes"], "Frozen FLUX shapes changed")
            modules[model] = entries_from_shapes(shapes)
        shapes = {e["name"]: e["shape"] for e in modules[model]}
        if domain == "llm":
            target_file = Path(old["run"]) / "target_matrices.json"
            require(set(read(target_file)) == set(shapes), "Inconsistent LLM target scope")
            directory = Path(old["selected_checkpoint"]["checkpoint_resolved"])
            step = old["checkpoint_step"]
            config_sources[str(target_file)] = file_hash(target_file)
        else:
            directory = root / old["checkpoint_path"]
            step = 750
        identity = validate_adapter(directory, method, old["capacity"], shapes)
        if domain == "llm":
            saved = old["selected_checkpoint"]["adapter_weights"]
            require(all(str(identity[k]) == str(saved[k]) for k in identity), "Causal-22 adapter changed")
        files = [directory / "adapter_config.json"]
        if method in {"pissa", "milora"}:
            files.append(directory / "spectral_training.json")
        for file in files:
            current_hash = file_hash(file)
            if domain == "flux":
                expected = old["input_hashes"].get(str(file.relative_to(root)))
            else:
                expected = (old["selected_checkpoint"].get("adapter_config_sha256")
                            if file.name == "adapter_config.json" else None)
                expected = expected or selection["sources"].get(str(file))
            require(expected is None or current_hash == expected, f"Changed frozen input: {file}")
            config_sources[str(file)] = current_hash
        tasks.append({"index": index, "source_index": old.get("array_index", old.get("index")),
                      "model": model, "method": method, "capacity": old["capacity"],
                      "budget": old.get("budget", "largest"), "learning_rate": old["learning_rate"],
                      "seed": old["seed"], "step": step, "adapter": str(directory),
                      "adapter_identity": identity,
                      "expected_weights_sha256": old.get("checkpoint_sha256"),
                      "source_checkpoint_id": old.get("run_id", old.get("checkpoint_id"))})
    require(len(tasks) == (36 if domain == "llm" else 18), "Wrong largest-capacity checkpoint count")
    groups = Counter((t["model"], t["method"], t["capacity"]) for t in tasks)
    require(set(groups.values()) == {3}, "Missing three-seed configuration")
    require(len(groups) == (12 if domain == "llm" else 6), "More than one capacity per model/method")
    for model, method, capacity in groups:
        group = [t for t in tasks if (t["model"], t["method"], t["capacity"]) == (model, method, capacity)]
        require({t["seed"] for t in group} == ({13, 37, 73} if domain == "llm" else {0, 1, 2}),
                "Wrong seed set")
        require(len({t["learning_rate"] for t in group}) == 1, "Seed-dependent LR selection")
    require({k: len(v) for k, v in modules.items()} ==
            ({"qwen": 196, "llama": 224} if domain == "llm" else {"flux_cat": 80}), "Wrong matrix scope")
    pilots = [t["index"] for t in tasks if t["seed"] == PILOT_SEEDS[domain]]
    require(len(pilots) == len(groups), "Missing first-seed pilot")
    plan = {"schema": "he-causal22-v1", "domain": domain, "status": "prepared_not_launched",
            "selection": str(selection_path), "selection_sha256": file_hash(selection_path),
            "output_root": str(OUTPUTS[domain]), "settings": SETTINGS, "bases": bases,
            "modules": modules, "tasks": tasks, "pilot_indices": pilots,
            "capacity_scope": "largest_only", "pilot_seed": PILOT_SEEDS[domain],
            "pilot_seed_rule": "first saved seed: LLM=13, FLUX=0",
            "sources": {str(p): file_hash(p) for p in source_files(domain)},
            "input_sources": config_sources,
            "storage": "summaries + bounded sampled cosines only; no weights/SVD/Gram caches"}
    plan["identity"] = digest(plan)
    return plan


def check(plan, *, index=None):
    payload = {k: v for k, v in plan.items() if k != "identity"}
    require(digest(payload) == plan["identity"], "HE plan identity mismatch")
    require(file_hash(plan["selection"]) == plan["selection_sha256"], "Frozen selection changed")
    for path, expected in {**plan["sources"], **plan["input_sources"]}.items():
        require(file_hash(path) == expected, f"Frozen source/input changed: {path}")
    for model, base in plan["bases"].items():
        for path, identity in base["weights"].items():
            require(tensor_header(path)[1] == identity, f"Base shard changed: {path}")
        config = Path(base["path"]) / ("transformer/config.json" if plan["domain"] == "flux" else "config.json")
        require(file_hash(config) == base["config_sha256"], f"Base config changed: {model}")
    rows = plan["tasks"] if index is None else [plan["tasks"][index]]
    for task in rows:
        path = Path(task["adapter"]) / "adapter_model.safetensors"
        require(tensor_header(path)[1] == task["adapter_identity"], f"Checkpoint changed: {path}")
    return plan

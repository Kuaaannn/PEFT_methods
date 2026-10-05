"""Qwen/Llama adapter for the sibling SPECINT package; no training changes.

Metadata inspection is stdlib-only. Tensor work requires a CUDA GPU. Operators, feasibility and geometry come from SPECINT unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from .paths import reject_quarantined

LIBRARY_HASH = "1c23e0fd5056c835cfb19afc8e71612f"
SHARED = Path(__file__).resolve().parents[3] / "specint"
CORE = ("restore", "base", "trained", "reconstruct", "match_base",
        "match_edit_plus", "match_edit_minus")


def build_merge_check(native_vs_merged_error: float, native_repeat_error: float,
                      merged_repeat_error: float, *, native_finite: bool,
                      merged_finite: bool, tolerance: float = 0.01) -> dict:
    """Record adapter/merge compatibility while gating the evaluated endpoint.

    Standard evaluation uses the merged BF16 representation, so native PEFT
    compatibility is informative but cannot reject that representation. The
    merged model itself must remain finite and repeatable.
    """
    native_vs_merged_finite = math.isfinite(native_vs_merged_error)
    native_repeat_finite = math.isfinite(native_repeat_error)
    merged_repeat_finite = math.isfinite(merged_repeat_error)
    compatibility_passed = (native_finite and merged_finite
                            and native_vs_merged_finite
                            and native_vs_merged_error <= tolerance)
    native_repeat_passed = (native_finite and native_repeat_finite
                            and native_repeat_error <= tolerance)
    merged_repeat_passed = (merged_finite and merged_repeat_finite
                            and merged_repeat_error <= tolerance)
    hard_gate_passed = native_repeat_passed and merged_repeat_passed
    record = {
        "target_representation": "merged_bfloat16",
        "native_vs_merged_relative_error": (native_vs_merged_error
                                             if native_vs_merged_finite else None),
        "native_repeat_relative_error": native_repeat_error if native_repeat_finite else None,
        "untouched_repeat_relative_error": (native_repeat_error
                                             if native_repeat_finite else None),
        "merged_repeat_relative_error": merged_repeat_error if merged_repeat_finite else None,
        "native_logits_finite": native_finite,
        "merged_logits_finite": merged_finite,
        "tolerance": tolerance,
        "native_vs_merged_status": "within_tolerance" if compatibility_passed else "warning",
        "native_vs_merged_gate": "diagnostic_only",
        "native_repeat_passed": native_repeat_passed,
        "merged_endpoint_passed": merged_repeat_passed,
        "hard_gate_passed": hard_gate_passed,
        "hard_gate": "native and merged BF16 finiteness and repeatability",
        "dtype": "bfloat16",
    }
    if not hard_gate_passed:
        raise RuntimeError(f"BF16 repeatability check failed: {record}")
    return record


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def randomness_id(bank_hash: str) -> str:
    """Pair perturbation draws across methods/checkpoints using frozen inputs."""
    return "frozen-bank/" + bank_hash


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_record(value), indent=2, allow_nan=False))
    temporary.replace(path)


def write_endpoint_spectra(directory, rows, metadata):
    """Persist already-computed endpoint spectra; no factorization or model access."""
    import numpy as np
    directory = Path(directory)
    arrays, matrices = {}, []
    for index, row in enumerate(rows):
        base_key, adapted_key = f"base_{index:03d}", f"adapted_{index:03d}"
        arrays[base_key] = np.asarray(row["base"], dtype=np.float32)
        arrays[adapted_key] = np.asarray(row["adapted"], dtype=np.float32)
        matrices.append({"name": row["name"], "shape": row["shape"],
                         "n_singular": len(row["base"]),
                         "base_key": base_key, "adapted_key": adapted_key})
    path = directory / "singular_values.npz"
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)
    atomic_json(directory / "singular_values.json", {
        "schema": 1, **metadata, "file": path.name, "sha256": file_hash(path),
        "matrix_count": len(matrices), "matrices": matrices, "dtype": "float32",
        "order": "descending", "scope": "all_adapted_matrices",
        "source": "existing immutable Factors.s0 and Factors.s_star; no additional SVD",
        "base_weight": "pretrained BF16 weight, losslessly widened for factorization",
        "adapted_weight": "standard PEFT merged BF16 weight, losslessly widened for factorization",
        "stage": "trained checkpoint endpoints, before any causal cell",
    })


def json_record(value):
    """Return strict-JSON data, preserving where undefined numbers occurred.

    Standard evaluation intentionally reports some conditional statistics as NaN when their
    denominator is empty (for example, accuracy among non-truncated generations when every
    generation truncated). That is an undefined optional measurement, not an execution error.
    JSON has no portable NaN value, so persist it as ``null`` and record its exact path. Required
    primary metrics are checked separately by ``run_cell`` and still fail the cell when absent or
    non-finite.
    """
    paths = []

    def visit(node, path):
        if isinstance(node, float) and not math.isfinite(node):
            paths.append(path)
            return None
        if isinstance(node, dict):
            return {key: visit(item, f"{path}.{key}") for key, item in node.items()}
        if isinstance(node, (list, tuple)):
            return [visit(item, f"{path}[{index}]") for index, item in enumerate(node)]
        return node

    result = visit(value, "$")
    if paths and isinstance(result, dict):
        result["undefined_numeric_fields"] = sorted(set(paths))
        result["undefined_numeric_policy"] = (
            "non-finite optional measurements are serialized as JSON null"
        )
    return result


def atomic_jsonl_upsert(path, record_id, rows):
    """Atomically replace one record's rows in a derived JSONL index.

    Cell JSON files are the completion markers. If a worker stops between a
    JSONL update and that marker, retrying the cell must replace, rather than
    duplicate, the already written rows.
    """
    path = Path(path)
    retained = []
    position = None
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("record_id") != record_id:
                retained.append(row)
            elif position is None:
                position = len(retained)
    position = len(retained) if position is None else position
    values = [json_record(row) for row in
              (*retained[:position], *rows, *retained[position:])]
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = "".join(json.dumps(row, allow_nan=False) + "\n" for row in values)
    temporary.write_text(text)
    temporary.replace(path)


def shared_identity() -> dict:
    """Check the pin without importing torch or running any tensor operations."""
    names = ("__init__.py", "rng.py", "ops.py", "geometry.py", "contract.py", "plan.py")
    h = hashlib.blake2b(digest_size=16)
    for name in names:
        h.update(name.encode())
        h.update((SHARED / name).read_bytes())
    if h.hexdigest() != LIBRARY_HASH:
        raise ValueError("SPECINT source differs from the shared 1.3.0 pin")
    contract_path = SHARED / "contract.json"
    override = os.environ.get("SPECINT_CONTRACT")
    if override and file_hash(Path(override)) != file_hash(contract_path):
        raise ValueError("SPECINT_CONTRACT differs from the shared contract")
    return {"library_version": "1.3.0", "library_hash": h.hexdigest(),
            "contract_version": "1.2.0", "contract_hash": file_hash(contract_path)}


def inspect_run(run: str | Path, steps=None) -> list[dict]:
    """Inspect actual saved checkpoints; never infer them from telemetry steps.

    A final adapter fills only a missing final step. It never duplicates step625.
    Missing requested snapshots remain explicit entries in the returned inventory.
    """
    run = reject_quarantined(run).resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["method"] not in {"lora", "rslora", "dora", "pissa", "milora", "hra",
                                   "oft"}:
        raise ValueError(f"{run}: unsupported matrix-only adapter method")
    maximum = int(manifest["max_steps"])
    interval = manifest.get("eval_steps")
    if steps is None and not interval:
        raise ValueError(f"{run}: missing eval_steps; supply explicit steps")
    wanted = sorted(set(steps if steps is not None else
                        [*range(int(interval), maximum, int(interval)), maximum]))
    rows = []
    for step in wanted:
        if step <= 0 or step > maximum:
            raise ValueError(f"invalid checkpoint step {step} for max_steps={maximum}")
        path = run / "ckpt" / f"step{step}"
        if step == maximum and not (path / "adapter_config.json").exists():
            path = run / "adapter"
        config_file = path / "adapter_config.json"
        weights = path / "adapter_model.safetensors"
        status, reason = "ok", None
        if not config_file.is_file() or not weights.is_file():
            status, reason = "failed", "saved adapter config or safetensors is missing"
        else:
            cfg = json.loads(config_file.read_text())
            spectral = manifest["method"] in {"pissa", "milora"}
            expected = ("HRA" if manifest["method"] == "hra" else
                        "OFT" if "oft" in manifest["method"] else "LORA")
            if cfg.get("peft_type") != expected:
                raise ValueError(f"{path}: adapter type disagrees with run manifest")
            if cfg.get("base_model_name_or_path") != manifest["model_id"]:
                raise ValueError(f"{path}: adapter base disagrees with run manifest")
            capacity_field = "oft_block_size" if expected == "OFT" else "r"
            export_capacity = manifest["capacity"] * (2 if spectral else 1)
            if cfg.get(capacity_field) != export_capacity:
                raise ValueError(f"{path}: adapter capacity disagrees with run manifest")
            method = manifest.get("method_kwargs", {})
            checks = ({"use_cayley_neumann": "use_cayley_neumann",
                       "num_cayley_neumann_terms": "num_cayley_neumann_terms",
                       "block_share": "block_share", "coft": "coft"} if expected == "OFT"
                      else {"apply_GS": "apply_GS"} if expected == "HRA"
                      else {} if spectral else {"alpha": "lora_alpha"})
            for source, target in checks.items():
                if source in method and method[source] != cfg.get(target):
                    raise ValueError(f"{path}: {target} disagrees with run manifest")
            if manifest["method"] == "dora" and cfg.get("use_dora") is not True:
                raise ValueError(f"{path}: DoRA export does not enable use_dora")
            if expected == "HRA" and cfg.get("apply_GS") is not False:
                raise ValueError(f"{path}: HRA export differs from the standard non-GS convention")
            if spectral:
                # Training changes the residual base, but standard evaluation loads
                # the portable rank-2r DIFFERENCE onto the untouched pretrained base.
                # Never initialize PiSSA/MiLoRA again or subtract its initial update here.
                info = json.loads((path / "spectral_training.json").read_text())
                if (info.get("method") != manifest["method"]
                        or info.get("base_model_id") != manifest["model_id"]
                        or info.get("checkpoint_representation") != "standard_lora_difference"
                        or info.get("training_rank") != manifest["capacity"]
                        or info.get("export_rank") != export_capacity
                        or info.get("export_alpha") != cfg.get("lora_alpha")
                        or cfg.get("lora_alpha") != 2 * export_capacity
                        or cfg.get("init_lora_weights") is not True
                        or cfg.get("use_dora", False)):
                    raise ValueError(f"{path}: spectral adapter is not a portable standard LoRA difference")
            # This restriction makes restoring all adapted matrices a FULL base reference.
            if (cfg.get("bias", "none") != "none" or cfg.get("lora_bias")
                    or any(cfg.get(k) for k in ("modules_to_save", "trainable_token_indices",
                                               "target_parameters", "layer_replication",
                                               "fan_in_fan_out", "alora_invocation_tokens"))):
                raise ValueError(f"{path}: non-matrix/changed-architecture adapter is unsupported")
        rows.append({"checkpoint_id": f"{manifest['run_id']}/step{step}",
                     "run": str(run), "path": str(path), "step": step,
                     "schedule_fraction": step / maximum, "manifest": manifest,
                     "status": status, "status_reason": reason})
    return rows


def inspect_run_resilient(run: str | Path, steps=None) -> list[dict]:
    """Keep a bad saved snapshot from hiding the good snapshots in its run."""
    run = reject_quarantined(run).resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    maximum = int(manifest["max_steps"])
    interval = manifest.get("eval_steps")
    if steps is None and (not interval or int(interval) <= 0):
        raise ValueError("A positive eval_steps or explicit steps are required")
    wanted = sorted(set(steps if steps is not None else [*range(int(interval), maximum, int(interval)), maximum]))
    rows = []
    for step in wanted:
        try:
            rows.extend(inspect_run(run, [step]))
        except Exception as exc:
            rows.append({"checkpoint_id": f"{manifest.get('run_id', run.name)}/step{step}",
                         "run": str(run), "path": str(run / "ckpt" / f"step{step}"),
                         "step": step, "manifest": manifest, "status": "failed",
                         "status_reason": f"{type(exc).__name__}: {exc}"})
    return rows


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    return torch


def build_edit(f, operator, params, checkpoint_id):
    """Both adapters use the canonical SPECINT dispatch and randomness."""
    from specint.plan import build_edit as shared_build_edit
    return shared_build_edit(f, operator, params, checkpoint_id)


class CheckpointAnalysis:
    """One BF16 model, compact immutable factors and direct installed-weight measurement.

    The evaluator supplies frozen inputs and outcomes, and is called for base and
    trained first. Large factors are streamed directly between disk and CUDA.
    """

    def __init__(self, checkpoint, output, evaluator, base_path=None, evaluation_id="v1",
                 activation_layers=None, activation_inputs=False, factor_cache=None,
                 evaluation_plan=None):
        self.torch = require_gpu()
        self.identity = shared_identity()
        import specint
        if Path(specint.__file__).resolve().parent != SHARED:
            raise RuntimeError("A different SPECINT installation is shadowing the sibling package")
        self.ck, self.evaluator = checkpoint, evaluator
        if checkpoint["status"] != "ok":
            raise ValueError(checkpoint["status_reason"])
        self.root = reject_quarantined(output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_path = base_path
        self.evaluation_id = evaluation_id
        self.evaluation_plan = evaluation_plan
        self.activation_layers = (sorted(set(activation_layers))
                                  if activation_layers is not None else None)
        self.activation_inputs = activation_inputs
        self.factor_cache = Path(factor_cache).resolve() if factor_cache else None
        if self.activation_inputs != bool(self.activation_layers):
            raise ValueError(
                "Frozen-input geometry requires --activation-inputs and a nonempty "
                "--activation-layers list together")
        self.slots = []
        self.reference_slots = []
        self.restoration_success = False

    @staticmethod
    def antithetic_params(operator, params):
        from specint.plan import antithetic_spec
        return antithetic_spec(operator, params)

    def progress(self, phase, **details):
        event = {"phase": phase, **details}
        atomic_json(self.work / "progress.json", event)
        print("PROGRESS " + json.dumps(event), flush=True)

    def prepare(self):
        import inspect
        import importlib.metadata
        from huggingface_hub import hf_hub_download
        from peft import PeftModel, OFTConfig
        from transformers import AutoModelForCausalLM
        from specint import ops
        from .methods import canonical_name
        torch = self.torch
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        adapter = Path(self.ck["path"])
        cfg = json.loads((adapter / "adapter_config.json").read_text())
        if cfg.get("peft_type") == "OFT":
            unknown = set(cfg) - set(OFTConfig.__dataclass_fields__)
            if unknown:
                raise RuntimeError(
                    f"Unsupported OFT checkpoint options: {sorted(unknown)}. "
                    "Use a checkpoint generated with the bundled PEFT runtime."
                )
        base_id = self.base_path or self.ck["manifest"]["model_id"]
        base = Path(base_id)
        if not base.is_dir():
            # Resolve the cached model config only. snapshot_download also demands
            # unrelated README/license files from otherwise usable offline caches.
            base = Path(hf_hub_download(base_id, "config.json", revision=cfg.get("revision"),
                                        local_files_only=True)).parent
        base_files = sorted(base.glob("*.safetensors"))
        if not base_files:
            raise ValueError("A local safetensors base snapshot is required")
        self.checkpoint_hash = digest({
            "adapter": {p.name: file_hash(p) for p in sorted(adapter.iterdir()) if p.is_file()},
            "base": {p.name: file_hash(p) for p in [base / "config.json", *base_files]},
            "run_manifest": self.ck["manifest"], "step": self.ck["step"]})
        source_hash = digest({p.name: file_hash(p) for p in
                              (Path(__file__), Path(__file__).with_name("protocol_metrics.py"),
                               Path(__file__).parent / "data" / "metamath.py",
                               Path(__file__).with_name("evaluate.py"),
                               Path(__file__).with_name("eval_settings.py"),
                               Path(__file__).with_name("protocol_plan.py"),
                               Path(__file__).parents[1] / "eval" / "run_eval.py",
                               Path(__file__).parents[1] / "eval" / "server.py",
                               Path(__file__).parents[1] / "eval" / "checkpoint_server.py")})
        from peft.tuners.lora.layer import Linear as LoraLinear
        from peft.tuners.oft.layer import Linear as OFTLinear
        peft_layers = (LoraLinear, OFTLinear)
        if self.ck["manifest"]["method"] == "hra":
            from peft.tuners.hra.layer import HRALinear
            peft_layers += (HRALinear,)
        self.profile = {"storage": "bfloat16", "merge": "bfloat16",
                        "factor": "float32", "spectrum": "float32", "reduce": "float64",
                        "evaluation_compute": self.evaluator.inference_dtype,
                        "gsm8k_outcome_engine": self.evaluator.outcome_engine,
                        "evaluation_id": self.evaluation_id,
                        "measurement_plan": self.evaluation_plan,
                        "intervention_scope": "all_adapted_matrices",
                        "activation_layers": self.activation_layers,
                        "activation_inputs": self.activation_inputs,
                        "tf32": False, "attention": "sdpa", "solver": "gesvd",
                        "spectrum_solver": "gesvd; one base SVD; analytic planned spectra; measured installed spectra",
                        "factor_format": "split_bf16_endpoints_v2",
                        "weight_installation": "direct_copy_standard_dtype",
                        "randomness": "frozen bank hash, matrix name, sign, draw; paired across methods",
                        "fallback": "none; failures are recorded", "adapter_source": source_hash,
                        "packages": {p: importlib.metadata.version(p) for p in
                                     ("torch", "transformers", "peft", "safetensors")},
                        "hardware": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
                        "peft_layer_hashes": {c.__name__ + str(i): file_hash(Path(inspect.getfile(c)))
                                              for i, c in enumerate(peft_layers)}}
        if self.ck["manifest"]["method"] == "hra":
            self.profile["hra_orientation_diagnostic"] = {
                "primary": "fixed_base_svd_analytically_transported_by_hra",
                "evaluation_and_intervention_weights": "unchanged_merged_bfloat16",
                "source_sha256": file_hash(Path(__file__).with_name("hra_orientation.py"))}
        self.key = digest({**self.identity, "checkpoint_hash": self.checkpoint_hash,
                           "bank_hash": self.evaluator.bank_hash, "profile": self.profile})
        self.work = self.root / self.key
        self.factor_root = ((self.factor_cache / self.key)
                            if self.factor_cache is not None else self.work)
        # Reuse only completed immutable factor files, checked by their content digest.
        self.work.mkdir(exist_ok=True)
        self.factor_root.mkdir(parents=True, exist_ok=True)
        self.scratch = self.factor_root / "references"
        self.scratch.mkdir(exist_ok=True)
        import fcntl
        self.lock = (self.work / ".lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"Loading {self.ck['checkpoint_id']} on CUDA; work={self.work}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(base), dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", local_files_only=True).eval()
        if self.model.config.model_type not in {"llama", "qwen2"}:
            raise ValueError("Only Qwen2/Qwen2.5 and Llama backbones are supported")
        self.evaluator.validate_model(
            self.model, self.ck["manifest"]["model_id"], self.ck["manifest"])
        self.model.config.use_cache = False
        wrapped = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False).eval()
        layers = {canonical_name(n + ".weight"): mod for n, mod in wrapped.named_modules()
                  if hasattr(mod, "get_base_layer") and hasattr(mod, "merge")}
        expected = json.loads((Path(self.ck["run"]) / "target_matrices.json").read_text())
        if set(layers) != set(expected):
            raise ValueError("Adapter module coverage disagrees with target_matrices.json")
        self.entries = []
        for name in sorted(layers):
            w = layers[name].get_base_layer().weight
            match = re.fullmatch(r"model.layers.(\d+).(self_attn|mlp).([a-z_]+).weight", name)
            if w.ndim != 2 or not match:
                raise ValueError(f"Unsupported module/orientation: {name}")
            entry = {"name": name, "shape": list(w.shape),
                                 "orientation": "y = W h", "unit": "whole_matrix",
                                 "fused_parts": None, "layer": int(match[1]),
                                 "role": match[3], "block_type": match[2]}
            # Causal interventions always operate jointly on every adapted
            # target matrix.  Layer selection is reserved for the optional
            # frozen-input diagnostic below and cannot narrow this list.
            self.entries.append(entry)
            path = self.factor_root / f"{digest(name)}.base.pt"
            # Read from the fresh base each time, so an interrupted write cannot
            # supply a stale/corrupt W0 when factors have to be rebuilt.
            torch.save(w.detach(), path)
            if self.ck["manifest"]["method"] == "hra":
                active = layers[name].active_adapters
                if len(active) != 1 or layers[name].hra_apply_GS[active[0]]:
                    raise ValueError("HRA analysis requires one saved non-GS adapter")
                torch.save(layers[name].hra_u[active[0]].detach(),
                           self.factor_root / f"{digest(name)}.hra.pt")
        if not self.entries:
            raise ValueError("The verified adapter has no target matrices")
        self.activation_entries = []
        if self.activation_inputs:
            available = {entry["layer"] for entry in self.entries}
            missing = set(self.activation_layers) - available
            if missing:
                raise ValueError(
                    f"Activation-probe layers are absent from the module manifest: {sorted(missing)}")
            self.activation_entries = [entry for entry in self.entries
                                       if entry["layer"] in self.activation_layers]
        self.module_hash = digest({"intervention_scope": "all_adapted_matrices",
                                   "modules": self.entries})
        inputs = self.evaluator.probe_inputs()
        with torch.inference_mode():
            native = wrapped(**inputs, use_cache=False).logits.float()
            native_repeat = wrapped(**inputs, use_cache=False).logits.float()
            native_finite = bool(torch.isfinite(native).all())
            repeat_error = ops._fro(native_repeat - native) / max(ops._fro(native), 1e-30)
            del native_repeat
            self.model = wrapped.merge_and_unload(safe_merge=True).eval()
            merged = self.model(**inputs, use_cache=False).logits.float()
            merged_repeat = self.model(**inputs, use_cache=False).logits.float()
            merged_finite = (bool(torch.isfinite(merged).all())
                             and bool(torch.isfinite(merged_repeat).all()))
            merge_error = ops._fro(merged - native) / max(ops._fro(native), 1e-30)
            merged_repeat_error = (ops._fro(merged_repeat - merged)
                                   / max(ops._fro(merged), 1e-30))
            del merged_repeat, merged, native, wrapped, layers
        self.merge_check = build_merge_check(
            merge_error, repeat_error, merged_repeat_error,
            native_finite=native_finite, merged_finite=merged_finite)
        if self.merge_check["native_vs_merged_status"] == "warning":
            print("WARNING native PEFT and merged BF16 logits differ beyond the "
                  f"diagnostic tolerance: {self.merge_check}", flush=True)
        params = dict(self.model.named_parameters())
        for entry in self.entries:
            name = entry["name"]
            self.reference_slots.append((name, params[name]))
        self.factor_hashes = {}
        endpoint_spectra = []
        hra_diagnostics = []
        common_band = float("inf")
        for i, entry in enumerate(self.entries, 1):
            name = entry["name"]
            path = self.factor_root / f"{digest(name)}.factors.pt"
            endpoint_path = self.endpoint_path(name)
            base_path = self.factor_root / f"{digest(name)}.base.pt"
            stamp = path.with_suffix(".sha256")
            if path.exists() and endpoint_path.exists() and stamp.exists():
                pair_hash = digest([file_hash(path), file_hash(endpoint_path)])
                if pair_hash != stamp.read_text():
                    raise ValueError(f"Factor cache checksum mismatch: {path}")
                f = self.factors(path)
            else:
                w0 = torch.load(base_path, map_location="cuda:0", weights_only=True)
                f = ops.factor(w0, params[name].detach().clone(), name=name, driver="gesvd")
                # Original BF16 endpoints are losslessly widened during matrix work. Store
                # them separately so reference installation does not read the much larger SVD.
                torch.save({"base": w0, "trained": params[name].detach()}, endpoint_path)
                payload = {k: v for k, v in vars(f).items()
                           if k not in ("W0", "W_star", "_cache")}
                payload["_cache"] = f.scalar_cache()
                torch.save(payload, path)
                stamp.write_text(digest([file_hash(path), file_hash(endpoint_path)]))
                del w0, payload
            endpoint_spectra.append({"name": name, "shape": entry["shape"],
                                     "base": f.s0.detach().tolist(),
                                     "adapted": f.s_star.detach().tolist()})
            if self.ck["manifest"]["method"] == "hra":
                from .hra_orientation import transport_diagnostic
                hra_path = self.factor_root / f"{digest(name)}.hra.pt"
                opt_u = torch.load(hra_path, map_location="cuda:0", weights_only=True)
                hra_diagnostics.append(transport_diagnostic(f, opt_u))
                del opt_u
                hra_path.unlink()
            base_path.unlink(missing_ok=True)
            self.factor_hashes[name] = stamp.read_text()
            self.slots.append((entry, path, params[name]))
            spectrum_norm = float(torch.linalg.vector_norm(f.s_star.double()))
            if spectrum_norm == 0:
                common_band = 0.0
            else:
                for band in ops.BANDS:
                    amplitude, _ = ops.amplitude_band(f, band, 1.0)
                    if amplitude is None:
                        common_band = 0.0
                        continue
                    common_band = min(common_band,
                                      ops.max_paired_scale(f, amplitude) / spectrum_norm)
            del f
            self.progress("factorization", matrix=i, total=len(self.entries), name=name,
                          elapsed_seconds=round(time.perf_counter() - started, 2))
        self.factor_hash = digest(self.factor_hashes)
        write_endpoint_spectra(self.work, endpoint_spectra, {
            "checkpoint_id": self.ck["checkpoint_id"], "checkpoint_hash": self.checkpoint_hash,
            "factor_hash": self.factor_hash, "model_id": self.ck["manifest"]["model_id"],
            "method": self.ck["manifest"]["method"], "step": self.ck["step"],
        })
        del endpoint_spectra
        if hra_diagnostics:
            atomic_json(self.work / "hra_orientation_diagnostics.json", {
                "scope": "all_adapted_matrices", "matrix_count": len(hra_diagnostics),
                "note": "analytic mechanism, not a claim about SVD uniqueness of BF16 endpoints",
                "matrices": hra_diagnostics})
        self.band_common_d_rel_max = common_band if common_band != float("inf") else None
        self.evaluator.bind(self.work, self.torch, scratch=self.scratch)
        self.evaluator_metadata = self.evaluator.metadata
        modules = dict(self.model.named_modules())
        self.activation_modules = {}
        if self.activation_inputs:
            for entry in self.activation_entries:
                module_name = entry["name"].removesuffix(".weight")
                module = modules.get(module_name)
                if module is None or not hasattr(module, "weight"):
                    raise ValueError(f"Cannot hook selected matrix input: {entry['name']}")
                self.activation_modules[entry["name"]] = module
        torch.cuda.synchronize()
        atomic_json(self.work / "manifest.json", {
            **self.identity, "checkpoint": self.ck, "checkpoint_hash": self.checkpoint_hash,
            "module_manifest": self.entries, "module_manifest_hash": self.module_hash,
            "all_adapted_modules": self.entries,
            "intervention_scope": {
                "policy": "all_adapted_matrices",
                "matrix_count": len(self.entries),
                "layers": sorted({entry["layer"] for entry in self.entries}),
            },
            "factor_hashes": self.factor_hashes, "dtype_profile": self.profile,
            "factor_storage": ("node_local_ephemeral" if self.factor_cache is not None
                               else "task_output_transient"),
            "merge_check": self.merge_check, "banks": self.evaluator_metadata,
            "frozen_input_geometry": {"enabled": self.activation_inputs,
                                      "scope": list(self.activation_modules)},
            "spectral_band_largest_common_paired_d_rel": self.band_common_d_rel_max,
            "preparation_cost": {"wall_seconds": time.perf_counter() - started,
                                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                 "hardware": torch.cuda.get_device_name(0)},
            "nonselected_tensor_changes": "none; every adapted target matrix is intervened jointly",
        })

    def factors(self, path):
        from specint.ops import Factors
        payload = self.torch.load(path, map_location="cuda:0", weights_only=True)
        endpoints = self.torch.load(self.endpoint_path(payload["name"]),
                                    map_location="cuda:0", weights_only=True)
        return Factors(W0=endpoints["base"].float(), W_star=endpoints["trained"].float(),
                       **payload)

    def endpoint_path(self, name):
        return self.factor_root / f"{digest(name)}.endpoints.pt"

    def apply(self, operator, params, *, measure_function=True):
        from specint import geometry as G, ops
        torch = self.torch
        aggregate = G.EnergyAggregator()
        rows, errors = [], []
        started = time.perf_counter()
        for index, (entry, path, param) in enumerate(self.slots, 1):
            f = self.factors(path)
            edit = build_edit(f, operator, params, randomness_id(self.evaluator.bank_hash))
            if not edit.feasible:
                errors.append(f"{entry['name']}: {edit.reason}")
                rows.append({**entry, "status": "infeasible", "status_reason": edit.reason})
                del f, edit
                continue
            # Same installation boundary as FLUX. Serialization adds no arithmetic;
            # GPU fixtures verify that this is identical to the old BF16 save/reload path.
            param.data.copy_(edit.W.to(param.dtype))
            realized_storage = param.detach()
            # SVD is unsupported for BF16 and protocol geometry requires FP32 or
            # higher. This is the exact BF16 tensor used for evaluation, widened
            # losslessly to FP32 after installation.
            realized = realized_storage.float()
            geometry, planned = G.edit_geometry(f, edit, realized)
            aggregate.add_geometry(geometry)
            if geometry.s_m is not None:
                aggregate.add("s", geometry.s_m ** 2 * ops._fro2(f.s0), ops._fro2(f.s0))
            else:
                aggregate.add("s", 0.0, 0.0)
            aggregate.add("floor", ((f.rebuild_floor or 0.0) * f.W0_norm) ** 2,
                          f.W0_norm ** 2)
            row = {**entry, **geometry.as_dict(), "planned": planned.as_dict(),
                   "changed_entries": int((realized != f.W_star).sum()),
                   "planned_weight_norm": ops._fro(edit.W), "realized_weight_norm": ops._fro(realized),
                   "storage_cast_relative_error": (ops._fro(realized - edit.W) /
                        max(f.W0_norm, 1e-30)), "storage_dtype": "bfloat16",
                   "solver": f.solver, "rebuild_floor": f.rebuild_floor,
                   "base_rebuild_floor": f.base_rebuild_floor,
                   "degenerate": f.degenerate,
                   "degenerate_replacement_spread": f.degenerate_replacement_spread,
                   "operator_info": edit.info, "status": "ok", "status_reason": None}
            if operator == "spectral_sign" and params == {"t": 1.0, "z": "restore"}:
                row["identical_to_restore"] = bool(torch.equal(
                    realized_storage, f.W_R.to(param.dtype)))
            if (self.activation_inputs and measure_function
                    and entry["name"] in self.activation_modules):
                row["frozen_input_geometry"] = self.evaluator.function_geometry(
                    entry["name"], f.W0, f.W_star, realized)
            opposite = self.antithetic_params(operator, params)
            if opposite is not None:
                other = build_edit(f, opposite[0], opposite[1],
                                   randomness_id(self.evaluator.bank_hash))
                if other.feasible:
                    # Match the real storage cast before checking E+ + E- = 0.
                    realized_other = other.W.to(param.dtype).float()
                    numerator = ops._fro((realized - f.W_star) +
                                         (realized_other - f.W_star))
                    denominator = max(geometry.edit_norm, ops._fro(realized_other - f.W_star),
                                      1e-30)
                    row["antithetic_postcast_relative_error"] = numerator / denominator
                    row["antithetic_operator_params"] = opposite[1]
                    del realized_other
                else:
                    row["antithetic_postcast_relative_error"] = None
                    row["antithetic_status_reason"] = other.reason
                del other
            if geometry.b_m is None or geometry.er_norm_fraction is None:
                row["ratio_reason"] = "zero base or update norm; affected ratios are undefined"
            if operator == "restore":
                # Weyl's bound makes the measured BF16 weight-cast error a valid
                # upper scale for its induced singular-value error.
                verified, tolerance = G.restoration_valid(
                    f, realized, geometry, row["storage_cast_relative_error"])
                row.update(restoration_verified=verified, restoration_spectrum_tolerance=tolerance)
                if not verified:
                    row.update(status="failed", status_reason="restored spectrum failed its numerical invariant")
            if operator.startswith("match_"):
                row["matching"] = {}
                for axis, target, attribute in (
                        ("base", f.restoration_base_distance, "base_distance"),
                        ("trained", f.restoration_edit_norm, "edit_norm")):
                    row["matching"][axis] = {"target_distance": target,
                        "planned_distance": getattr(planned, attribute),
                        "realized_distance": getattr(geometry, attribute),
                        "planned_relative_error": G.matching_error(getattr(planned, attribute), target),
                        "realized_relative_error": G.matching_error(getattr(geometry, attribute), target)}
                base_control = operator == "match_base"
                target = f.restoration_base_distance if base_control else f.restoration_edit_norm
                attribute = "base_distance" if base_control else "edit_norm"
                for kind, measurement in (("planned", planned), ("realized", geometry)):
                    status, error = G.check_match(getattr(measurement, attribute), target)
                    row[f"{kind}_match_error"] = error
                    row["match_axis"] = "base" if base_control else "trained"
                    if status != "ok":
                        row.update(status=status, status_reason="undefined or >1% norm match error")
            if "requested_edit_norm" in edit.info:
                requested = edit.info["requested_edit_norm"]
                error = G.matching_error(geometry.edit_norm, requested)
                row["requested_edit_norm"] = requested
                row["realized_edit_norm_match_error"] = error
                if error is None or error > 0.01:
                    row.update(status="unmatched",
                               status_reason="realized spectral edit missed its requested norm by >1%")
            if operator == "spectral_band":
                row["largest_common_paired_d_rel"] = self.band_common_d_rel_max
            if operator == "global_gain":
                requested = edit.params["target_edit_norm"]
                error = G.matching_error(geometry.edit_norm, requested)
                row["shape_matched_target_edit_norm"] = requested
                row["shape_match_relative_error"] = error
                if error is None or error > 0.01:
                    row.update(status="unmatched",
                               status_reason="global gain missed the shape edit norm by >1%")
            if operator == "spectral_shape":
                trained_norm = ops._fro(f.W_star)
                norm_error = G.matching_error(ops._fro(realized), trained_norm)
                row["spectral_norm_preservation_relative_error"] = norm_error
                if norm_error is None or norm_error > 0.01:
                    row.update(status="unmatched",
                               status_reason="shape edit failed to preserve spectral/Frobenius norm")
            if operator not in ("base", "trained", "reconstruct") and G.below_floor(
                    planned.edit_norm, (f.rebuild_floor or 0) * f.W0_norm):
                if row["status"] not in ("failed", "infeasible"):
                    row.update(status="unresolved", status_reason="edit at/below reconstruction floor")
            rows.append(row)
            del f, edit, realized, realized_storage
            if index % 8 == 0 or index == len(self.slots):
                self.progress("construction_geometry", operator=operator, matrix=index,
                              total=len(self.slots),
                              elapsed_seconds=round(time.perf_counter() - started, 2))
        summary = aggregate.as_dict()
        if operator == "spectral_sign" and params == {"t": 1.0, "z": "restore"}:
            summary["identical_to_restore"] = (
                len(rows) == len(self.entries)
                and all(row.get("identical_to_restore", False) for row in rows))
        summary["edit_norm"] = sum(r.get("edit_norm", 0.0) ** 2 for r in rows) ** 0.5
        summary["base_distance"] = sum(r.get("base_distance", 0.0) ** 2 for r in rows) ** 0.5
        paired = [r.get("antithetic_postcast_relative_error") for r in rows
                  if r.get("antithetic_postcast_relative_error") is not None]
        if paired:
            summary["antithetic_postcast_relative_error_max"] = max(paired)
        if self.activation_inputs and measure_function:
            function_network = {}
            for comparison in ("base_to_trained", "trained_to_edit"):
                function_network[comparison] = {}
                for role in ("target_eval", "general_eval"):
                    values = [row["frozen_input_geometry"][comparison][role] for row in rows
                              if "frozen_input_geometry" in row]
                    edit_energy = sum(v["sum_output_edit_energy"] for v in values)
                    reference_energy = sum(v["reference_output_energy"] for v in values)
                    function_network[comparison][role] = {
                        "sum_output_edit_energy": edit_energy,
                        "reference_output_energy": reference_energy,
                        "relative_output_edit_energy": (edit_energy / reference_energy
                                                        if reference_energy else None),
                        "n_matrix_inputs": sum(v["n_inputs"] for v in values),
                        "status": "ok" if values and reference_energy else "infeasible",
                        "status_reason": None if values and reference_energy else
                            "no captured matrix inputs or zero reference output energy"}
            summary["frozen_input_geometry"] = function_network
        if errors:
            # Never evaluate a partially built network.
            self.restore_trained()
            return "infeasible", "; ".join(errors), rows, summary
        for status in ("failed", "infeasible", "unmatched", "unresolved"):
            if any(row["status"] == status for row in rows):
                return status, "see per-matrix status reasons", rows, summary
        return "ok", None, rows, summary

    def copy_reference(self, reference):
        if reference not in ("base", "trained"):
            raise ValueError(reference)
        for name, param in self.reference_slots:
            endpoints = self.torch.load(self.endpoint_path(name),
                                        map_location=param.device, weights_only=True)
            param.data.copy_(endpoints[reference])
            del endpoints

    def restore_trained(self):
        self.copy_reference("trained")

    def prepare_references(self):
        """Prerequisite predictions only; these do not create variant records."""
        with self.torch.inference_mode():
            for reference in ("base", "trained"):
                self.progress("reference_evaluation", reference=reference)
                self.copy_reference(reference)
                all_names = [name for name, _ in self.reference_slots]
                selected_names = [entry["name"] for entry, _, _ in self.slots]
                weight_names = self.evaluator.vllm_weight_scope(
                    all_names, selected_names, reference)
                if weight_names:
                    self.evaluator.sync_vllm_weights(
                        self.model, weight_names,
                        unselected_trained=reference == "trained")
                self.evaluator.cache_reference(self.model, reference)
                if reference == "trained" and self.evaluator.verify_standard_parity:
                    self.progress("standard_loading_parity")
                    self.evaluator.verify_standard_loading(self.model, json.loads(
                        (self.work / "reference-trained.json").read_text()))
                if self.activation_inputs:
                    print(f"Capturing {reference} common module inputs", flush=True)
                    self.evaluator.capture_module_inputs(
                        self.model, self.activation_modules, reference)
            print("Measuring repeated trained-reference loss floor", flush=True)
            self.evaluator.measure_trained_floor(self.model)

    def verify_restoration(self, built=None):
        """Validate the first measured intervention using its existing geometry.

        Mathematical validity and numerical resolvability are separate: an OFT
        null can be unresolved while its restoration is correctly constructed.
        """
        with self.torch.inference_mode():
            # The first measured restore passes its already constructed geometry here.
            # Standalone verification remains available to GPU fixtures.
            status, reason, rows, geometry = (built if built is not None else self.apply(
                "restore", {}, measure_function=False))
            passed = (status not in ("failed", "infeasible") and len(rows) == len(self.entries)
                      and all(r.get("restoration_verified", False) for r in rows))
            result = {"passed": passed, "coverage": len(rows), "expected": len(self.entries),
                      "status": status, "status_reason": reason, "matrix_rows": rows, "geometry": geometry}
            atomic_json(self.work / "restoration_preflight.json", result)
            if not passed:
                raise RuntimeError("Restoration preflight failed; see restoration_preflight.json")
            self.restoration_success = True
            print(f"RESTORATION VERIFIED {len(rows)}/{len(self.entries)} matrices", flush=True)
            return result



    def run_cell(self, operator, params):
        from specint import contract, rng
        torch = self.torch
        rid = digest({"analysis": self.key, "module_manifest_hash": self.module_hash,
                      "factors": self.factor_hash, "operator": operator, "params": params})
        record_path = self.work / f"{rid}.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else None
        if record is not None and record.get("execution_status") == "success":
            if operator == "restore":
                self.restoration_success = record.get("restoration_verified", False) and record.get("execution_status") == "success"
                self._restore_record = record
            return record
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        intervention_seconds = 0.0
        evaluation_seconds = 0.0
        transfer_seconds = 0.0
        reused_from = None
        metrics, examples, rows, geometry = None, [], [], {}
        self.progress("cell_start", operator=operator, params=params)
        self.evaluator.last_timings = {}
        try:
            contract.require_operator(operator)
            if operator != "restore" and not self.restoration_success:
                raise RuntimeError("Restoration must succeed before any other measured variant")
            with torch.inference_mode():
                intervention_started = time.perf_counter()
                # All cells are absolute functions of immutable endpoints and overwrite
                # every adapted tensor. No reset is needed between successful cells.
                status, reason, rows, geometry = self.apply(operator, params)
                if operator == "restore":
                    self.verify_restoration((status, reason, rows, geometry))
                intervention_seconds = time.perf_counter() - intervention_started
                if status not in ("infeasible", "failed"):
                    evaluation_started = time.perf_counter()
                    restored = getattr(self, "_restore_record", None)
                    if (geometry.get("identical_to_restore") and restored
                            and restored["execution_status"] == "success"):
                        # Reuse only after every installed BF16 target tensor is proven equal.
                        reused_from = restored["record_id"]
                        metrics = restored["metrics"]
                        examples = [{k: v for k, v in row.items() if k != "record_id"}
                                    for line in (self.work / "per_prompt_metrics.jsonl").read_text().splitlines()
                                    if (row := json.loads(line)).get("record_id") == reused_from]
                    else:
                        # Base/trained outcomes already exist in the reference cache;
                        # their diagnostic KL pass uses only the Transformers model.
                        if operator not in ("base", "trained"):
                            self.progress("weight_transfer", operator=operator)
                            transfer_started = time.perf_counter()
                            names = [entry["name"] for entry, _, _ in self.slots]
                            weight_names = self.evaluator.vllm_weight_scope(names, names, operator)
                            if weight_names:
                                self.evaluator.sync_vllm_weights(
                                    self.model, weight_names, unselected_trained=True)
                            transfer_seconds = time.perf_counter() - transfer_started
                        self.progress("evaluation", operator=operator)
                        metrics, examples = self.evaluator(self.model, operator)
                    evaluation_seconds = time.perf_counter() - evaluation_started
                    # The two standard-evaluation outcomes are mandatory for
                    # every causal cell.  The small-context NLLs remain required
                    # only because their paired KL diagnostics share the same
                    # frozen rows; they are not substitutes for retention_nll.
                    required_metrics = (
                        "task_accuracy", "retention_nll",
                        "answer_token_nll", "general_text_nll",
                    )
                    invalid_required = [
                        key for key in required_metrics
                        if metrics.get(key) is None
                        or not isinstance(metrics.get(key), (int, float))
                        or not math.isfinite(float(metrics[key]))
                    ]
                    if invalid_required:
                        status = "failed"
                        reason = ("required primary metrics are absent or non-finite: "
                                  + ", ".join(invalid_required))
                    if metrics.get("measurement_failures"):
                        status, reason = "failed", "some outcomes failed; independent measurements are retained"
        except Exception as exc:
            status, reason = "failed", f"{type(exc).__name__}: {exc}"
        torch.cuda.synchronize()
        random_keys = ({entry["name"]: rng.rng_key(randomness_id(self.evaluator.bank_hash), entry["name"],
                                                "sign", params.get("draw", 0))
                        for entry in self.entries}
                       if operator in ("spectral_sign", "relative_sign", "spectral_shape",
                                       "global_gain", "spectral_band")
                       and params.get("z") in ("random", "neg_random") else None)
        record = {**self.identity, "record_id": rid, "checkpoint_id": self.ck["checkpoint_id"],
                  "checkpoint_hash": self.checkpoint_hash, "module_manifest_hash": self.module_hash,
                  "factor_hash": self.factor_hash, "operator_id": operator, "operator_params": params,
                  "rng_key": random_keys,
                  "dtype_profile": self.profile, "bank_hash": self.evaluator.bank_hash,
                  "metrics": metrics, "geometry": geometry, "status": status, "status_reason": reason,
                  # Infeasible/unmatched/unresolved are protocol observations, not
                  # execution failures. They remain explicit and do not fail sibling cells.
                  "execution_status": "failed" if status == "failed" else "success",
                  "restoration_verified": (len(rows) == len(self.entries) and
                       all(r.get("restoration_verified", False) for r in rows)) if operator == "restore" else None,
                  "wall_seconds": time.perf_counter() - started,
                  "intervention_wall_seconds": intervention_seconds,
                  "evaluation_wall_seconds": evaluation_seconds,
                  "weight_transfer_wall_seconds": transfer_seconds,
                  "evaluation_reused_from": reused_from,
                  "evaluation_stage_seconds": ({} if reused_from else
                      dict(getattr(self.evaluator, "last_timings", {}))),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "hardware": torch.cuda.get_device_name(0)}
        for filename, values in (("matrix_geometry.jsonl", rows),
                                 ("per_prompt_metrics.jsonl", examples)):
            atomic_jsonl_upsert(self.work / filename, rid,
                                [{"record_id": rid, **row} for row in values])
        atomic_jsonl_upsert(self.work / "aggregate.jsonl", rid, [record])
        atomic_json(record_path, record)
        if operator == "restore":
            self.restoration_success = record["restoration_verified"] and record["execution_status"] == "success"
            self._restore_record = record
        self.progress("cell_complete", operator=operator, status=record["execution_status"],
                      construction_seconds=round(intervention_seconds, 2),
                      evaluation_seconds=round(evaluation_seconds, 2))
        return record

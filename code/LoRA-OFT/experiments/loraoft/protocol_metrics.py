"""Frozen, pretokenized LLM banks for PROTOCOL.md section 4.

Full-vocabulary reference-to-variant KL and NLL share exactly the same shifted
mask. Outcomes reuse GSM8K's existing scorer or a declared choice-scoring rule.
No tokenization, downloads or benchmark selection occur during measurement.
"""
from __future__ import annotations

import json
import shutil
import socket
import struct
import tempfile
import time
from pathlib import Path

from .checkpoint_analysis import atomic_json, digest, file_hash, json_record
from .eval_settings import GSM8K_DEV_N, GSM8K_MAX_LENGTH, GSM8K_MAX_NEW_TOKENS


def validate_tokens(row, general=False):
    ids, attention, score = (row[k] for k in ("input_ids", "attention_mask", "score_mask"))
    if not ids or len(ids) != len(attention) or len(ids) != len(score):
        raise ValueError("Token IDs, attention_mask and score_mask must have equal nonzero lengths")
    if any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("input_ids must contain nonnegative integers")
    if any(x not in (0, 1) for x in [*attention, *score]):
        raise ValueError("Masks must contain 0/1 values")
    if general and score != attention:
        raise ValueError("Raw general-text score_mask must equal attention_mask")


class FrozenTokenEvaluator:
    def __init__(self, path, chunk_size=128, generation_batch_size=8,
                 inference_dtype="bfloat16", vllm_socket=None, vllm_scratch=None,
                 verify_standard_parity=False):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if generation_batch_size < 1:
            raise ValueError("generation_batch_size must be positive")
        if inference_dtype not in ("float32", "bfloat16"):
            raise ValueError("inference_dtype must be float32 or bfloat16")
        self.chunk_size = chunk_size
        self.generation_batch_size = generation_batch_size
        self.inference_dtype = inference_dtype
        if bool(vllm_socket) != bool(vllm_scratch):
            raise ValueError("vLLM outcome evaluation requires both socket and scratch paths")
        self.vllm_socket = Path(vllm_socket) if vllm_socket else None
        self.vllm_scratch = Path(vllm_scratch) if vllm_scratch else None
        self.outcome_engine = "vllm" if self.vllm_socket else "transformers_sdpa"
        self._vllm_unselected_trained = False
        self.verify_standard_parity = verify_standard_parity
        self.standard_gsm8k = None
        self.bank = json.loads(Path(path).read_text())
        for key in ("model_id", "tokenizer_revision", "eos_policy", "mask_policy"):
            if not self.bank.get(key):
                raise ValueError(f"Frozen bank requires {key}")
        revision = self.bank["tokenizer_revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("tokenizer_revision must be an immutable 40-character commit SHA")
        if self.bank["mask_policy"] != "label_indexed_answer_tokens/general_all_valid":
            raise ValueError("Unsupported frozen-bank mask policy")
        standard_retention = self.bank.get("standard_retention")
        if not isinstance(standard_retention, dict):
            raise ValueError("Frozen bank lacks the standard retention-NLL contract")
        required_retention = {
            "source", "source_file_sha256", "content_sha256", "n_rows", "max_length",
            "batch_size", "aggregation", "base_retention_nll", "kl_n_rows",
            "kl_max_length", "kl_vocabulary", "kl_aggregation",
        }
        if not required_retention <= set(standard_retention):
            raise ValueError("Frozen bank has an incomplete standard retention-NLL contract")
        if (standard_retention["n_rows"] != 200
                or standard_retention["max_length"] != 768
                or standard_retention["batch_size"] != 2
                or standard_retention["aggregation"] != "token_pooled_mean_nll"
                or standard_retention["kl_n_rows"] != 200
                or standard_retention["kl_max_length"] != 768
                or standard_retention["kl_vocabulary"] != "full"
                or standard_retention["kl_aggregation"]
                != "example_mean_and_token_pooled"):
            raise ValueError("Standard retention settings differ from the training protocol")
        if (not isinstance(standard_retention["base_retention_nll"], (int, float))
                or not standard_retention["base_retention_nll"] > 0):
            raise ValueError("Standard base retention NLL must be a positive number")
        retention_path = Path(__file__).resolve().parents[1] / "results/retention_bank.json"
        if file_hash(retention_path) != standard_retention["source_file_sha256"]:
            raise ValueError("Standard retention source file checksum mismatch")
        retention = json.loads(retention_path.read_text())
        if (retention.get("sha256") != standard_retention["content_sha256"]
                or retention.get("n") != standard_retention["n_rows"]
                or len(retention.get("rows", [])) != standard_retention["n_rows"]):
            raise ValueError("Standard retention bank contents disagree with the frozen contract")
        self.standard_retention = standard_retention
        self.standard_retention_texts = retention["rows"]
        self.loss_rows = None
        seen = set()
        for role in ("selection", "target_eval", "general_eval"):
            bank = self.bank[role]
            if not bank["rows"] or not bank.get("source"):
                raise ValueError(f"{role} needs nonempty rows and source provenance")
            for row in bank["rows"]:
                validate_tokens(row, general=role == "general_eval")
                if row["id"] in seen:
                    raise ValueError("Example IDs must be unique across selection/target/general banks")
                seen.add(row["id"])
            if role != "selection":
                if not bank.get("outcomes"):
                    raise ValueError(f"{role} needs frozen outcome examples as well as NLL/KL contexts")
                for row in bank["outcomes"]:
                    if row["type"] == "choice":
                        if row["scoring"] not in ("sum_logprob", "mean_logprob"):
                            raise ValueError("Declare choice scoring as sum_logprob or mean_logprob")
                        if not 0 <= row["gold_index"] < len(row["choices"]):
                            raise ValueError("Choice gold_index is out of range")
                        for choice in row["choices"]:
                            validate_tokens(choice)
                            if not any(choice["attention_mask"][i-1] and choice["attention_mask"][i]
                                       and choice["score_mask"][i] for i in range(1, len(choice["input_ids"]))):
                                raise ValueError("A choice has no scored continuation tokens")
                    elif row["type"] == "gsm8k":
                        if not row["prompt_ids"] or row["max_new_tokens"] < 1:
                            raise ValueError("GSM8K outcome needs prompt_ids and a positive generation cap")
                    else:
                        raise ValueError(f"Unknown outcome type {row['type']!r}")
        if (self.bank.get("target_kl_context_n") != len(self.bank["target_eval"]["rows"])
                or any(len(self.bank[role]["rows"])
                       != self.bank.get("activation_context_n_per_role")
                       for role in ("target_eval", "general_eval"))):
            raise ValueError("Frozen target-KL/activation diagnostic panel changed")
        self.bank_hash = digest({"bank": self.bank, "logprob_chunk_size": chunk_size,
                                 "generation_batch_size": generation_batch_size,
                                 "inference_dtype": inference_dtype,
                                 "outcome_engine": self.outcome_engine})
        self.metadata = {"bank_hash": self.bank_hash, "source_file_hash": file_hash(Path(path)),
                         "model_id": self.bank["model_id"], "tokenizer_revision": self.bank["tokenizer_revision"],
                         "eos_policy": self.bank["eos_policy"], "mask_policy": self.bank["mask_policy"],
                         "generation_batch_size": generation_batch_size,
                         "inference_compute_dtype": inference_dtype,
                         "gsm8k_outcome_engine": self.outcome_engine,
                         "standard_retention": self.standard_retention,
                         "role_hashes": {r: digest(self.bank[r]) for r in
                                         ("selection", "target_eval", "general_eval")}}

    def standard_retention_metrics(self, model):
        """Run the exact standard 200-document retention-NLL implementation."""
        from .evaluate import token_nll

        value = token_nll(
            model,
            self.tokenizer,
            self.standard_retention_texts,
            max_length=self.standard_retention["max_length"],
            batch_size=self.standard_retention["batch_size"],
        )
        return {
            "retention_nll": value,
            "retention_bank_sha": self.standard_retention["content_sha256"][:12],
            "retention_n": self.standard_retention["n_rows"],
            "retention_max_length": self.standard_retention["max_length"],
            "retention_batch_size": self.standard_retention["batch_size"],
            "retention_aggregation": self.standard_retention["aggregation"],
        }

    def validate_model(self, model, model_id, run_manifest=None):
        import torch
        from transformers import AutoTokenizer
        if self.bank["model_id"] != model_id:
            raise ValueError("Frozen token bank is for a different backbone")
        self.torch = torch
        if self.inference_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16 inference")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=self.bank["tokenizer_revision"], local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if len(self.tokenizer) > model.config.vocab_size:
            raise ValueError("Tokenizer vocabulary exceeds model vocabulary")
        general_rows = []
        for index, text in enumerate(self.standard_retention_texts):
            encoded = self.tokenizer(
                text, add_special_tokens=True, truncation=True,
                max_length=self.standard_retention["kl_max_length"])
            row = {
                "id": f"standard-retention/{index}",
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "score_mask": list(encoded["attention_mask"]),
            }
            validate_tokens(row, general=True)
            general_rows.append(row)
        if len(general_rows) != self.standard_retention["kl_n_rows"]:
            raise ValueError("Standard KL panel has the wrong number of documents")
        self.loss_rows = {
            "target_eval": self.bank["target_eval"]["rows"],
            "general_eval": general_rows,
        }
        self.metadata["standard_kl"] = {
            "source": self.standard_retention["source"],
            "content_sha256": self.standard_retention["content_sha256"],
            "n_rows": len(general_rows),
            "max_length": self.standard_retention["kl_max_length"],
            "vocabulary": self.standard_retention["kl_vocabulary"],
            "aggregation": self.standard_retention["kl_aggregation"],
            "tokenized_rows_hash": digest(general_rows),
            "reference_cache": "final_hidden_states_native_dtype",
        }
        for role in ("selection", "target_eval", "general_eval"):
            for row in self.bank[role]["rows"]:
                if max(row["input_ids"]) >= model.config.vocab_size:
                    raise ValueError("Frozen input contains a token outside the model vocabulary")
        if any(max(row["input_ids"]) >= model.config.vocab_size
               for row in general_rows):
            raise ValueError("Standard KL input contains a token outside the model vocabulary")
        if self.vllm_socket is not None:
            if run_manifest is None:
                raise ValueError("Standard vLLM evaluation requires the checkpoint run manifest")
            self._bind_standard_gsm8k(run_manifest)

    def _bind_standard_gsm8k(self, run_manifest):
        """Load the ordinary post-hoc dev evaluation and prove the bank identifies it exactly."""
        from eval.run_eval import dev_prompts

        prompts, golds = dev_prompts(
            run_manifest["model_id"],
            run_manifest.get("group_holdout"),
            GSM8K_DEV_N,
            run_manifest.get("max_seq_length", 768),
        )
        rows = [row for row in self.bank["target_eval"]["outcomes"]
                if row["type"] == "gsm8k"]
        if len(rows) != len(prompts) or len(golds) != len(prompts):
            raise ValueError("Checkpoint bank and standard GSM8K evaluation have different sizes")
        for row, prompt, gold in zip(rows, prompts, golds):
            prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            budget = max(1, min(GSM8K_MAX_NEW_TOKENS,
                                GSM8K_MAX_LENGTH - len(prompt_ids)))
            if row["prompt_ids"] != prompt_ids or row["gold"] != gold or row["max_new_tokens"] != budget:
                raise ValueError("Checkpoint bank differs from the standard GSM8K evaluation")
        self.standard_gsm8k = {"prompts": prompts, "golds": golds}

    def bind(self, work, torch, *, scratch=None):
        self.work, self.torch = work, torch
        self.scratch = work if scratch is None else scratch
        self.scratch.mkdir(parents=True, exist_ok=True)
        # A requeued worker has new node-local scratch. Invalidate only indexes
        # with missing payloads; present-but-corrupt payloads still fail checksum checks.
        for reference in ("base", "trained"):
            for prefix in ("reference", "activation-inputs"):
                marker = work / f"{prefix}-{reference}.json"
                if marker.is_file():
                    saved = json.loads(marker.read_text())
                    files = (list(saved["files"]) if prefix == "reference"
                             else [row["file"] for row in saved["rows"]])
                    if any(not (self.scratch / name).is_file() for name in files):
                        marker.unlink()
        self._activation_metadata = {}
        self._function_geometry_cache = {}

    def inference_context(self):
        from contextlib import nullcontext
        if self.inference_dtype == "bfloat16":
            return self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
        return nullcontext()

    def inputs(self, row):
        return {k: self.torch.tensor([row[k]], device="cuda:0", dtype=self.torch.long)
                for k in ("input_ids", "attention_mask")}

    def probe_inputs(self):
        # A real frozen context; no generated probe or training data is introduced.
        row = self.bank["target_eval"]["rows"][0]
        return self.inputs({k: row[k][:64] for k in ("input_ids", "attention_mask")})

    def _vllm_request(self, request):
        if self.vllm_socket is None:
            raise RuntimeError("vLLM outcome evaluation is not configured")
        encoded = json.dumps(request, allow_nan=False).encode()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(self.vllm_socket))
            connection.sendall(struct.pack("!Q", len(encoded)) + encoded)

            def receive_exact(size):
                blocks = []
                while size:
                    block = connection.recv(size)
                    if not block:
                        raise ConnectionError("vLLM worker disconnected during a response")
                    blocks.append(block)
                    size -= len(block)
                return b"".join(blocks)

            size = struct.unpack("!Q", receive_exact(8))[0]
            if size > 512 << 20:
                raise ValueError("vLLM response exceeds 512 MiB")
            response = json.loads(receive_exact(size))
        if response.get("status") != "ok":
            raise RuntimeError(f"vLLM checkpoint worker failed: {response.get('error')}")
        return response

    def vllm_weight_scope(self, all_names, selected_names, operator):
        """Choose the smallest patch that still gives an exact isolated variant."""
        if self.vllm_socket is None:
            return []
        if operator == "base" or not self._vllm_unselected_trained:
            return list(all_names)
        return list(selected_names)

    def sync_vllm_weights(self, model, names, *, unselected_trained):
        """Stream current BF16 tensors through job-local memory, then delete them."""
        if self.vllm_socket is None:
            return None
        if not names:
            raise ValueError("A vLLM weight update cannot be empty")
        patch = self.vllm_scratch / "weight-patch"
        shutil.rmtree(patch, ignore_errors=True)
        patch.mkdir(parents=True)
        params = dict(model.named_parameters())
        manifest = []
        try:
            for index, name in enumerate(sorted(set(names))):
                if name not in params:
                    raise ValueError(f"Model has no parameter {name!r} for vLLM sync")
                filename = f"{index:04d}.pt"
                self.torch.save(params[name].detach(), patch / filename)
                manifest.append({"name": name, "file": filename})
            (patch / "manifest.json").write_text(json.dumps({"weights": manifest}))
            response = self._vllm_request(
                {"action": "load_patch", "patch_path": str(patch)})
        except Exception:
            self._vllm_unselected_trained = False
            raise
        finally:
            shutil.rmtree(patch, ignore_errors=True)
        self._vllm_unselected_trained = bool(unselected_trained)
        return response

    def vllm_gsm8k(self, rows):
        """Run the ordinary post-hoc GSM8K generator and scorer on installed weights."""
        from eval.run_eval import score
        from .data.metamath import is_correct

        if self.standard_gsm8k is None:
            raise RuntimeError("Standard GSM8K evaluation has not been bound")
        response = self._vllm_request({
            "action": "generate",
            "prompts": self.standard_gsm8k["prompts"],
            "max_new_tokens": GSM8K_MAX_NEW_TOKENS,
            "max_length": GSM8K_MAX_LENGTH,
        })
        if len(response["rows"]) != len(rows):
            raise RuntimeError("vLLM returned a different number of GSM8K outcomes")
        self._last_standard_gsm8k_score = score(
            response["rows"], self.standard_gsm8k["golds"])
        values = []
        correctness = [
            is_correct(generated["text"], gold)
            for generated, gold in zip(response["rows"], self.standard_gsm8k["golds"])
        ]
        for generated, correct in zip(response["rows"], correctness):
            generated.update(status="ok", outcome_engine="vllm",
                             correct=bool(correct))
            values.append((generated.pop("correct"), generated))
        return values

    def activation_path(self, reference, role, item_id, module_name):
        return self.scratch / f"{digest(['module-input', reference, role, item_id, module_name])}.pt"

    def verify_standard_loading(self, model, cached):
        """Pilot-only check of full standard loading versus intervention patches.

        Uses all 1000 unchanged GSM8K prompts. The full checkpoint exists only
        inside node-local scratch and is removed even if validation fails.
        """
        if self.vllm_socket is None:
            raise ValueError("Standard-loading parity requires the vLLM worker")
        started = time.perf_counter()
        expected = [row for row in cached["outcomes"]
                    if row.get("role") == "target_eval" and row.get("task") == "gsm8k_dev"]
        with tempfile.TemporaryDirectory(prefix="standard-parity-", dir=self.scratch) as temp:
            model.save_pretrained(temp, safe_serialization=True, max_shard_size="2GB")
            self.tokenizer.save_pretrained(temp)
            response = self._vllm_request({"action": "load_checkpoint", "model_path": temp})
            actual = self.vllm_gsm8k(self.bank["target_eval"]["outcomes"])
        mismatches = []
        for index, ((correct, extra), baseline) in enumerate(zip(actual, expected)):
            # Compare every generated field (text, tokens, cap flags, etc.), not
            # merely aggregate accuracy. No timing fields are returned by Worker.generate.
            if correct != baseline["correct"] or any(
                    baseline.get(key) != value for key, value in extra.items()):
                mismatches.append(index)
        full_score = json_record({"score": self._last_standard_gsm8k_score})["score"]
        score_equal = all(cached["metrics"].get(k) == v for k, v in full_score.items())
        passed = len(actual) == len(expected) == 1000 and not mismatches and score_equal
        atomic_json(self.work / "standard-loading-parity.json", {
            "passed": passed, "n": len(actual), "mismatched_indices": mismatches,
            "standard_score_equal": score_equal, "swap_tier": response["swap_tier"],
            "definition": "same merged BF16 weights; full standard loader versus intervention patch",
            "wall_seconds": time.perf_counter() - started})
        if not passed:
            raise RuntimeError("Standard-loading parity failed; see standard-loading-parity.json")
        print(f"STANDARD LOADING PARITY PASSED: {len(actual)} GSM8K prompts", flush=True)

    def capture_module_inputs(self, model, modules, reference):
        """Capture exact common inputs for the protocol's stratified function-space probe.

        Only frozen target/general context rows are run.  One tensor is serialized per
        example and module so later energy calculations can stream them instead of
        retaining every layer activation in accelerator memory.
        """
        if reference not in ("base", "trained"):
            raise ValueError(reference)
        marker = self.work / f"activation-inputs-{reference}.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved.get("modules") != sorted(modules):
                raise ValueError("Cached activation module scope disagrees with this analysis")
            if any(not (self.scratch / row["file"]).is_file()
                   or file_hash(self.scratch / row["file"]) != row["hash"]
                   for row in saved["rows"]):
                raise ValueError("Frozen activation cache checksum mismatch")
            self._activation_metadata[reference] = saved
            return saved

        rows = []
        for role in ("target_eval", "general_eval"):
            for item in self.bank[role]["rows"]:
                attention = self.inputs(item)["attention_mask"][0].bool()
                captured = {}

                def make_hook(name):
                    def hook(_module, args):
                        if not args or not self.torch.is_tensor(args[0]):
                            raise ValueError(f"{name} did not receive a tensor input")
                        value = args[0].detach()
                        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != attention.numel():
                            raise ValueError(f"Unexpected input shape for {name}: {tuple(value.shape)}")
                        captured.setdefault(name, []).append(value[0, attention].float())
                    return hook

                handles = [module.register_forward_pre_hook(make_hook(name))
                           for name, module in modules.items()]
                try:
                    with self.inference_context():
                        model(**self.inputs(item), use_cache=False)
                finally:
                    for handle in handles:
                        handle.remove()
                missing = set(modules) - set(captured)
                if missing:
                    raise ValueError(f"Selected modules were not called: {sorted(missing)}")
                for name in sorted(captured):
                    tensor = self.torch.cat(captured[name], dim=0)
                    expected = modules[name].weight.shape[1]
                    if tensor.ndim != 2 or tensor.shape[1] != expected:
                        raise ValueError(f"Captured input width disagrees with {name}")
                    path = self.activation_path(reference, role, item["id"], name)
                    self.torch.save(tensor, path)
                    rows.append({"reference": reference, "role": role, "item_id": item["id"],
                                 "module": name, "file": path.name, "hash": file_hash(path),
                                 "n_inputs": tensor.shape[0], "input_width": tensor.shape[1]})
                    del tensor
        saved = {"reference": reference, "bank_hash": self.bank_hash,
                 "modules": sorted(modules), "rows": rows,
                 "cached_input_digest": digest([{k: row[k] for k in
                     ("reference", "role", "item_id", "module", "hash", "n_inputs", "input_width")}
                     for row in rows])}
        atomic_json(marker, saved)
        self._activation_metadata[reference] = saved
        return saved

    def function_geometry(self, module_name, W0, W_star, realized):
        """Measure base->trained and trained->edit energy on their required common inputs."""
        from specint.geometry import output_edit_energy
        output = {}
        comparisons = {
            "base_to_trained": ("base", W_star - W0, W0),
            "trained_to_edit": ("trained", realized - W_star, W_star),
        }
        for label, (reference, edit, reference_weight) in comparisons.items():
            fixed_key = (module_name, label)
            if label == "base_to_trained" and fixed_key in self._function_geometry_cache:
                output[label] = self._function_geometry_cache[fixed_key]
                continue
            has_edit = bool(self.torch.count_nonzero(edit))
            metadata = self._activation_metadata.get(reference)
            if metadata is None:
                marker = self.work / f"activation-inputs-{reference}.json"
                if not marker.is_file():
                    raise RuntimeError(f"Missing {reference} frozen module inputs")
                metadata = json.loads(marker.read_text())
                self._activation_metadata[reference] = metadata
            by_role = {}
            for role in ("target_eval", "general_eval"):
                selected = [row for row in metadata["rows"]
                            if row["module"] == module_name and row["role"] == role]
                edit_energy = reference_energy = 0.0
                n_inputs = 0
                for row in selected:
                    H = self.torch.load(self.scratch / row["file"], map_location="cuda:0",
                                        weights_only=True)
                    if has_edit:
                        edit_energy += output_edit_energy(edit, H)
                    reference_key = (module_name, reference, role, row["file"])
                    if reference_key not in self._function_geometry_cache:
                        self._function_geometry_cache[reference_key] = output_edit_energy(
                            reference_weight, H)
                    reference_energy += self._function_geometry_cache[reference_key]
                    n_inputs += int(H.shape[0])
                    del H
                ratio = edit_energy / reference_energy if reference_energy else None
                by_role[role] = {
                    "sum_output_edit_energy": edit_energy,
                    "reference_output_energy": reference_energy,
                    "relative_output_edit_energy": ratio,
                    "n_inputs": n_inputs,
                    "cached_input_digest": metadata["cached_input_digest"],
                    "status": "ok" if n_inputs and reference_energy else "infeasible",
                    "status_reason": None if n_inputs and reference_energy else
                        "no captured inputs or zero reference output energy",
                }
            output[label] = by_role
            if label == "base_to_trained":
                self._function_geometry_cache[fixed_key] = by_role
        return output

    def row_logits(self, model, row, *, include_hidden=False):
        """Return scored logits, labels, and optionally the full final hidden state.

        Reference KL caches store final hidden states rather than dense vocabulary
        distributions.  The output embedding is unchanged by every supported
        intervention, so applying that same head later reconstructs the reference
        logits without an approximation while reducing transient storage by roughly
        ``vocab_size / hidden_size``.  The complete sequence tensor is retained so
        reconstruction calls the output head with exactly the original shape.
        """
        torch = self.torch
        inputs = self.inputs(row)
        attention = inputs["attention_mask"][0].bool()
        score = torch.tensor(row["score_mask"], device="cuda:0", dtype=torch.bool)
        # logit[t] predicts label[t+1]; context and label must both be valid.
        mask = attention[:-1] & attention[1:] & score[1:]
        captured = []
        handle = None
        if include_hidden:
            def capture_head_input(_module, args):
                if not args or not torch.is_tensor(args[0]):
                    raise ValueError("Output embedding did not receive hidden states")
                captured.append(args[0].detach())
            handle = model.get_output_embeddings().register_forward_pre_hook(
                capture_head_input)
        try:
            with self.inference_context():
                output = model(**inputs, use_cache=False, return_dict=True)
                logits = output.logits[0, :-1][mask]
        finally:
            if handle is not None:
                handle.remove()
        if include_hidden and len(captured) != 1:
            raise ValueError("Expected exactly one output-embedding invocation")
        hidden = captured[0] if include_hidden else None
        labels = inputs["input_ids"][0, 1:][mask]
        return logits, labels, hidden

    def logprob_chunks(self, model, row):
        torch = self.torch
        logits, labels, _ = self.row_logits(model, row)
        for start in range(0, labels.numel(), self.chunk_size):
            lp = torch.log_softmax(logits[start:start+self.chunk_size].float(), dim=-1)
            if not bool(torch.isfinite(lp).all()):
                raise FloatingPointError("Nonfinite full-vocabulary log-probabilities")
            yield start, lp, labels[start:start+self.chunk_size]

    def logprobs(self, model, row):
        yield from self.logprob_chunks(model, row)

    def reference_path(self, reference, role, item_id):
        return self.scratch / f"{digest([reference, role, item_id])}.hidden.pt"

    def reference_logits(self, model, reference, role, row):
        """Reconstruct scored reference logits through the unchanged output head."""
        hidden = self.torch.load(
            self.reference_path(reference, role, row["id"]),
            map_location="cuda:0", weights_only=True)
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("Reference hidden-state cache has the wrong shape")
        with self.inference_context():
            logits = model.get_output_embeddings()(hidden)
        attention = self.torch.tensor(
            row["attention_mask"], device="cuda:0", dtype=self.torch.bool)
        score = self.torch.tensor(row["score_mask"], device="cuda:0", dtype=self.torch.bool)
        mask = attention[:-1] & attention[1:] & score[1:]
        scored = logits[0, :-1][mask]
        if not bool(self.torch.isfinite(scored).all()):
            raise ValueError("Reconstructed reference logits are nonfinite")
        return scored

    def outcome(self, model, row):
        torch = self.torch
        if row["type"] == "choice":
            scores = []
            for choice in row["choices"]:
                total, count = 0.0, 0
                for _, lp, labels in self.logprobs(model, choice):
                    total += float(lp.gather(1, labels[:, None]).double().sum())
                    count += labels.numel()
                if not count:
                    raise ValueError("Choice has no valid continuation tokens")
                scores.append(total / count if row["scoring"] == "mean_logprob" else total)
            prediction = max(range(len(scores)), key=scores.__getitem__)
            return prediction == row["gold_index"], {"choice_scores": scores, "prediction": prediction}
        raise ValueError("GSM8K outcomes are evaluated by the batched path")

    def gsm8k_batch(self, model, rows):
        """Greedy left-padded generation, with recursive GPU-batch isolation.

        The frozen bank contains each prompt's exact PEFT/vLLM dual token cap. Rows
        with different caps are never mixed. If a batch fails (including OOM), it is
        bisected on GPU until only the affected example fails; sibling examples and
        later intervention cells can still complete.
        """
        from .data.metamath import is_correct
        torch = self.torch
        if not rows:
            return []
        caps = {row["max_new_tokens"] for row in rows}
        if len(caps) != 1:
            raise ValueError("A GSM8K generation batch must have one frozen token cap")
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            eos = model.config.eos_token_id
        eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in eos}
        width = max(len(row["prompt_ids"]) for row in rows)
        ids, attention = [], []
        for row in rows:
            missing = width - len(row["prompt_ids"])
            ids.append([pad] * missing + row["prompt_ids"])
            attention.append([0] * missing + [1] * len(row["prompt_ids"]))
        input_ids = torch.tensor(ids, device="cuda:0", dtype=torch.long)
        attention_mask = torch.tensor(attention, device="cuda:0", dtype=torch.long)
        with self.inference_context():
            output = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, do_sample=False,
                num_beams=1, use_cache=True, max_new_tokens=rows[0]["max_new_tokens"],
                eos_token_id=eos, pad_token_id=pad)
        results = []
        for row, sequence in zip(rows, output):
            generated = sequence[width:].tolist()
            # Transformers may retain padding after EOS in the returned rectangle.
            stop = next((index for index, token in enumerate(generated) if token in eos_ids), None)
            if stop is not None:
                generated = generated[:stop + 1]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            correct = bool(is_correct(text, row["gold"]))
            truncated = (len(generated) >= row["max_new_tokens"] and
                         (not generated or generated[-1] not in eos_ids))
            results.append((correct, {"text": text, "generated_ids": generated,
                                      "n_prompt_tokens": len(row["prompt_ids"]),
                                      "n_generated_tokens": len(generated),
                                      "truncated": truncated,
                                      "generation_batch_size_used": len(rows)}))
        return results

    def isolated_gsm8k_batch(self, model, rows):
        try:
            return self.gsm8k_batch(model, rows)
        except Exception as exc:
            if len(rows) == 1:
                return [(None, {"status": "failed",
                                "status_reason": f"{type(exc).__name__}: {exc}",
                                "generation_batch_size_used": 1})]
            # CUDA OOM is recoverable after discarding the failed batch tensors.
            del exc
            self.torch.cuda.empty_cache()
            middle = len(rows) // 2
            return (self.isolated_gsm8k_batch(model, rows[:middle]) +
                    self.isolated_gsm8k_batch(model, rows[middle:]))

    def outcomes(self, model):
        metrics, examples = {}, []
        for role in ("target_eval", "general_eval"):
            started = time.perf_counter()
            by_task = {}
            outcomes = self.bank[role]["outcomes"]
            resolved = {}
            gsm8k = [row for row in outcomes if row["type"] == "gsm8k"]
            if gsm8k and self.vllm_socket is not None:
                try:
                    values = self.vllm_gsm8k(gsm8k)
                except Exception as exc:
                    values = [(None, {"status": "failed", "outcome_engine": "vllm",
                                      "status_reason": f"{type(exc).__name__}: {exc}"})
                              for _ in gsm8k]
                resolved.update((row["id"], value) for row, value in zip(gsm8k, values))
            else:
                # Preserve bank order while batching only equal-cap Transformers prompts.
                for cap in sorted({row["max_new_tokens"] for row in gsm8k}):
                    selected = [row for row in gsm8k if row["max_new_tokens"] == cap]
                    for start in range(0, len(selected), self.generation_batch_size):
                        batch = selected[start:start + self.generation_batch_size]
                        values = self.isolated_gsm8k_batch(model, batch)
                        resolved.update((row["id"], value) for row, value in zip(batch, values))
            for row in outcomes:
                try:
                    if row["type"] == "gsm8k":
                        correct, extra = resolved[row["id"]]
                        if correct is not None:
                            extra["status"] = "ok"
                    else:
                        correct, extra = self.outcome(model, row)
                        extra["status"] = "ok"
                except Exception as exc:
                    correct = None
                    extra = {"status": "failed", "status_reason": f"{type(exc).__name__}: {exc}"}
                by_task.setdefault(row["task"], []).append(int(correct) if correct is not None else None)
                examples.append({"kind": "outcome", "role": role, "item_id": row["id"],
                                 "task": row["task"], "correct": correct, **extra})
            scores = {task: sum(values) / len(values) if None not in values else None
                      for task, values in by_task.items()}
            if role == "target_eval":
                metrics["target_component_scores"] = scores
                metrics["task_accuracy"] = (sum(map(sum, by_task.values())) / sum(map(len, by_task.values()))
                                            if None not in scores.values() else None)
                if gsm8k and self.vllm_socket is not None:
                    if not hasattr(self, "_last_standard_gsm8k_score"):
                        raise RuntimeError("Standard GSM8K evaluation did not complete")
                    standard = dict(self._last_standard_gsm8k_score)
                    if metrics["task_accuracy"] != standard["eval_accuracy"]:
                        raise RuntimeError("Protocol and standard GSM8K scorers disagree")
                    metrics.update(standard)
            else:
                metrics["general_suite_scores"] = scores
            print(f"EVAL outcomes {role}: {len(outcomes)} samples in "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
        return metrics, examples

    def cache_reference(self, model, reference):
        torch = self.torch
        started = time.perf_counter()
        marker = self.work / f"reference-{reference}.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved.get("cache_representation") != "final_hidden_states_native_dtype":
                raise ValueError("Reference prediction cache uses a stale representation")
            if any(not (self.scratch / p).is_file() or file_hash(self.scratch / p) != h
                   for p, h in saved["files"].items()):
                raise ValueError("Reference prediction cache checksum mismatch")
            return
        if self.loss_rows is None:
            raise RuntimeError("Model validation must freeze the KL rows before caching")
        files, rows = {}, []
        for role in ("target_eval", "general_eval"):
            for row in self.loss_rows[role]:
                nll, n = 0.0, 0
                logits, labels, hidden = self.row_logits(
                    model, row, include_hidden=True)
                path = self.reference_path(reference, role, row["id"])
                if (hidden is None
                        or hidden.dtype != model.get_output_embeddings().weight.dtype):
                    raise ValueError(
                        "Reference hidden-state cache must preserve the inference dtype")
                torch.save(hidden, path)
                files[path.name] = file_hash(path)
                for start in range(0, labels.numel(), self.chunk_size):
                    selected = labels[start:start+self.chunk_size]
                    lp = torch.log_softmax(
                        logits[start:start+self.chunk_size].float(), dim=-1)
                    if not bool(torch.isfinite(lp).all()):
                        raise FloatingPointError(
                            "Nonfinite full-vocabulary reference log-probabilities")
                    nll -= float(lp.gather(1, selected[:, None]).double().sum())
                    n += selected.numel()
                rows.append({"role": role, "item_id": row["id"], "n_tokens": n,
                             "nll": nll / n if n else None, "nll_sum": nll})
        metrics, outcomes = self.outcomes(model)
        metrics.update(self.aggregate_losses(rows))
        metrics.update(self.standard_retention_metrics(model))
        self.add_retention_consistency(metrics)
        metrics["measurement_failures"] = [{k: row[k] for k in ("role", "item_id", "status_reason")}
                                            for row in outcomes if row.get("status") == "failed"]
        result = {"metrics": metrics, "rows": rows, "outcomes": outcomes,
                  "wall_seconds": time.perf_counter() - started}
        atomic_json(marker, {"files": files, "cache_representation":
                             "final_hidden_states_native_dtype", **result})
        # Keep endpoint results after the tensor cache is removed.
        atomic_json(self.work / f"reference-results-{reference}.json", result)

    def measure_trained_floor(self, model):
        """Repeat continuous trained-reference losses on identical frozen inputs."""
        marker = self.work / "trained-evaluation-floor.json"
        if marker.exists():
            self.numerical_floor = json.loads(marker.read_text())
            return self.numerical_floor
        reference = json.loads((self.work / "reference-trained.json").read_text())
        repeated = []
        for role in ("target_eval", "general_eval"):
            for row in self.loss_rows[role]:
                nll, n = 0.0, 0
                for _, lp, labels in self.logprobs(model, row):
                    nll -= float(lp.gather(1, labels[:, None]).double().sum())
                    n += labels.numel()
                repeated.append({"role": role, "item_id": row["id"], "n_tokens": n,
                                 "nll": nll / n if n else None, "nll_sum": nll})
        aggregate = self.aggregate_losses(repeated)
        repeated_standard = self.standard_retention_metrics(model)
        floors = {}
        for metric in ("answer_token_nll", "general_text_nll"):
            first, second = reference["metrics"].get(metric), aggregate.get(metric)
            floors[metric] = abs(second - first) if first is not None and second is not None else None
        selection_first = self.selection_nll(model)
        selection_second = self.selection_nll(model)
        floors["selection_nll"] = (abs(selection_second["selection_nll"] -
            selection_first["selection_nll"]) if selection_first["selection_nll"] is not None
            and selection_second["selection_nll"] is not None else None)
        reference_standard = reference["metrics"].get("retention_nll")
        floors["retention_nll"] = (
            abs(repeated_standard["retention_nll"] - reference_standard)
            if reference_standard is not None else None
        )
        result = {"reference": "trained", "bank_hash": self.bank_hash,
                  "definition": "absolute repeated-evaluation difference on identical inputs",
                  "metrics": floors, "repeat_metrics": aggregate,
                  "repeat_standard_retention": repeated_standard,
                  "selection_reference": selection_first,
                  "status": "ok" if all(v is not None for v in floors.values()) else "failed"}
        atomic_json(marker, result)
        self.numerical_floor = result
        return result

    @staticmethod
    def aggregate_losses(rows):
        result = {}
        for role, name in (("target_eval", "answer_token_nll"), ("general_eval", "general_text_nll")):
            valid = [r for r in rows if r["role"] == role and r["n_tokens"]]
            count = sum(r["n_tokens"] for r in valid)
            result[name] = sum(r["nll"] for r in valid) / len(valid) if valid else None
            result[f"{name}_token_pooled"] = sum(r["nll_sum"] for r in valid) / count if count else None
            result[f"{role}_empty_masks"] = sum(r["role"] == role and not r["n_tokens"] for r in rows)
        return result

    @staticmethod
    def add_retention_consistency(metrics):
        """Record agreement of the standard evaluator and KL-panel token NLL."""
        standard = metrics.get("retention_nll")
        panel = metrics.get("general_text_nll_token_pooled")
        metrics["retention_nll_from_kl_panel"] = panel
        metrics["retention_nll_vs_kl_panel_abs_error"] = (
            abs(standard - panel) if standard is not None and panel is not None else None)

    def selection_nll(self, model):
        """Answer-token NLL on the frozen selection bank, used only for pilot ranges."""
        rows = []
        for row in self.bank["selection"]["rows"]:
            nll, n = 0.0, 0
            for _, lp, labels in self.logprobs(model, row):
                nll -= float(lp.gather(1, labels[:, None]).double().sum())
                n += labels.numel()
            rows.append({"item_id": row["id"], "n_tokens": n,
                         "nll": nll / n if n else None, "nll_sum": nll})
        valid = [row for row in rows if row["n_tokens"]]
        count = sum(row["n_tokens"] for row in valid)
        return {"selection_nll": (sum(row["nll"] for row in valid) / len(valid)
                                  if valid else None),
                "selection_nll_token_pooled": (sum(row["nll_sum"] for row in valid) / count
                                                if count else None),
                "selection_empty_masks": len(rows) - len(valid), "rows": rows}

    def __call__(self, model, operator):
        torch = self.torch
        started = time.perf_counter()
        self.last_timings = {}
        references = {r: json.loads((self.work / f"reference-{r}.json").read_text())
                      for r in ("base", "trained")}
        rows = []
        for role in ("target_eval", "general_eval"):
            for row in self.loss_rows[role]:
                nll, n, kl = 0.0, 0, {"base": 0.0, "trained": 0.0}
                logits, labels, _ = self.row_logits(model, row)
                reference_logits = {
                    reference: self.reference_logits(model, reference, role, row)
                    for reference in kl
                }
                if any(value.shape != logits.shape for value in reference_logits.values()):
                    raise ValueError("Reference/variant scored-logit shapes disagree")
                for start in range(0, labels.numel(), self.chunk_size):
                    selected = labels[start:start+self.chunk_size]
                    lp = torch.log_softmax(
                        logits[start:start+self.chunk_size].float(), dim=-1)
                    if not bool(torch.isfinite(lp).all()):
                        raise FloatingPointError(
                            "Nonfinite full-vocabulary variant log-probabilities")
                    nll -= float(lp.gather(1, selected[:, None]).double().sum())
                    n += selected.numel()
                    for reference in kl:
                        ref = torch.log_softmax(
                            reference_logits[reference][start:start+self.chunk_size].float(),
                            dim=-1)
                        if ref.shape != lp.shape or not bool(torch.isfinite(ref).all()):
                            raise ValueError(
                                "Reconstructed reference log-probabilities are invalid")
                        kl[reference] += float((ref.exp().double() * (ref.double() - lp.double())).sum())
                        del ref
                del logits, reference_logits
                rows.append({"kind": "loss", "role": role, "item_id": row["id"],
                             "n_tokens": n, "nll": nll / n if n else None, "nll_sum": nll,
                             **{f"kl_vs_{r}": v / n if n else None for r, v in kl.items()},
                             **{f"kl_sum_vs_{r}": v for r, v in kl.items()}})
                if len(rows) % 25 == 0:
                    print(f"EVAL {operator} KL {len(rows)}/"
                          f"{sum(map(len, self.loss_rows.values()))} documents; "
                          f"{time.perf_counter() - started:.1f}s", flush=True)
        self.last_timings["kl_and_loss_seconds"] = time.perf_counter() - started
        if operator in references:
            metrics = dict(references[operator]["metrics"])
            outcomes = references[operator]["outcomes"]
        else:
            stage = time.perf_counter()
            metrics, outcomes = self.outcomes(model)
            self.last_timings["outcomes_seconds"] = time.perf_counter() - stage
            stage = time.perf_counter()
            metrics.update(self.standard_retention_metrics(model))
            self.last_timings["standard_retention_seconds"] = time.perf_counter() - stage
        metrics.update(self.aggregate_losses(rows))
        self.add_retention_consistency(metrics)
        metrics["measurement_failures"] = [{k: row[k] for k in ("role", "item_id", "status_reason")}
                                            for row in outcomes if row.get("status") == "failed"]
        for reference, cached in references.items():
            metrics[f"token_kl_vs_{reference}"] = {}
            for role in ("target_eval", "general_eval"):
                valid = [r for r in rows if r["role"] == role and r["n_tokens"]]
                count = sum(r["n_tokens"] for r in valid)
                metrics[f"token_kl_vs_{reference}"][role] = {
                    "example_mean": sum(r[f"kl_vs_{reference}"] for r in valid) / len(valid) if valid else None,
                    "token_pooled": sum(r[f"kl_sum_vs_{reference}"] for r in valid) / count if count else None,
                    "n_examples": len(valid),
                    "n_tokens": count,
                    "vocabulary": "full",
                    "max_length": (self.standard_retention["kl_max_length"]
                                   if role == "general_eval" else None),
                }
            def differences(current, baseline):
                return {k: differences(v, baseline[k]) if isinstance(v, dict) else
                        (v - baseline[k] if v is not None and baseline[k] is not None else None)
                        for k, v in current.items() if k in baseline
                        and (v is None or isinstance(v, (int, float, dict)))}
            metrics[f"delta_vs_{reference}"] = differences(metrics, cached["metrics"])
            indexed = {(r["role"], r["item_id"]): r for r in cached["rows"]}
            for row in rows:
                baseline = indexed[row["role"], row["item_id"]]["nll"]
                row[f"nll_delta_vs_{reference}"] = row["nll"] - baseline if row["nll"] is not None else None
            indexed_outcomes = {(r["role"], r["item_id"]): r for r in cached["outcomes"]}
            for row in outcomes:
                baseline = indexed_outcomes[row["role"], row["item_id"]]
                row[f"correct_delta_vs_{reference}"] = (int(row["correct"]) - int(baseline["correct"])
                    if row["correct"] is not None and baseline["correct"] is not None else None)
        pinned_base = self.standard_retention["base_retention_nll"]
        standard_forgetting = metrics["retention_nll"] - pinned_base
        metrics["forgetting"] = {
            "general_text_nll": standard_forgetting,
            "definition": "retention_nll_minus_pinned_pretrained_base",
            "retention_nll": metrics["retention_nll"],
            "base_retention_nll": pinned_base,
            "n_rows": self.standard_retention["n_rows"],
            "max_length": self.standard_retention["max_length"],
            "aggregation": self.standard_retention["aggregation"],
            "standard_200_document_general_text_nll_example_mean":
                metrics["delta_vs_base"]["general_text_nll"],
            "standard_200_document_general_text_nll_token_pooled":
                metrics["delta_vs_base"]["general_text_nll_token_pooled"],
            "general_suite_scores": {k: -v if v is not None else None
                                     for k, v in metrics["delta_vs_base"]["general_suite_scores"].items()}}
        if hasattr(self, "numerical_floor"):
            metrics["numerical_floor"] = self.numerical_floor
        return metrics, rows + outcomes

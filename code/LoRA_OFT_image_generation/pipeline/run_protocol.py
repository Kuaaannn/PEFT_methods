"""Run checkpoint interventions using the shared matrix operators.

    python -m pipeline.run_protocol --checkpoints <id> [...] --store core
"""
from __future__ import annotations

import argparse
import time

import torch

import specint
from specint import ops

from . import manifest as M
from . import merge as MG
from . import model as MD
from . import plan as P
from . import records as R
from . import runner as RUN
from .evaluator import StandardEvaluator


def build_slots(
    ck,
    transformer,
    autocast_adapter_dtype=True,
    print_fn=print,
):
    """Capture the standard-dtype merged checkpoint, then factor it for SPECINT."""
    # Factor and write on the transformer's allocated GPU.
    device = next(transformer.parameters()).device
    sd = MD.adapter_state_dict(ck)
    peft_model = MD.attach(
        ck,
        transformer=transformer,
        device=device,
        autocast_adapter_dtype=autocast_adapter_dtype,
    )
    mm = MG.build_manifest(peft_model, sd)
    names = [entry.name for entry in mm.entries]

    pairs, worst = {}, 0.0
    for e, W0, W_star, chk in MG.iter_pairs(peft_model, mm, device=device, names=names):
        if not chk.passed:
            print_fn(
                f"  WARNING native-vs-merged diagnostic exceeded tolerance for {e.name}: "
                f"relative error {chk.rel_error:.2e}"
            )
        worst = max(worst, chk.rel_error)
        pairs[e.name] = (e, W0.clone(), W_star.clone())
    peft_model.unload()
    del peft_model
    torch.cuda.empty_cache()
    print_fn(f"  merged {len(pairs)} matrices, worst native-vs-merged rel error {worst:.2e}")

    params = dict(transformer.named_parameters())
    slots = []
    for name, (e, W0, W_star) in pairs.items():
        f = ops.factor(W0, W_star, name=name, driver="gesvd")
        f.delta_norm, f.W0_norm            # force the scalars before leaving the device
        slots.append(RUN.MatrixSlot(entry=e, factors=f.to("cpu"), param=params[name]))
        del f
        torch.cuda.empty_cache()
    return mm, slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--store", required=True)
    a = ap.parse_args()

    from peft.utils import infer_device
    from utils import get_pipeline, init_accelerator
    init_accelerator()
    device = infer_device()

    cks = M.load()
    selected = [M.get(c, cks) for c in a.checkpoints]
    subjects = sorted({c.subject for c in selected})
    print(f"{len(selected)} checkpoints, subjects: {', '.join(subjects)}", flush=True)

    store = R.Store(a.store)
    hardware = torch.cuda.get_device_name(0)

    for ck in selected:
        print(f"\n=== {ck.checkpoint_id} ===", flush=True)
        t0 = time.perf_counter()
        cfg = M.load_train_config(ck)
        pipeline = get_pipeline(
            model_id=cfg.model_id,
            dtype=cfg.dtype,
            compile=False,
            peft_config=None,
            autocast_adapter_dtype=cfg.autocast_adapter_dtype,
            use_gc=cfg.use_gc,
            device_type=device,
        )
        evaluator = StandardEvaluator(pipeline, cfg, device=device)
        pipeline.transformer.to(device).eval()
        tf = pipeline.transformer
        mm, slots = build_slots(
            ck,
            tf,
            autocast_adapter_dtype=cfg.autocast_adapter_dtype,
        )
        if cfg.compile:
            # Standard evaluation compiles only after attaching the checkpoint. Build editable
            # merged slots first, then wrap that same transformer at the identical point.
            pipeline.transformer = torch.compile(tf, dynamic=True)
        builder = RUN.VariantBuilder(slots)
        builder.compute_note = "standard_eval"
        builder.restore_base()
        evaluator.prepare_base_reference()
        builder.restore_trained()
        bank_hash = evaluator.bank_hash()
        floor = builder.reconstruction_floor()
        print(f"  factored in {time.perf_counter() - t0:.0f}s, floor {floor:.3e}, "
              f"evaluation inputs {bank_hash[:12]}, profile {builder.dtype_profile}", flush=True)

        # Match the LLM runner's independent hard preflight. This must run even when a measured
        # restore record is already cached, so resume can never bypass restoration validation.
        preflight = builder.apply("restore", P.make_cell("restore", {}, bank_hash).build)
        preflight_rows = preflight["rows"]
        preflight_ok = (
            preflight["status"] not in ("failed", "infeasible")
            and len(preflight_rows) == len(slots)
            and all(row.get("restoration_verified", False) for row in preflight_rows)
        )
        builder.restore_trained()
        if not preflight_ok:
            raise RuntimeError("restoration preflight failed; refusing all measured cells")
        print(f"  restoration verified {len(preflight_rows)}/{len(slots)} matrices", flush=True)

        cells = [P.make_cell(op, params, bank_hash) for op, params in P.PLAN_22
                 if not op.startswith("rotation_topk_")]
        for cell in cells:
            t1 = time.perf_counter()
            rec = RUN.measure(builder, cell.operator, cell.build, evaluator.measure,
                              checkpoint=ck, module_manifest_hash=mm.hash(), bank_hash=bank_hash,
                              operator_params=cell.params, store=store,
                              rng_ids=cell.rng_ids or None, hardware=hardware,
                              opposite=cell.opposite)
            if rec is None:
                print(f"  {cell.operator:16} {str(cell.params)[:30]:30} cached", flush=True)
                continue
            mm_ = rec.metrics
            print(f"  {cell.operator:16} {str(cell.params)[:30]:30} {rec.status:11} "
                  f"dino {mm_.get('test dino_similarity', float('nan')):.4f}  "
                  f"drift {mm_.get('drift', float('nan')):.4f}  "
                  f"({time.perf_counter() - t1:.0f}s)", flush=True)
            if cell.operator == "restore":
                rows = rec.geometry.get("matrix_rows", [])
                if not rows or not all(row.get("restoration_verified", False) for row in rows):
                    raise RuntimeError("restoration verification failed; refusing remaining cells")
        builder.restore_trained()
        del evaluator, pipeline, tf, slots, builder
        torch.cuda.empty_cache()

    print(f"\nrecords -> {store.path}", flush=True)
    print(f"library {specint.__version__} {specint.library_hash()[:12]}, "
          f"contract {R.contract.version()}")


if __name__ == "__main__":
    main()

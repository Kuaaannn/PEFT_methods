"""Variant construction and measurement for the shared SPECINT protocol.

The orchestration that turns a checkpoint plus an operator into a record:

    factor once  ->  build the edit per matrix  ->  write it into the model  ->  measure the
    weights that are actually there  ->  evaluate  ->  record

Three protocol rules are enforced here rather than left to the caller:

* geometry is computed on the tensor actually stored by the model, after its normal dtype cast,
  never only on the planned FP32 tensor;
* non-selected tensors stay at the trained state; only the ``base`` reference restores the whole
  model, which is done by copying immutable cached weights rather than by unmerging;
* an infeasible or unmatched cell is recorded with a reason, never silently repaired or skipped.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

import specint
from specint import geometry as G
from specint import ops
from specint.rng import rng_key

from . import records as R

def dtype_profile(param_dtype) -> str:
    """Record the normal model dtype plus the shared geometry precision."""
    return f"store_{str(param_dtype).replace('torch.', '')}/factor_fp32/reduce_fp64"


@dataclass
class MatrixSlot:
    """One adapted matrix: its factors (in host memory) and a handle on the model parameter.

    Factors are held on the CPU and streamed to the parameter's device one matrix at a time.
    Keeping all of them resident does not fit beside the model.
    """
    entry: object                  # pipeline.modules.ModuleEntry
    factors: ops.Factors
    param: torch.nn.Parameter

    def live(self) -> ops.Factors:
        """Factors on the parameter's device, for the duration of one edit."""
        return self.factors.to(self.param.device)


class VariantBuilder:
    """Holds the factors for one checkpoint and writes variants into a live model."""

    def __init__(self, slots: list, print_fn=print, compute_note: str = ""):
        self.slots = slots
        self.print_fn = print_fn
        self.compute_note = compute_note
        self._floor = None
        self._band_common_d_rel_max = None

    @property
    def dtype_profile(self) -> str:
        """Storage and compute precision together: both change what a cell measures."""
        base = dtype_profile(self.slots[0].param.dtype)
        return f"{base}/{self.compute_note}" if self.compute_note else base

    # -- weight movement ------------------------------------------------------------------
    def _write(self, slot: MatrixSlot, W: torch.Tensor) -> torch.Tensor:
        """Copy into the model and return what is actually stored, after the dtype cast."""
        slot.param.data.copy_(W.to(slot.param.dtype))
        return slot.param.data.detach().to(torch.float32)

    def restore_trained(self) -> None:
        for s in self.slots:
            s.param.data.copy_(s.factors.W_star.to(device=s.param.device, dtype=s.param.dtype))

    def restore_base(self) -> None:
        for s in self.slots:
            s.param.data.copy_(s.factors.W0.to(device=s.param.device, dtype=s.param.dtype))

    # -- the measured reconstruction floor -------------------------------------------------
    def reconstruction_floor(self) -> float:
        """Network reconstruction floor: the realized distance of the ``reconstruct`` operator.

        This is the yardstick that decides whether a small requested edit is resolvable. It is
        measured through the same write path as every edit, so it includes the storage cast.
        """
        if self._floor is None:
            agg = G.EnergyAggregator()
            for s in self.slots:
                f = s.live()
                realized = self._write(s, ops.op_reconstruct(f).W)
                agg.add("floor", G._fro2(realized - f.W_star), f.W0_norm ** 2)
                del f
                torch.cuda.empty_cache()
            self.restore_trained()
            self._floor = agg.value("floor") or 0.0
        return self._floor

    # -- building one variant ---------------------------------------------------------------
    def apply(
        self,
        operator: str,
        build: Callable,
        tol: float = 0.01,
        opposite: Optional[Callable] = None,
    ) -> dict:
        """Build ``operator`` on every slot, write it, and measure what landed.

        ``build(factors, entry) -> specint.ops.Edit``. Returns a summary carrying the status,
        the per-matrix geometry rows and the network aggregates.
        """
        agg = G.EnergyAggregator()
        rows, infeasible = [], []
        requested_energy = 0.0
        realized_energy = 0.0

        for s in self.slots:
            f = s.live()
            edit = build(f, s.entry)
            if not edit.feasible:
                infeasible.append({"matrix": s.entry.name, "reason": edit.reason,
                                   **{k: v for k, v in edit.info.items()}})
                rows.append({
                    "matrix": s.entry.name,
                    "role": s.entry.role,
                    "layer": s.entry.layer,
                    "block_type": s.entry.block_type,
                    "m": s.entry.m,
                    "n": s.entry.n,
                    "operator": operator,
                    "status": "infeasible",
                    "status_reason": edit.reason,
                    "operator_info": edit.info,
                })
                del f
                continue
            planned_edit_norm = G._fro(edit.W - f.W_star)
            realized = self._write(s, edit.W)
            g, planned = G.edit_geometry(f, edit, realized)
            agg.add_geometry(g)
            if g.s_m is not None:
                agg.add("s", g.s_m ** 2 * ops._fro2(f.s0), ops._fro2(f.s0))
            else:
                agg.add("s", 0.0, 0.0)
            agg.add(
                "floor",
                ((f.rebuild_floor or 0.0) * f.W0_norm) ** 2,
                f.W0_norm ** 2,
            )
            requested_energy += planned_edit_norm ** 2
            realized_energy += g.edit_norm ** 2
            row = g.as_dict()
            row.update(matrix=s.entry.name, role=s.entry.role, layer=s.entry.layer,
                       block_type=s.entry.block_type, m=s.entry.m, n=s.entry.n,
                       operator=operator, planned=planned.as_dict(),
                       changed_entries=int((realized != f.W_star).sum()),
                       planned_edit_norm=planned_edit_norm,
                       planned_weight_norm=ops._fro(edit.W),
                       realized_weight_norm=ops._fro(realized),
                       storage_cast_relative_error=(ops._fro(realized - edit.W) /
                           max(f.W0_norm, 1e-30)),
                       storage_dtype=str(s.param.dtype).replace("torch.", ""),
                       solver=f.solver, rebuild_floor=f.rebuild_floor,
                       base_rebuild_floor=f.base_rebuild_floor,
                       degenerate=f.degenerate,
                       degenerate_replacement_spread=f.degenerate_replacement_spread,
                       operator_info=edit.info, status="ok", status_reason=None)
            if g.b_m is None or g.er_norm_fraction is None:
                row["ratio_reason"] = "zero base or update norm; affected ratios are undefined"
            if "g" in edit.info or "c" in edit.info:
                row["matching"] = {}
                for axis, target, attribute in (
                    ("base", f.restoration_base_distance, "base_distance"),
                    ("trained", f.restoration_edit_norm, "edit_norm"),
                ):
                    row["matching"][axis] = {
                        "target_distance": target,
                        "planned_distance": getattr(planned, attribute),
                        "realized_distance": getattr(g, attribute),
                        "planned_relative_error": G.matching_error(getattr(planned, attribute), target),
                        "realized_relative_error": G.matching_error(getattr(g, attribute), target),
                    }
                base_control = operator == "match_base"
                target = f.restoration_base_distance if base_control else f.restoration_edit_norm
                attribute = "base_distance" if base_control else "edit_norm"
                for kind, measurement in (("planned", planned), ("realized", g)):
                    st, err = G.check_match(getattr(measurement, attribute), target, tol)
                    row[f"{kind}_match_error"] = err
                    row["match_axis"] = "base" if base_control else "trained"
                    if st != "ok":
                        row.update(status=st, status_reason="undefined or >1% norm match error")
            if "requested_edit_norm" in edit.info:
                requested = edit.info["requested_edit_norm"]
                err = G.matching_error(g.edit_norm, requested)
                row.update(requested_edit_norm=requested, realized_edit_norm_match_error=err)
                if err is None or err > tol:
                    row.update(status="unmatched",
                               status_reason="realized spectral edit missed its requested norm by >1%")
            if operator == "global_gain":
                requested = edit.params["target_edit_norm"]
                err = G.matching_error(g.edit_norm, requested)
                row.update(shape_matched_target_edit_norm=requested, shape_match_relative_error=err)
                if err is None or err > tol:
                    row.update(status="unmatched",
                               status_reason="global gain missed the shape edit norm by >1%")
            if operator == "spectral_shape":
                err = G.matching_error(ops._fro(realized), ops._fro(f.W_star))
                row["spectral_norm_preservation_relative_error"] = err
                if err is None or err > tol:
                    row.update(status="unmatched",
                               status_reason="shape edit failed to preserve spectral/Frobenius norm")
            if operator == "restore":
                verified, tolerance = G.restoration_valid(
                    f, realized, g, row["storage_cast_relative_error"])
                row.update(
                    restoration_verified=verified,
                    restoration_spectrum_tolerance=tolerance,
                )
                if not verified:
                    row.update(status="failed",
                               status_reason="restored spectrum failed its numerical invariant")
            if opposite is not None:
                opposite_result = opposite(f, s.entry)
                if opposite_result is not None:
                    opposite_params, other = opposite_result
                    if other.feasible:
                        realized_other = other.W.to(s.param.dtype).float()
                        numerator = ops._fro((realized - f.W_star) + (realized_other - f.W_star))
                        denominator = max(g.edit_norm, ops._fro(realized_other - f.W_star), 1e-30)
                        row["antithetic_postcast_relative_error"] = numerator / denominator
                        row["antithetic_operator_params"] = opposite_params
                    else:
                        row["antithetic_postcast_relative_error"] = None
                        row["antithetic_status_reason"] = other.reason
                    del other
            if operator == "spectral_band":
                row["largest_common_paired_d_rel"] = self.band_common_d_rel_max()
            if operator not in ("base", "trained", "reconstruct") and G.below_floor(
                planned.edit_norm, (f.rebuild_floor or 0.0) * f.W0_norm
            ) and row["status"] not in ("failed", "infeasible"):
                row.update(status="unresolved", status_reason="edit at/below reconstruction floor")
            rows.append(row)
            del f, edit, realized
            torch.cuda.empty_cache()

        unmatched = [
            {"matrix": row["matrix"], "status_reason": row.get("status_reason")}
            for row in rows if row.get("status") == "unmatched"
        ]
        status, reason = "ok", None
        if infeasible:
            status = "infeasible"
            reason = f"{len(infeasible)} of {len(self.slots)} matrices refused; " \
                     f"first: {infeasible[0]['reason']}"
        elif any(row["status"] == "failed" for row in rows):
            status, reason = "failed", "see per-matrix status reasons"
        elif any(row["status"] == "infeasible" for row in rows):
            status, reason = "infeasible", "see per-matrix status reasons"
        elif unmatched:
            status = "unmatched"
            reason = f"{len(unmatched)} matrices exceeded the {tol:.0%} matching tolerance"
        elif any(row["status"] == "unresolved" for row in rows):
            status, reason = "unresolved", "see per-matrix status reasons"
        else:
            req = requested_energy ** 0.5
            if req > 0 and G.below_floor(req, self.reconstruction_floor() * self._network_W0_norm()):
                status = "unresolved"
                reason = f"requested edit norm {req:.3e} is at or below the measured " \
                         f"reconstruction floor"

        network = {
            **agg.as_dict(),
            "requested_edit_norm": requested_energy ** 0.5,
            "realized_edit_norm": realized_energy ** 0.5,
            "edit_norm": sum(row.get("edit_norm", 0.0) ** 2 for row in rows) ** 0.5,
            "base_distance": sum(row.get("base_distance", 0.0) ** 2 for row in rows) ** 0.5,
            "reconstruction_floor": self.reconstruction_floor(),
        }
        paired = [
            row["antithetic_postcast_relative_error"]
            for row in rows
            if row.get("antithetic_postcast_relative_error") is not None
        ]
        if paired:
            network["antithetic_postcast_relative_error_max"] = max(paired)
        if infeasible:
            self.restore_trained()
        return {
            "status": status, "status_reason": reason, "rows": rows,
            "evaluable": bool(rows) and status not in ("failed", "infeasible"),
            "infeasible": infeasible, "unmatched": unmatched,
            "network": network,
        }

    def band_common_d_rel_max(self):
        """Largest relative band amplitude feasible for every selected matrix and sign."""
        if self._band_common_d_rel_max is not None:
            return self._band_common_d_rel_max
        common = float("inf")
        for slot in self.slots:
            f = slot.live()
            spectrum_norm = float(torch.linalg.vector_norm(f.s_star.double()))
            if spectrum_norm == 0:
                common = 0.0
            else:
                for band in ops.BANDS:
                    amplitude, _ = ops.amplitude_band(f, band, 1.0)
                    if amplitude is None:
                        common = 0.0
                        continue
                    common = min(common, ops.max_paired_scale(f, amplitude) / spectrum_norm)
            del f
        self._band_common_d_rel_max = common if common != float("inf") else None
        return self._band_common_d_rel_max

    def _network_W0_norm(self) -> float:
        if "W0_norm_network" not in getattr(self, "_scalars", {}):
            self._scalars = getattr(self, "_scalars", {})
            self._scalars["W0_norm_network"] = sum(
                s.factors.W0_norm ** 2 for s in self.slots) ** 0.5
        return self._scalars["W0_norm_network"]


def measure(builder: VariantBuilder, operator: str, build: Callable, evaluate: Callable,
            *, checkpoint, module_manifest_hash: str, bank_hash: str,
            operator_params: dict, store: R.Store, rng_ids=None,
            hardware: str = "", force: bool = False,
            opposite: Optional[Callable] = None) -> Optional[R.Record]:
    """Build, evaluate and record one cell. Returns None when the cell is already stored."""
    key_rng = rng_key(*rng_ids) if rng_ids else None
    rid = R.cache_key(checkpoint_hash=checkpoint.checkpoint_hash,
                      module_manifest_hash=module_manifest_hash, operator=operator,
                      operator_params=operator_params, rng_key=key_rng,
                      dtype_profile=builder.dtype_profile, bank_hash=bank_hash)
    if store.has(rid) and not force:
        return None

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    built = builder.apply(operator, build, opposite=opposite)

    metrics, status, reason = {}, built["status"], built["status_reason"]
    if built["evaluable"]:
        try:
            metrics = evaluate()
            required = ("test dino_similarity", "drift")
            invalid = [
                key for key in required
                if metrics.get(key) is None
                or not isinstance(metrics.get(key), (int, float))
                or not math.isfinite(float(metrics[key]))
            ]
            if invalid:
                status = "failed"
                reason = ("required primary metrics are absent or non-finite: "
                          + ", ".join(invalid))
        except Exception as exc:                      # a failed evaluation is data, not a crash
            status, reason = "failed", f"{type(exc).__name__}: {exc}"

    rec = R.Record(
        record_id=rid,
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_hash=checkpoint.checkpoint_hash,
        module_manifest_hash=module_manifest_hash,
        operator_id=operator,
        operator_params=operator_params,
        rng_key=key_rng,
        dtype_profile=builder.dtype_profile,
        bank_hash=bank_hash,
        status=status,
        status_reason=reason,
        metrics=metrics,
        geometry={"network": built["network"],
                  "n_infeasible": len(built["infeasible"]),
                  "n_unmatched": len(built["unmatched"]),
                  "infeasible_examples": built["infeasible"][:3],
                  "matrix_rows": built["rows"]},
        wall_seconds=round(time.perf_counter() - t0, 2),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        hardware=hardware,
    )
    store.append(rec)
    return rec

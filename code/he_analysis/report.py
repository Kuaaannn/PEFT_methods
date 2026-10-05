"""Small saved-statistic tables and plots; never loads weights or recomputes HE."""
from collections import defaultdict
import csv
from pathlib import Path
import statistics

from .plan import atomic_json, digest, read, require
from .runner import aggregate, task_directory, valid_record

METRICS = ("relative_he_change", "absolute_relative_he_change", "pair_cosine_rms_change",
           "pair_kernel_relative_l1_change")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(plan, *, plots=False):
    output = Path(plan["output_root"]) / "report"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints, matrices, groups, incomplete = [], [], defaultdict(list), []
    projections = []
    for task in plan["tasks"]:
        directory = task_directory(plan, task["index"])
        meta = directory / "metadata.json"
        if not meta.exists() or read(meta).get("status") not in {"complete", "complete_with_issues"}:
            incomplete.append(task["index"])
            continue
        identity = digest({"plan": plan["identity"], "task": task})
        require(read(meta)["identity"] == identity, "Report has mixed plan identities")
        records = []
        for i, entry in enumerate(plan["modules"][task["model"]]):
            path = directory / f"matrix_{i:03d}.json"
            require(valid_record(path, identity, entry["name"]), "Incomplete completed checkpoint")
            row = read(path)
            records.append(row)
            matrices.append({"index": task["index"], "model": task["model"], "method": task["method"],
                             "capacity": task["capacity"], "seed": task["seed"], **row["entry"],
                             "status": row["measurement"]["status"],
                             **{key: row["measurement"][key] for key in METRICS}})
        stats = aggregate(records)
        point = {k: task[k] for k in ("index", "model", "method", "capacity", "budget", "learning_rate", "seed")}
        point.update(stats)
        checkpoints.append(point)
        by_projection = defaultdict(list)
        for row in records:
            by_projection[(row["entry"]["block_type"], row["entry"]["projection_type"])].append(row)
        for (block_type, projection), subset in sorted(by_projection.items()):
            projections.append({k: task[k] for k in ("index", "model", "method", "capacity", "seed")}
                               | {"block_type": block_type, "projection_type": projection} | aggregate(subset))
        # Never silently aggregate a partial set of valid matrices as a full result.
        if stats["invalid_matrix_count"] == 0:
            groups[(task["model"], task["method"], task["capacity"], task["learning_rate"])].append(point)
    seeds = []
    for (model, method, capacity, lr), rows in sorted(groups.items()):
        point = {"model": model, "method": method, "capacity": capacity, "learning_rate": lr,
                 "n_seeds": len(rows), "complete_three_seeds": len(rows) == 3}
        for metric in METRICS:
            values = [row["macro_" + metric] for row in rows]
            point[metric + "_mean"] = statistics.mean(values)
            point[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        seeds.append(point)
    atomic_json(output / "summary.json", {"plan_identity": plan["identity"], "incomplete_indices": incomplete,
                "checkpoint_count": len(checkpoints), "checkpoints": checkpoints, "seed_aggregates": seeds})
    write_csv(output / "matrices.csv", matrices)
    write_csv(output / "seed_aggregates.csv", seeds)
    if projections:
        fields = sorted(set().union(*(row.keys() for row in projections)))
        write_csv(output / "projections.csv", [{k: row.get(k) for k in fields} for row in projections])
    # Invalid checkpoints can lack scalar metrics, so normalize the CSV field set.
    if checkpoints:
        fields = sorted(set().union(*(row.keys() for row in checkpoints)))
        write_csv(output / "checkpoints.csv", [{k: row.get(k) for k in fields} for row in checkpoints])
    if plots and seeds:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for model in sorted({row["model"] for row in seeds}):
            points = [row for row in seeds if row["model"] == model]
            fig, axes = plt.subplots(1, 2, figsize=(max(9, len(points) * .65), 4), constrained_layout=True)
            for ax, key, label in zip(axes, ("absolute_relative_he_change", "pair_cosine_rms_change"),
                                       ("Mean |relative HE change|", "Mean pairwise cosine RMS drift")):
                values = [row[key + "_mean"] for row in points]
                errors = [row[key + "_sd"] or 0 for row in points]
                ax.errorbar(range(len(points)), values, yerr=errors, fmt="o", capsize=3)
                positives = [v for v in values if v > 0]
                ax.set_yscale("symlog", linthresh=min(positives) / 10 if positives else 1e-12)
                ax.set_xticks(range(len(points)), [f"{r['method']} {r['capacity']}\nn={r['n_seeds']}" for r in points], rotation=70)
                ax.set_ylabel(label)
                ax.grid(axis="y", alpha=.25)
            fig.suptitle(f"{model}: all adapted matrices; mean ± seed SD")
            for suffix in ("png", "pdf"):
                fig.savefig(output / f"{model}_he_and_cosine_drift.{suffix}", dpi=180)
            plt.close(fig)
    print(f"Reported {len(checkpoints)}/{len(plan['tasks'])} checkpoints: {output}", flush=True)
    return {"complete": len(checkpoints), "incomplete": incomplete, "seed_aggregates": seeds}

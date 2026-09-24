#!/usr/bin/env python3
"""Empirical CDFs of per-task steps/resources and per-step call times.

Run after analyze_task_metrics.py: python characterization/visual/plot_task_metrics.py RUN_DIR
Each task has equal weight in task CDFs; each step has equal weight in timing CDFs.
Includes failed model attempts and zero-tool steps. Missing/nonfinite values are
excluded per series and reported in cdf_summary.json. No smoothing is applied.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import typer


def ecdf(values) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    x, counts = np.unique(values, return_counts=True)
    return x, np.cumsum(counts) / len(values) if len(values) else np.array([])


def draw(ax, series: dict, xlabel: str, population: str, summary: dict):
    for label, values in series.items():
        x, y = ecdf(values)
        observations = np.asarray(values, dtype=float)
        finite = observations[np.isfinite(observations)]
        summary[label] = {"count": int(len(finite)),
                          "excluded": int(len(observations) - len(finite)),
                          "population": population, "measurement": xlabel,
                          "average": float(np.mean(finite)) if len(finite) else None,
                          "p50": float(np.percentile(finite, 50)) if len(finite) else None,
                          "p99": float(np.percentile(finite, 99)) if len(finite) else None,
                          "x": x.tolist(), "cdf": y.tolist()}
        if len(x):
            ax.step(np.r_[x[0], x], np.r_[0, y], where="post", linewidth=1.8,
                    label=f"{label} (n={summary[label]['count']})")
    ax.set(xlabel=xlabel, ylabel=f"Fraction of {population} ≤ x", ylim=(0, 1.04))
    ax.grid(alpha=.2, linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    if ax.lines:
        ax.legend(frameon=False)
    else:
        ax.text(.5, .5, "No observations", ha="center", transform=ax.transAxes)
    rows = [[label, *[f"{summary[label][key]:.3g}" if summary[label][key] is not None else "—"
                      for key in ("average", "p50", "p99")]] for label in series]
    table = ax.table(cellText=rows, colLabels=["Metric", "Average", "p50", "p99"],
                     colWidths=[.4, .2, .2, .2], cellLoc="center", bbox=[0, -.48, 1, .26])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#eef1f5" if row == 0 else "#f8f9fb")
        if row == 0:
            cell.set_text_props(weight="bold")


def main(run_dir: Path):
    """Create three PNG figures from a run's task_metrics CSV files."""
    metrics = run_dir / "task_metrics"
    tasks, steps = pd.read_csv(metrics / "tasks.csv"), pd.read_csv(metrics / "steps.csv")
    out = metrics / "plots"
    out.mkdir(exist_ok=True)
    report = {"run": str(run_dir.resolve()), "definition": __doc__, "series": {}}
    subtitle = f"{len(tasks)} task(s), {len(steps)} timed step(s)"
    if len(tasks) == 1:
        subtitle += " · Task CDFs contain only one observation"

    fig, ax = plt.subplots(figsize=(7, 6))
    draw(ax, {"Steps per task": tasks.steps}, "Number of steps", "tasks", report["series"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    figures = [(fig, "task_steps_cdf", "Per-task step count")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw(axes[0], {"Average CPU": tasks.cpu_avg_percent, "Peak CPU": tasks.cpu_peak_percent},
         "Container CPU utilization (% of whole host)", "tasks", report["series"])
    draw(axes[1], {"Average memory": tasks.memory_avg_mib, "Peak memory": tasks.memory_peak_mib},
         "Container memory footprint (MiB)", "tasks", report["series"])
    figures.append((fig, "task_resources_cdf", "Per-task cgroup resources"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw(axes[0], {"Tool time": steps.tool_s}, "Total tool-call time per step (s)", "steps", report["series"])
    draw(axes[1], {"Inference time": steps.inference_s}, "Model-call time per step (s)", "steps", report["series"])
    figures.append((fig, "step_times_cdf", "Per-step elapsed call times"))

    for fig, name, title in figures:
        fig.suptitle(f"{title}\n{subtitle}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=180)
        plt.close(fig)
    coordinates = {label: {key: stats.pop(key) for key in ("x", "cdf")}
                   for label, stats in report["series"].items()}
    report["percentile_method"] = "Linear interpolation between sorted observations (NumPy default)."
    (out / "cdf_data.json").write_text(json.dumps({"run": report["run"], "series": coordinates}, indent=2) + "\n")
    (out / "cdf_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    table = pd.DataFrame.from_dict(report["series"], orient="index").rename_axis("metric")
    typer.echo(table[["count", "average", "p50", "p99"]].to_string(float_format=lambda value: f"{value:.4f}"))
    typer.echo(f"Wrote 3 CDF figures (PNG), summary JSON, and exact CDF data to {out}")


if __name__ == "__main__":
    typer.run(main)

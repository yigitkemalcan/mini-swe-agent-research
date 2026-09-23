#!/usr/bin/env python3
"""Plot one system_metrics run into <system_metrics>/plots/.

Outputs:

  gpu_timeseries.png            individual GPUs and average-across-GPUs in separate panels
  gpu_distributions.png         pooled-over-GPUs versus average-across-GPUs ECDFs
  gpu_per_gpu.png               min-max, average, p50 and p99 for every GPU
  container_timeseries.png      individual/average sandbox CPU and memory, plus controller memory
  container_distributions.png   pooled-over-containers versus average-across-containers ECDFs

Time is in seconds since the task analysis window started. Use
``--time-range START END`` to restrict every figure to that interval; omit it
to plot first task setup through last task completion (within logger coverage).
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_system_metrics import (  # noqa: E402
    CONTAINER_UNITS,
    GPU_METRICS,
    agent_process_rows,
    container_average_across,
    container_rows,
    gpu_average_across,
    gpu_metric,
    in_window,
    load_run,
)

POOLED, AVERAGE, INDIVIDUAL, INK = "#0072B2", "#D55E00", "#9A9A9A", "#333333"
GPU_COLORS = plt.get_cmap("tab10").colors
LABELS = {
    "gpu_utilization_percent": ("GPU utilization: share of time any kernel ran", "Utilization (%)"),
    "graphics_engine_active_percent": ("Graphics engine active: share of time the engine was busy", "Active time (%)"),
    "sm_active_percent": ("SM active: share of time SMs had work, mean over SMs", "Active time (%)"),
    "power_watts": ("Power draw", "Power (W)"),
    "power_instant_watts": ("Instant power draw", "Power (W)"),
    "cpu_utilization_percent": ("CPU utilization: share of all host CPUs used", "CPU utilization (%)"),
    "memory_used_mib": ("Memory used (excluding reclaimable file cache)", "Memory (MiB)"),
}
TIME_LABEL = "Time since window start (s)"
CDF_LABEL = "Cumulative fraction of samples"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": "#BBBBBB",
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
    }
)


def panels(n: int, cols: int, height: float, **kwargs):
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, height * rows), squeeze=False, **kwargs)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat[:n]


def ecdf(ax, values, color: str, label: str):
    values = np.asarray(values, dtype=float)
    values = np.sort(values[~np.isnan(values)])
    if len(values):
        ax.step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            color=color,
            lw=1.4,
            label=label,
        )


def mark_percentiles(ax):
    for quantile, name in [(0.5, "p50"), (0.99, "p99")]:
        ax.axhline(quantile, color=INK, lw=0.6, ls=":")
        ax.text(
            1.0,
            quantile,
            f" {name}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=7,
            color=INK,
        )
    ax.set_ylim(0, 1.02)


def label_panel(ax, metric: str, xlabel: str, ylabel: str):
    ax.set_title(LABELS[metric][0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelbottom=True)


def save(fig, path: Path, title: str, subtitle: str):
    fig.text(
        0.01,
        1,
        title,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        transform=offset_copy(fig.transFigure, fig=fig, y=-8, units="points"),
    )
    fig.text(
        0.01,
        1,
        subtitle,
        ha="left",
        va="top",
        fontsize=8,
        color=INK,
        transform=offset_copy(fig.transFigure, fig=fig, y=-24, units="points"),
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 40 / (fig.get_figheight() * 72)))
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_gpus(
    gpu: pd.DataFrame,
    gpu_ids: list,
    start: int,
    view_start: int,
    view_end: int,
    interval_ns: float,
    staleness_ns: float,
    out: Path,
    subtitle: str,
):
    grid = np.arange(view_start, view_end, interval_ns)
    all_series = {metric: gpu_metric(gpu, metric) for metric in GPU_METRICS}
    series = {metric: in_window(samples, view_start, view_end) for metric, samples in all_series.items()}
    averages = {
        metric: gpu_average_across(samples, gpu_ids, grid, staleness_ns)
        for metric, samples in all_series.items()
    }
    grid_s = (grid - start) / 1e9
    x_limits = ((view_start - start) / 1e9, (view_end - start) / 1e9)

    fig, axes = plt.subplots(
        len(GPU_METRICS),
        2,
        figsize=(13, 2.5 * len(GPU_METRICS)),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    for row, (metric, samples) in enumerate(series.items()):
        individual_ax, average_ax = axes[row]
        for color_index, (gpu_id, group) in enumerate(samples.groupby("gpu")):
            individual_ax.plot(
                (group["t_mono_ns"] - start) / 1e9,
                group["value"],
                color=GPU_COLORS[color_index % len(GPU_COLORS)],
                lw=0.8,
                alpha=0.85,
                label=f"GPU {gpu_id}",
            )
        average_ax.plot(grid_s, averages[metric], color=AVERAGE, lw=1.2, label="average across GPUs")
        label_panel(individual_ax, metric, TIME_LABEL, LABELS[metric][1])
        label_panel(average_ax, metric, TIME_LABEL, LABELS[metric][1])
        individual_ax.set_title(f"{LABELS[metric][0]} — individual GPUs")
        average_ax.set_title(f"{LABELS[metric][0]} — average across GPUs")
        individual_ax.set_xlim(*x_limits)
        average_ax.set_xlim(*x_limits)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=4)
    axes[0, 1].legend(loc="lower left", bbox_to_anchor=(0, 1.12))
    save(fig, out / "gpu_timeseries.png", "GPU metrics over time", subtitle)

    fig, axes = panels(len(GPU_METRICS), 3, 3.0)
    for ax, (metric, samples) in zip(axes, series.items()):
        ecdf(ax, samples["value"].to_numpy(float), POOLED, "pooled over GPUs")
        ecdf(ax, averages[metric], AVERAGE, "average across GPUs")
        mark_percentiles(ax)
        label_panel(ax, metric, LABELS[metric][1], CDF_LABEL)
    axes[0].legend(loc="upper left")
    save(fig, out / "gpu_distributions.png", "GPU metric distributions", subtitle)

    fig, axes = panels(len(GPU_METRICS), 3, 3.0)
    for ax, (metric, samples) in zip(axes, series.items()):
        by_gpu = samples.groupby("gpu")["value"]
        x = np.arange(by_gpu.ngroups)
        ax.vlines(x, by_gpu.min(), by_gpu.max(), color=INDIVIDUAL, lw=2, label="min to max")
        ax.scatter(x, by_gpu.mean(), color=AVERAGE, s=24, zorder=3, label="avg")
        ax.scatter(x, by_gpu.median(), color=POOLED, marker="D", s=18, zorder=3, label="p50")
        ax.scatter(x, by_gpu.quantile(0.99), color=INK, marker="^", s=22, zorder=3, label="p99")
        ax.set_xticks(x, [str(gpu_id) for gpu_id in by_gpu.groups])
        label_panel(ax, metric, "GPU index", LABELS[metric][1])
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=4)
    save(fig, out / "gpu_per_gpu.png", "GPU metrics per GPU", subtitle)


def plot_containers(
    rows: pd.DataFrame,
    agent_rows: pd.DataFrame,
    start: int,
    view_start: int,
    view_end: int,
    interval_ns: float,
    out: Path,
    subtitle: str,
):
    average = container_average_across(rows)
    average_s = average.index * interval_ns / 1e9
    count = rows["container_id"].nunique()
    x_limits = ((view_start - start) / 1e9, (view_end - start) / 1e9)

    panel_count = 4 + int(len(agent_rows) > 0)
    fig, axes = panels(panel_count, 3, 3.0, sharex=True)
    cpu_individual_ax, cpu_average_ax, memory_individual_ax, memory_average_ax = axes[:4]
    for color_index, (container_id, container) in enumerate(rows.groupby("container_id")):
        color = GPU_COLORS[color_index % len(GPU_COLORS)]
        time_s = (container["t_mono_ns"] - start) / 1e9
        cpu_individual_ax.plot(
            time_s,
            container["cpu_utilization_percent"],
            color=color,
            lw=0.8,
            alpha=0.85,
            label=container_id[:8],
        )
        memory_individual_ax.plot(
            time_s,
            container["memory_used_mib"],
            color=color,
            lw=0.8,
            alpha=0.85,
            label=container_id[:8],
        )
    cpu_average_ax.plot(
        average_s,
        average["cpu_utilization_percent"],
        color=AVERAGE,
        lw=1.0,
        label="average across containers",
    )
    memory_average_ax.plot(
        average_s,
        average["memory_used_mib"],
        color=AVERAGE,
        lw=1.0,
        label="average across containers",
    )

    label_panel(cpu_individual_ax, "cpu_utilization_percent", TIME_LABEL, LABELS["cpu_utilization_percent"][1])
    label_panel(cpu_average_ax, "cpu_utilization_percent", TIME_LABEL, LABELS["cpu_utilization_percent"][1])
    label_panel(memory_individual_ax, "memory_used_mib", TIME_LABEL, LABELS["memory_used_mib"][1])
    label_panel(memory_average_ax, "memory_used_mib", TIME_LABEL, LABELS["memory_used_mib"][1])
    cpu_individual_ax.set_title(f"{LABELS['cpu_utilization_percent'][0]} — individual containers")
    cpu_average_ax.set_title(f"{LABELS['cpu_utilization_percent'][0]} — average across containers")
    memory_individual_ax.set_title(f"{LABELS['memory_used_mib'][0]} — individual containers")
    memory_average_ax.set_title(f"{LABELS['memory_used_mib'][0]} — average across containers")
    legend_columns = max(1, min(count, 4))
    cpu_individual_ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.12), ncol=legend_columns, title="Container ID"
    )
    cpu_average_ax.legend(loc="lower left", bbox_to_anchor=(0, 1.12))
    memory_individual_ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.12), ncol=legend_columns, title="Container ID"
    )
    memory_average_ax.legend(loc="lower left", bbox_to_anchor=(0, 1.12))

    if len(agent_rows):
        agent_ax = axes[-1]
        agent_s = (agent_rows["t_mono_ns"] - start) / 1e9
        for metric, color, label in [
            ("rss_mib", POOLED, "RSS"),
            ("pss_mib", AVERAGE, "PSS"),
            ("uss_mib", "#009E73", "USS/private"),
        ]:
            agent_ax.plot(agent_s, agent_rows[metric], color=color, lw=1.0, label=label)
        agent_ax.set_title("Agent controller process memory")
        agent_ax.set_xlabel(TIME_LABEL)
        agent_ax.set_ylabel("Memory (MiB)")
        agent_ax.tick_params(labelbottom=True)
        agent_ax.legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=3)
    for ax in axes:
        ax.set_xlim(*x_limits)
    save(fig, out / "container_timeseries.png", "Sandbox and agent-controller metrics over time", subtitle)

    fig, axes = panels(len(CONTAINER_UNITS), 2, 3.0)
    for ax, metric in zip(axes, CONTAINER_UNITS):
        ecdf(ax, rows[metric].to_numpy(float), POOLED, "pooled over containers")
        ecdf(ax, average[metric].to_numpy(float), AVERAGE, "average across containers")
        mark_percentiles(ax)
        label_panel(ax, metric, LABELS[metric][1], CDF_LABEL)
    axes[0].legend(loc="lower right")
    save(fig, out / "container_distributions.png", "Container (sandbox) metric distributions", subtitle)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="run directory or its system_metrics directory")
    parser.add_argument("-o", "--output", type=Path, help="default: <system_metrics>/plots")
    parser.add_argument(
        "--max-staleness",
        type=float,
        default=1.0,
        help="seconds a GPU sample stays valid when averaging across GPUs",
    )
    parser.add_argument(
        "--host-cpus",
        type=int,
        default=os.cpu_count(),
        help="CPUs of the host the run was on (default: this host's)",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="plot START to END seconds after the task analysis window starts (default: complete task window)",
    )
    args = parser.parse_args()

    metrics_dir, meta, start, end = load_run(args.path)
    duration_s = (end - start) / 1e9
    if args.time_range:
        range_start_s, range_end_s = args.time_range
        if not 0 <= range_start_s < range_end_s <= duration_s:
            parser.error(f"--time-range must satisfy 0 <= START < END <= {duration_s:.3f}")
    else:
        range_start_s, range_end_s = 0.0, duration_s
    view_start = start + round(range_start_s * 1e9)
    view_end = start + round(range_end_s * 1e9)
    out = args.output or metrics_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    subtitle = f"{metrics_dir.parent.name}  ·  task window {duration_s:.1f} s"
    if meta["analysis_window"]["clipped_to_logger"]:
        subtitle += "  ·  clipped to logger coverage"
    if args.time_range:
        subtitle += f"  ·  plotted {range_start_s:g}–{range_end_s:g} s"

    gpu_path = metrics_dir / "gpu_samples.csv"
    if meta["gpu_interval_s"] > 0 and gpu_path.exists():
        plot_gpus(
            in_window(pd.read_csv(gpu_path), start, end),
            meta["gpu_ids"],
            start,
            view_start,
            view_end,
            meta["gpu_interval_s"] * 1e9,
            args.max_staleness * 1e9,
            out,
            f"{subtitle}  ·  {len(meta['gpu_ids'])} GPUs, sampled every {meta['gpu_interval_s']} s",
        )

    samples_path = metrics_dir / "container_samples.csv"
    samples = in_window(pd.read_csv(samples_path), start, end) if samples_path.exists() else pd.DataFrame()
    if meta["cpu_interval_s"] > 0 and len(samples):
        interval_ns = meta["cpu_interval_s"] * 1e9
        rows = in_window(container_rows(samples, start, interval_ns, args.host_cpus), view_start, view_end)
        agent_path = metrics_dir / "agent_process_samples.csv"
        agent_samples = in_window(pd.read_csv(agent_path), start, end) if agent_path.exists() else pd.DataFrame()
        agent_rows = (
            in_window(agent_process_rows(agent_samples), view_start, view_end)
            if len(agent_samples)
            else pd.DataFrame()
        )
        plot_containers(
            rows,
            agent_rows,
            start,
            view_start,
            view_end,
            interval_ns,
            out,
            f"{subtitle}  ·  {rows['container_id'].nunique()} container(s), "
            f"sampled every {meta['cpu_interval_s']} s, CPU relative to {args.host_cpus} host CPUs",
        )


if __name__ == "__main__":
    main()

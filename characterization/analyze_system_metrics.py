#!/usr/bin/env python3
"""Analyze system_metrics samples. CPU is % of all host CPUs; memory is MiB.
Writes analysis.json: count/min/max/avg/p50/p99, pooled and average-across views.
The window spans first task setup through last task completion, intersected with
the logger's recorded window. All trajectories must include task_timing.
"""
import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

MIB = 2**20
GPU_METRICS = {
    "gpu_utilization_percent": ("gpu_util_pct", 1, "% of time any kernel was running"),
    "graphics_engine_active_percent": ("gr_engine_active", 100, "% of time graphics/compute engine was busy"),
    "sm_active_percent": ("sm_active", 100, "% of time SMs had a resident warp, averaged over SMs"),
    "power_watts": ("power_w", 1, "W"),
    "power_instant_watts": ("power_instant_w", 1, "W"),
}
CONTAINER_UNITS = {
    "cpu_utilization_percent": "% of whole-host CPU capacity",
    "memory_used_mib": "MiB excluding reclaimable file cache (Docker-compatible for new recordings)",
}
AGENT_PROCESS_UNITS = {
    "rss_mib": "controller resident memory (MiB)",
    "pss_mib": "controller proportional memory (MiB)",
    "uss_mib": "controller private memory (MiB)",
}


def stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {k: 0 if k == "count" else None for k in ("count", "min", "max", "avg", "p50", "p99")}
    return {
        "count": int(len(v)), "min": float(v.min()), "max": float(v.max()), "avg": float(v.mean()),
        "p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99)),
    }


def rounded(obj, digits: int = 4):
    if isinstance(obj, dict):
        return {k: rounded(v, digits) for k, v in obj.items()}
    return round(obj, digits) if isinstance(obj, float) else obj


def on_grid(t: np.ndarray, values: np.ndarray, grid: np.ndarray, max_staleness_ns: float) -> np.ndarray:
    idx = np.searchsorted(t, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    ok = idx >= 0
    ok[ok] = grid[ok] - t[idx[ok]] <= max_staleness_ns
    out[ok] = values[idx[ok]]
    return out


def load_run(path: Path) -> tuple[Path, dict, int, int]:
    metrics_dir = path / "system_metrics" if (path / "system_metrics").is_dir() else path
    meta = json.loads((metrics_dir / "meta.json").read_text())
    if "clock_end" not in meta:
        raise SystemExit(f"{metrics_dir}/meta.json has no clock_end; logger still running or killed")
    trajectories = sorted((metrics_dir.parent / "results").glob("*/*.traj.json"))
    if not trajectories:
        raise SystemExit("Cannot determine task window: no task trajectories found next to system_metrics")
    windows = []
    for trajectory in trajectories:
        timing = json.loads(trajectory.read_text()).get("info", {}).get("task_timing")
        if not timing or not all(key in timing for key in ("start_mono_ns", "end_mono_ns")):
            raise SystemExit(f"Cannot determine task window: {trajectory} lacks task_timing")
        if timing["end_mono_ns"] <= timing["start_mono_ns"]:
            raise SystemExit(f"Invalid task window in {trajectory}")
        windows.append(timing)
    task_start = min(w["start_mono_ns"] for w in windows)
    task_end = max(w["end_mono_ns"] for w in windows)
    start = max(task_start, meta["clock_start"]["monotonic_ns"])
    end = min(task_end, meta.get("clock_stop", meta["clock_end"])["monotonic_ns"])
    if end <= start:
        raise SystemExit("Task window does not overlap the recorded logger window")
    meta = {**meta, "analysis_window": {
        "source": "first task setup to last task completion, intersected with logger coverage",
        "task_count": len(windows), "task_start_mono_ns": task_start, "task_end_mono_ns": task_end,
        "start_mono_ns": start, "end_mono_ns": end,
        "clipped_to_logger": start != task_start or end != task_end,
    }}
    return metrics_dir, meta, start, end


def in_window(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df[(df["t_mono_ns"] >= start) & (df["t_mono_ns"] <= end)]


def gpu_metric(gpu: pd.DataFrame, metric: str) -> pd.DataFrame:
    field, scale, _ = GPU_METRICS[metric]
    s = gpu[gpu["field"] == field]
    return s[["t_mono_ns", "gpu"]].assign(value=s["value"] * scale)


def gpu_average_across(s: pd.DataFrame, gpu_ids: list, grid: np.ndarray, staleness_ns: float) -> np.ndarray:
    if s["gpu"].nunique() != len(gpu_ids):
        return np.full(len(grid), np.nan)
    aligned = np.vstack([
        on_grid(g["t_mono_ns"].to_numpy(float), g["value"].to_numpy(float), grid, staleness_ns)
        for _, g in s.groupby("gpu")
    ])
    return np.where(np.isnan(aligned).any(axis=0), np.nan, aligned.mean(axis=0))


def analyze_gpus(gpu: pd.DataFrame, gpu_ids: list, grid: np.ndarray, staleness_ns: float) -> dict:
    pooled, average = {}, {}
    for metric in GPU_METRICS:
        s = gpu_metric(gpu, metric)
        pooled[metric] = stats(s["value"])
        average[metric] = stats(gpu_average_across(s, gpu_ids, grid, staleness_ns))
    return {"pooled_over_gpus": pooled, "average_across_gpus": average}


def container_rates(c: pd.DataFrame, host_cpus: int) -> pd.DataFrame:
    c = c.sort_values("t_mono_ns")
    cache = c["mem_total_inactive_file_bytes"] if "mem_total_inactive_file_bytes" in c else c["mem_inactive_file_bytes"]
    memory_used = c["mem_usage_bytes"] - cache.where(cache < c["mem_usage_bytes"], 0)
    return pd.DataFrame({
        "t_mono_ns": c["t_mono_ns"], "container_id": c["container_id"],
        "cpu_utilization_percent": c["cpu_usage_ns"].diff() / c["t_mono_ns"].diff() / host_cpus * 100,
        "memory_used_mib": memory_used / MIB,
    })


def container_rows(samples: pd.DataFrame, start: int, interval_ns: float, host_cpus: int) -> pd.DataFrame:
    rows = pd.concat([container_rates(c, host_cpus) for _, c in samples.groupby("container_id")])
    return rows.assign(tick=np.round((rows["t_mono_ns"] - start) / interval_ns).astype(int))


def container_average_across(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby("tick")[list(CONTAINER_UNITS)].mean()


def analyze_containers(rows: pd.DataFrame) -> dict:
    per_tick = container_average_across(rows)
    return {
        "pooled_over_containers": {metric: stats(rows[metric]) for metric in CONTAINER_UNITS},
        "average_across_containers": {metric: stats(per_tick[metric]) for metric in CONTAINER_UNITS},
    }


def agent_process_rows(samples: pd.DataFrame) -> pd.DataFrame:
    return samples[["t_mono_ns", "pid"]].assign(
        rss_mib=samples["rss_bytes"] / MIB, pss_mib=samples["pss_bytes"] / MIB, uss_mib=samples["uss_bytes"] / MIB,
    )


def analyze_agent_process(rows: pd.DataFrame) -> dict:
    return {metric: stats(rows[metric]) for metric in AGENT_PROCESS_UNITS}


def print_summary(result: dict):
    print(f"Window: {result['window']['duration_s']:.1f} s from {result['window']['start_utc']}")
    for section, views in [
        ("gpu", ["pooled_over_gpus", "average_across_gpus"]),
        ("container", ["pooled_over_containers", "average_across_containers"]),
        ("agent_process", ["memory"]),
    ]:
        for view in views if section in result else []:
            print(f"\n{section} / {view}:")
            print(f"  {'metric':32s}{'count':>8s}" + "".join(f"{k:>11s}" for k in ["min", "avg", "p50", "p99", "max"]))
            for metric, s in result[section][view].items():
                values = "".join(f"{s[k]:11.3f}" for k in ["min", "avg", "p50", "p99", "max"]) if s["count"] else ""
                print(f"  {metric:32s}{s['count']:8d}{values}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--max-staleness", type=float, default=1.0)
    parser.add_argument("--host-cpus", type=int, default=os.cpu_count())
    args = parser.parse_args()
    metrics_dir, meta, start, end = load_run(args.path)
    offset = meta["clock_start"]["realtime_ns"] - meta["clock_start"]["monotonic_ns"]

    def utc(t):
        return datetime.datetime.fromtimestamp((t + offset) / 1e9, datetime.timezone.utc).isoformat()

    result = {
        "run": str(metrics_dir.resolve()), "hostname": meta.get("hostname"),
        "window": {**meta["analysis_window"], "start_utc": utc(start), "end_utc": utc(end), "duration_s": (end-start)/1e9},
        "statistics": "count, min, max, avg, p50, p99",
    }
    gpu_path = metrics_dir / "gpu_samples.csv"
    if meta["gpu_interval_s"] > 0 and gpu_path.exists():
        grid = np.arange(start, end, meta["gpu_interval_s"] * 1e9)
        result["gpu"] = {
            "sampling_interval_s": meta["gpu_interval_s"], "gpu_count": len(meta["gpu_ids"]),
            "units": {metric: unit for metric, (_, _, unit) in GPU_METRICS.items()},
        } | analyze_gpus(in_window(pd.read_csv(gpu_path), start, end), meta["gpu_ids"], grid, args.max_staleness * 1e9)
    sample_path = metrics_dir / "container_samples.csv"
    samples = in_window(pd.read_csv(sample_path), start, end) if sample_path.exists() else pd.DataFrame()
    if meta["cpu_interval_s"] > 0 and len(samples):
        result["container"] = {
            "sampling_interval_s": meta["cpu_interval_s"], "container_count": samples["container_id"].nunique(),
            "host_cpus": args.host_cpus, "units": CONTAINER_UNITS,
        } | analyze_containers(container_rows(samples, start, meta["cpu_interval_s"] * 1e9, args.host_cpus))
    agent_path = metrics_dir / "agent_process_samples.csv"
    agent_samples = in_window(pd.read_csv(agent_path), start, end) if agent_path.exists() else pd.DataFrame()
    if meta["cpu_interval_s"] > 0 and len(agent_samples):
        rows = agent_process_rows(agent_samples)
        result["agent_process"] = {
            "sampling_interval_s": meta["cpu_interval_s"], "pid": int(rows["pid"].iloc[0]),
            "units": AGENT_PROCESS_UNITS, "memory": analyze_agent_process(rows),
        }
    output = args.output or metrics_dir / "analysis.json"
    output.write_text(json.dumps(rounded(result), indent=2))
    print_summary(result)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()

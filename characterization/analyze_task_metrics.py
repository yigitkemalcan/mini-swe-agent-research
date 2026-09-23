#!/usr/bin/env python3
"""Summarize task resources and step timings: python analyze_task_metrics.py RUN_DIR.

Writes task_metrics/{tasks.csv,steps.csv,analysis.json}. CPU is % of whole-host
capacity. Memory is Docker-compatible cache-adjusted MiB. Averages are time
weighted over covered intervals within each task, including environment setup
but excluding container cleanup after the task. Peaks are sampled peaks, not
guaranteed instantaneous maxima. Missing measurements remain null, never zero.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer


def resource_summary(samples: pd.DataFrame, start: int, end: int, host_cpus: int) -> dict:
    result = {
        "resource_status": "insufficient_samples", "sample_count": len(samples),
        "resource_coverage_s": 0.0, "cpu_avg_percent": None, "cpu_peak_percent": None,
        "memory_avg_mib": None, "memory_peak_mib": None,
    }
    c = samples.sort_values("t_mono_ns")
    if len(c) < 2:
        return result
    times = c.t_mono_ns.to_numpy(dtype=np.int64)
    widths = np.maximum(0, np.minimum(times[1:], end) - np.maximum(times[:-1], start)) / 1e9
    valid = widths > 0
    if not valid.any():
        result["resource_status"] = "no_overlap"
        return result
    cpu_deltas = np.diff(c.cpu_usage_ns.to_numpy(dtype=np.int64))
    if (cpu_deltas[valid] < 0).any():
        raise ValueError("Container CPU counter decreased within a task")
    rates = cpu_deltas[valid] / np.diff(times)[valid] * 100 / host_cpus
    cache_column = "mem_total_inactive_file_bytes" if "mem_total_inactive_file_bytes" in c else "mem_inactive_file_bytes"
    cache = c[cache_column]
    memory = ((c.mem_usage_bytes - cache.where(cache < c.mem_usage_bytes, 0)) / 2**20).to_numpy()
    inside = (times >= start) & (times <= end)
    result.update({
        "resource_status": "ok" if cache_column == "mem_total_inactive_file_bytes" else "legacy_cache_definition",
        "sample_count": int(inside.sum()),
        "resource_coverage_s": float(widths.sum()),
        "cpu_avg_percent": float(np.average(rates, weights=widths[valid])),
        "cpu_peak_percent": float(rates.max()),
        "memory_avg_mib": float(np.average(memory[:-1][valid], weights=widths[valid])),
        "memory_peak_mib": float(np.max(np.r_[memory[inside], memory[:-1][valid]])),
    })
    return result


def step_rows(trajectory: dict, task_id: str) -> list[dict]:
    return [
        {
            "task_id": task_id, "step": timing["step"],
            "inference_s": timing["inference"]["elapsed_s"],
            "tool_s": sum(tool["elapsed_s"] for tool in timing["tools"]),
            "action_count": len(timing["tools"]),
            "inference_outcome": timing["inference"]["outcome"],
            "tool_outcomes": ";".join(tool["outcome"] for tool in timing["tools"]),
            "start_mono_ns": timing["inference"]["start_mono_ns"],
            "end_mono_ns": (timing["tools"] or [timing["inference"]])[-1]["end_mono_ns"],
        }
        for timing in trajectory.get("step_timings", [])
    ]


def analyze(run_dir: Path, host_cpus: int) -> tuple[list[dict], list[dict]]:
    samples = pd.read_csv(run_dir / "system_metrics/container_samples.csv")
    groups = {cid: group for cid, group in samples.groupby("container_id")}
    tasks, steps = [], []
    for path in sorted((run_dir / "results").glob("*/*.traj.json")):
        trajectory = json.loads(path.read_text())
        info = trajectory["info"]
        task_id = trajectory.get("instance_id", path.parent.name)
        window = info.get("task_timing")
        task_steps = step_rows(trajectory, task_id)
        count = info["model_stats"]["api_calls"]
        record = {
            "task_id": task_id, "container_id": info.get("container_id"),
            "exit_status": info.get("exit_status"), "steps": count,
            "timed_steps": len(task_steps), "timing_complete": len(task_steps) == count,
            "task_duration_s": None, "environment_setup_s": None,
            "inference_total_s": sum(s["inference_s"] for s in task_steps) if task_steps else None,
            "tool_total_s": sum(s["tool_s"] for s in task_steps) if task_steps else None,
            "resource_status": "missing_task_window_or_container_id",
            "sample_count": 0, "resource_coverage_s": 0.0,
            "cpu_avg_percent": None, "cpu_peak_percent": None,
            "memory_avg_mib": None, "memory_peak_mib": None,
        }
        if window:
            start, end = window["start_mono_ns"], window["end_mono_ns"]
            record["task_duration_s"] = (end-start) / 1e9
            if window.get("environment_ready_mono_ns") is not None:
                record["environment_setup_s"] = (window["environment_ready_mono_ns"]-start) / 1e9
            if info.get("container_id"):
                record.update(resource_summary(groups.get(info["container_id"], samples.iloc[:0]), start, end, host_cpus))
        tasks.append(record)
        steps.extend(task_steps)
    return tasks, steps


def main(run_dir: Path, host_cpus: int | None = typer.Option(None, min=1)):
    """Analyze a run, matching containers to tasks by saved full container ID."""
    if host_cpus is None:
        host_cpus = json.loads((run_dir / "system_metrics/meta.json").read_text())["host_cpus"]
    tasks, steps = analyze(run_dir, host_cpus)
    if not tasks:
        raise typer.BadParameter("No task trajectories found in results/*/*.traj.json")
    output = run_dir / "task_metrics"
    output.mkdir(exist_ok=True)
    pd.DataFrame(tasks).to_csv(output / "tasks.csv", index=False)
    pd.DataFrame(steps, columns=[
        "task_id", "step", "inference_s", "tool_s", "action_count", "inference_outcome", "tool_outcomes",
        "start_mono_ns", "end_mono_ns",
    ]).to_csv(output / "steps.csv", index=False)
    meta = json.loads((run_dir / "system_metrics/meta.json").read_text())
    (output / "analysis.json").write_text(json.dumps({
        "host_cpus": host_cpus, "definitions": __doc__, "system_logger_errors": meta.get("errors", []),
        "tasks": tasks, "steps": steps,
    }, indent=2) + "\n")
    typer.echo(f"Wrote {len(tasks)} tasks and {len(steps)} timed steps to {output}")


if __name__ == "__main__":
    typer.run(main)

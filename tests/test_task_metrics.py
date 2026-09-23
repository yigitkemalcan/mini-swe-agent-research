import json

import pandas as pd
import pytest
from typer.testing import CliRunner
import typer

from characterization.analyze_task_metrics import analyze, main, resource_summary


def test_resource_window_weighting_cache_guard_and_missing_data():
    samples = pd.DataFrame({
        "t_mono_ns": [0, 1_000_000_000, 3_000_000_000, 4_000_000_000],
        "cpu_usage_ns": [0, 1_000_000_000, 5_000_000_000, 5_000_000_000],
        "mem_usage_bytes": [v * 2**20 for v in [100, 200, 300, 999]],
        "mem_total_inactive_file_bytes": [v * 2**20 for v in [10, 40, 400, 0]],
    })
    result = resource_summary(samples, 500_000_000, 3_500_000_000, 4)
    assert result["cpu_avg_percent"] == pytest.approx(37.5)
    assert result["cpu_peak_percent"] == 50
    assert result["memory_avg_mib"] == pytest.approx((90*.5 + 160*2 + 300*.5) / 3)
    assert result["memory_peak_mib"] == 300
    assert result["resource_coverage_s"] == 3
    assert result["sample_count"] == 2
    assert resource_summary(samples.iloc[:0], 0, 1, 4)["cpu_avg_percent"] is None
    assert resource_summary(samples, 5_000_000_000, 6_000_000_000, 4)["resource_status"] == "no_overlap"


def test_exact_task_container_join_and_legacy_unavailable(tmp_path):
    (tmp_path / "system_metrics").mkdir()
    pd.DataFrame({
        "container_id": ["wanted", "other", "wanted", "other"],
        "t_mono_ns": [0, 0, 1_000_000_000, 1_000_000_000],
        "cpu_usage_ns": [0, 0, 1_000_000_000, 4_000_000_000],
        "mem_usage_bytes": [2**20, 99*2**20, 2**20, 99*2**20],
        "mem_total_inactive_file_bytes": [0, 0, 0, 0],
    }).to_csv(tmp_path / "system_metrics/container_samples.csv", index=False)
    for name, info in [("new", {
        "container_id": "wanted",
        "task_timing": {"start_mono_ns": 0, "environment_ready_mono_ns": 100_000_000, "end_mono_ns": 1_000_000_000},
    }), ("old", {})]:
        folder = tmp_path / "results" / name
        folder.mkdir(parents=True)
        (folder / f"{name}.traj.json").write_text(json.dumps({
            "instance_id": name, "info": {"model_stats": {"api_calls": 2}, **info},
        }))
    tasks, steps = analyze(tmp_path, 4)
    assert tasks[0]["cpu_avg_percent"] == 25
    assert tasks[0]["memory_avg_mib"] == 1
    assert tasks[0]["environment_setup_s"] == .1
    assert tasks[1]["cpu_avg_percent"] is None
    assert tasks[1]["steps"] == 2 and not tasks[1]["timing_complete"]
    assert steps == []
    (tmp_path / "system_metrics/meta.json").write_text(json.dumps({"host_cpus": 4, "errors": []}))
    app = typer.Typer()
    app.command()(main)
    result = CliRunner().invoke(app, [str(tmp_path)])
    assert result.exit_code == 0, result.output
    exported = json.loads((tmp_path / "task_metrics/analysis.json").read_text())
    assert exported["host_cpus"] == 4
    assert exported["tasks"][0]["cpu_avg_percent"] == 25
    assert (tmp_path / "task_metrics/tasks.csv").exists()
    assert "inference_s" in (tmp_path / "task_metrics/steps.csv").read_text()

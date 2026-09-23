import json

import pytest

from characterization.analyze_system_metrics import load_run


def test_task_envelope_and_logger_intersection(tmp_path):
    metrics = tmp_path / "system_metrics"
    metrics.mkdir()
    meta = {"clock_start": {"monotonic_ns": 10}, "clock_stop": {"monotonic_ns": 90},
            "clock_end": {"monotonic_ns": 100}}
    (metrics / "meta.json").write_text(json.dumps(meta))
    for name, start, end in [("a", 30, 60), ("b", 20, 70)]:
        folder = tmp_path / "results" / name
        folder.mkdir(parents=True)
        (folder / f"{name}.traj.json").write_text(json.dumps({
            "info": {"task_timing": {"start_mono_ns": start, "end_mono_ns": end}},
        }))
    assert load_run(tmp_path)[2:] == (20, 70)
    assert load_run(metrics)[2:] == (20, 70)
    assert not load_run(tmp_path)[1]["analysis_window"]["clipped_to_logger"]
    meta["clock_start"]["monotonic_ns"] = 25
    meta["clock_stop"]["monotonic_ns"] = 65
    (metrics / "meta.json").write_text(json.dumps(meta))
    assert load_run(tmp_path)[2:] == (25, 65)
    assert load_run(tmp_path)[1]["analysis_window"]["clipped_to_logger"]
    assert json.loads((metrics / "meta.json").read_text()) == meta
    (tmp_path / "results/b/b.traj.json").write_text('{"info": {}}')
    with pytest.raises(SystemExit, match="lacks task_timing"):
        load_run(tmp_path)


def test_no_trajectories_does_not_silently_use_logger_window(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({
        "clock_start": {"monotonic_ns": 0}, "clock_end": {"monotonic_ns": 100},
    }))
    with pytest.raises(SystemExit, match="no task trajectories"):
        load_run(tmp_path)

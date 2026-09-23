#!/usr/bin/env python3
"""Docker/cgroup comparison recovered from the former /tmp comparison script.
Run: python characterization/compare_docker_metrics.py RUN_DIR [--window-seconds 10]
Writes docker_comparison/{per_container.csv,aligned_windows.csv,comparison.json,comparison.png}.
Raw recordings are unchanged. Docker CPU is normalized by the recorded host CPU count.
Receipt times approximate measurement times. Window plots smooth peaks; peak fields
use the original full recordings, and full-overlap means do NOT average plot bins.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40}


def memory(value: str) -> float:
    match = re.fullmatch(r"([\d.]+)(\w+)", value.split(" / ")[0])
    if match is None or match[2] not in UNITS:
        raise ValueError(f"Unrecognized Docker memory value: {value}")
    return float(match[1]) * UNITS[match[2]] / 2**20


def held_mean(t, v, a, b) -> float:
    knots = np.r_[a, t[(t > a) & (t < b)], b]
    idx = np.searchsorted(t, knots[:-1], side="right") - 1
    if (idx < 0).any() or b <= a:
        raise ValueError("Requested mean outside covered interval")
    return float(np.dot(np.diff(knots), v[idx]) / (b-a))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.window_seconds <= 0:
        parser.error("window-seconds must be positive")
    run = args.run_dir
    meta = json.loads((run/"system_metrics/meta.json").read_text())
    dm = json.loads((run/"docker_stats/meta.json").read_text())
    cpus = json.loads((run/"docker-metadata.json").read_text())["daemon"]["info"]["NCPU"]
    if meta.get("hostname") != dm.get("hostname"):
        raise ValueError("Monotonic clocks must refer to the same host")
    origin = meta["clock_start"]["monotonic_ns"]
    start = max(0, (dm["clock_start"]["monotonic_ns"]-origin)/1e9)
    stop = (min(meta["clock_stop"]["monotonic_ns"], dm["clock_stop"]["monotonic_ns"])-origin)/1e9
    c = pd.read_csv(run/"system_metrics/container_samples.csv")
    d = pd.read_json(run/"docker_stats/docker_stats.jsonl", lines=True)
    receipts = pd.read_csv(run/"docker_stats/receipt_times.csv")
    if not len(d) or not len(c):
        raise SystemExit("No Docker/cgroup samples to compare")
    if len(d) != len(receipts) or len(d) != dm["rows"] or not (d.ID == receipts.container_id).all():
        raise ValueError("Docker rows and receipt index disagree")
    d["t"] = (receipts.t_mono_ns-origin)/1e9
    c["t"] = (c.t_mono_ns-origin)/1e9
    d["cpu"] = d.CPUPerc.str.rstrip("%").astype(float)/cpus
    d["mem"] = d.MemUsage.map(memory)
    cache_col = "mem_total_inactive_file_bytes" if "mem_total_inactive_file_bytes" in c else "mem_inactive_file_bytes"
    cache = c[cache_col]
    c["mem"] = (c.mem_usage_bytes-cache.where(cache < c.mem_usage_bytes, 0))/2**20
    matches = [cid for cid in c.container_id.unique() if cid in set(d.ID)]
    if not matches:
        raise SystemExit("No matching containers")
    fig, axes = plt.subplots(len(matches), 2, figsize=(14, 3*len(matches)), squeeze=False)
    events = pd.read_csv(run/"system_metrics/container_events.csv")
    rows, windows = [], []
    for i, cid in enumerate(matches):
        cg = c[c.container_id == cid].sort_values("t")
        dg_raw = d[d.ID == cid].sort_values("t")
        # Same-time duplicate CLI rows have no extra weight.
        dg = dg_raw.drop_duplicates("t", keep="last")
        if len(cg) < 2 or len(dg) < 2:
            continue
        ct, dt = cg.t.to_numpy(), dg.t.to_numpy()
        a, b = max(start, ct[0], dt[0]), min(stop, ct[-1], dt[-1])
        if b <= a:
            continue
        cpu_counter = cg.cpu_usage_ns.to_numpy()/1e9

        def cpu_mean(left, right):
            return float((np.interp(right, ct, cpu_counter)-np.interp(left, ct, cpu_counter))/(right-left)*100/cpus)

        name = dg.Name.iloc[0]
        image_rows = events.loc[events.container_id == cid, "image"]
        record = {
            "name": name, "container_id": cid, "image": image_rows.iloc[0] if len(image_rows) else "",
            "overlap_s": b-a, "cgroup_samples": len(cg), "docker_rows": len(dg_raw),
            "docker_unique_receipt_times": len(dg),
            "docker_median_receipt_interval_s": float(np.median(np.diff(dt))),
            "cgroup_mean_cpu_host_pct": cpu_mean(a,b),
            "docker_mean_cpu_host_pct": held_mean(dt,dg.cpu.to_numpy(),a,b),
            "cgroup_mean_memory_mib": held_mean(ct,cg.mem.to_numpy(),a,b),
            "docker_mean_memory_mib": held_mean(dt,dg.mem.to_numpy(),a,b),
            "cgroup_peak_cpu_host_pct": float((np.diff(cpu_counter)/np.diff(ct)*100/cpus).max()),
            "docker_peak_cpu_host_pct": float(dg.cpu.max()),
            "cgroup_peak_memory_mib": float(cg.mem.max()), "docker_peak_memory_mib": float(dg.mem.max()),
        }
        for left in np.arange(a,b-args.window_seconds+1e-9,args.window_seconds):
            right = left+args.window_seconds
            windows.append({
                "name": name, "start_s": left, "end_s": right,
                "cgroup_cpu_host_pct": cpu_mean(left,right),
                "docker_cpu_host_pct": held_mean(dt,dg.cpu.to_numpy(),left,right),
                "cgroup_memory_mib": held_mean(ct,cg.mem.to_numpy(),left,right),
                "docker_memory_mib": held_mean(dt,dg.mem.to_numpy(),left,right),
            })
        own = pd.DataFrame([w for w in windows if w["name"] == name])
        for j, metric in enumerate(["cpu_host_pct","memory_mib"]):
            ax = axes[i,j]
            if len(own):
                ax.plot(own.start_s-a+args.window_seconds/2, own["cgroup_"+metric], label="cgroup")
                ax.plot(own.start_s-a+args.window_seconds/2, own["docker_"+metric], label="Docker", linestyle="--")
                record[metric+"_window_mean_absolute_difference"] = float((own["cgroup_"+metric]-own["docker_"+metric]).abs().mean())
                ax.legend()
            else:
                ax.text(.5,.5,"Overlap shorter than plotting window",ha="center",transform=ax.transAxes)
            ax.set_title(name)
            ax.set_ylabel("CPU (% of host)" if j == 0 else "Memory (MiB)")
            ax.set_xlabel("Seconds from common container window start")
            ax.grid(alpha=.2)
        rows.append(record)
    if not rows:
        raise SystemExit("No overlapping container measurements")
    out = run/"docker_comparison"
    out.mkdir(exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out/"per_container.csv",index=False)
    pd.DataFrame(windows).to_csv(out/"aligned_windows.csv",index=False)
    fig.suptitle(f"Docker vs cgroup: {args.window_seconds:g}-second windows; CPU normalized by {cpus} CPUs")
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(out/"comparison.png",dpi=150)
    plt.close(fig)
    pooled = {key: float(np.average(summary[key],weights=summary.overlap_s)) for key in [
        "cgroup_mean_cpu_host_pct","docker_mean_cpu_host_pct","cgroup_mean_memory_mib","docker_mean_memory_mib",
    ]}
    (out/"comparison.json").write_text(json.dumps({
        "run": str(run.resolve()), "host_cpus": cpus, "method": __doc__,
        "window_seconds": args.window_seconds, "cache_column": cache_col,
        "logger_errors": meta.get("errors",[]), "pooled_time_weighted": pooled, "containers": rows,
    },indent=2)+"\n")
    print(summary.to_string(index=False))
    print(json.dumps(pooled,indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

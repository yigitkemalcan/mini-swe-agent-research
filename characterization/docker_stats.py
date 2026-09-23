#!/usr/bin/env python3
"""Record Docker CLI stats unchanged, with separate receipt timestamps.

Usage: python characterization/docker_stats.py NEW_OUTPUT_DIR [--duration 60]
Writes docker_stats.jsonl, receipt_times.csv, meta.json and a ready marker.
Default names match ^minisweagent-. Use --container-name-regex '.*' for all.
Docker 28.1.1 refreshes CLI output every 500 ms; measurements can repeat.
Receipt time is not the underlying measurement time. Only ANSI escape sequences
and surrounding whitespace are removed; numeric values and units are unchanged.
Docker CPU uses 100% per busy logical CPU; divide by host CPU count to compare
against analyze_system_metrics.py. MemUsage is cache-adjusted, not process RSS.
"""
import csv
import json
import os
import re
import selectors
import signal
import socket
import subprocess
import time
from pathlib import Path

import typer

ANSI = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")


def clock_pair() -> dict[str, int]:
    return {"monotonic_ns": time.monotonic_ns(), "realtime_ns": time.time_ns()}


def main(
    output_dir: Path,
    duration: float = typer.Option(0.0, min=0.0, help="Recording seconds; 0 waits for SIGINT/SIGTERM."),
    container_name_regex: str = typer.Option(r"^minisweagent-", help="Matching container names."),
    docker_executable: str = typer.Option("docker", help="Docker executable; honors its context/environment."),
):
    name_re = re.compile(container_name_regex)
    command = [docker_executable, "stats", "--no-trunc", "--format", "{{json .}}"]
    output_dir.mkdir(parents=True, exist_ok=False)
    meta = {
        "command": command, "container_name_regex": container_name_regex,
        "hostname": socket.gethostname(), "clock_start": clock_pair(),
        "requested_duration_s": duration, "timestamp_meaning": "local receipt time, not Docker measurement time",
        "cadence": "Docker-controlled; repeated values are preserved",
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    stop_clock: dict = {}

    def request_stop(*_):
        if not stop_clock:
            stop_clock.update(clock_pair())

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    rows = 0
    process = None
    completed = False
    try:
        with (
            (output_dir / "docker_stats.jsonl").open("wb", buffering=0) as raw,
            (output_dir / "receipt_times.csv").open("w", newline="", buffering=1) as timings,
            selectors.DefaultSelector() as selector,
        ):
            writer = csv.writer(timings)
            writer.writerow(["jsonl_line", "t_mono_ns", "t_realtime_ns", "container_id", "name"])
            process = subprocess.Popen(command, stdout=subprocess.PIPE, start_new_session=True)
            selector.register(process.stdout, selectors.EVENT_READ)
            pending = b""
            deadline = time.monotonic() + duration if duration else float("inf")
            typer.echo(f"Recording Docker's native stream to {output_dir}; Ctrl-C to stop.")
            while not stop_clock:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    request_stop()
                    break
                if not selector.select(timeout=min(0.2, remaining)):
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                received = clock_pair()
                if not chunk:
                    break
                if not (output_dir / "ready").exists():
                    (output_dir / "ready").touch()
                pending += chunk
                lines = pending.split(b"\n")
                pending = lines.pop()
                for line in lines:
                    line = ANSI.sub(b"", line).strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not name_re.search(record["Name"]):
                        continue
                    raw.write(line + b"\n")
                    rows += 1
                    writer.writerow([
                        rows, received["monotonic_ns"], received["realtime_ns"], record["ID"], record["Name"],
                    ])
            completed = True
    finally:
        requested_stop = bool(stop_clock)
        if not stop_clock:
            stop_clock.update(clock_pair())
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            process.stdout.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        meta.update({
            "clock_stop": stop_clock, "clock_end": clock_pair(), "rows": rows,
            "docker_exit_code": process.returncode if process is not None else None,
            "completed": completed, "stop_requested": requested_stop,
        })
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    if not requested_stop and process.returncode:
        raise typer.Exit(process.returncode)
    typer.echo(f"Saved {rows} Docker stats rows to {output_dir}.")


if __name__ == "__main__":
    typer.run(main)

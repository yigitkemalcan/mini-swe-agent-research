#!/usr/bin/env python3
"""Raw system metrics for mini-SWE-agent/vLLM experiments.

Outputs gpu_samples.csv, container_samples.csv, container_events.csv,
agent_process_samples.csv and meta.json. All t_mono_ns use CLOCK_MONOTONIC.
GPU timestamps originate in DCGM realtime and are converted to monotonic.
CPU counters are cumulative nanoseconds, memory is bytes. No aggregation here.
This collector targets the original host's hybrid cgroup v1/v2 layout.
"""

import argparse
import csv
import http.client
import json
import os
import re
import signal
import socket
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path

DCGM_BINDINGS = "/usr/share/datacenter-gpu-manager-4/bindings/python3"
CGROUP = Path("/sys/fs/cgroup")
DOCKER_CGROUP_DIR = CGROUP / "cpuacct/docker"
DOCKER_SOCK = "/var/run/docker.sock"
CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")
GPU_FIELDS = {
    203: "gpu_util_pct", 1001: "gr_engine_active", 1002: "sm_active",
    155: "power_w", 157: "power_instant_w",
}
CONTAINER_FILES = {
    "cpu_usage": "cpuacct/docker/{id}/cpuacct.usage",
    "cpu_user": "cpuacct/docker/{id}/cpuacct.usage_user",
    "cpu_sys": "cpuacct/docker/{id}/cpuacct.usage_sys",
    "mem_usage": "memory/docker/{id}/memory.usage_in_bytes",
    "mem_stat": "memory/docker/{id}/memory.stat",
    "io_pressure": "unified/docker/{id}/io.pressure",
}
CONTAINER_COLUMNS = {
    "t_mono_ns": "CLOCK_MONOTONIC ns before reading container files",
    "container_id": "full Docker container ID",
    "cpu_usage_ns": "cpuacct.usage: cumulative CPU time (user+system), ns",
    "cpu_user_ns": "cpuacct.usage_user: cumulative user CPU time, ns",
    "cpu_sys_ns": "cpuacct.usage_sys: cumulative system CPU time, ns",
    "mem_usage_bytes": "memory.usage_in_bytes (includes page cache)",
    "mem_rss_bytes": "memory.stat rss (anonymous memory)",
    "mem_cache_bytes": "memory.stat cache (page cache)",
    "mem_inactive_file_bytes": "memory.stat inactive_file (local cgroup only)",
    "mem_total_inactive_file_bytes": "memory.stat total_inactive_file (Docker cgroup v1 cache subtraction)",
    "io_some_total_us": "io.pressure some: cumulative stall microseconds",
    "io_full_total_us": "io.pressure full: cumulative stall microseconds",
}
AGENT_PROCESS_COLUMNS = {
    "t_mono_ns": "CLOCK_MONOTONIC ns before reading smaps_rollup",
    "pid": "host PID of agent controller",
    "rss_bytes": "resident memory, shared pages counted in full",
    "pss_bytes": "proportional memory, shared pages divided among processes",
    "uss_bytes": "private clean, dirty and hugetlb pages",
}


class DockerSocket(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("localhost", timeout=5)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(DOCKER_SOCK)


def inspect_container(container_id: str) -> dict | None:
    conn = DockerSocket()
    conn.request("GET", f"/containers/{container_id}/json")
    response = conn.getresponse()
    body = response.read()
    conn.close()
    if response.status != 200:
        return None
    info = json.loads(body)
    return {"name": info["Name"].lstrip("/"), "image": info["Config"]["Image"]}


class Container:
    """Keep cgroup descriptors open and sample with pread."""

    def __init__(self, container_id: str):
        self.id, self.name, self.image = container_id, "", ""
        self.discovered_ns, self.stopped_ns = time.monotonic_ns(), None
        self.pending: list[list] | None = []
        self.fds = {}
        try:
            for key, path in CONTAINER_FILES.items():
                self.fds[key] = os.open(CGROUP / path.format(id=container_id), os.O_RDONLY)
        except OSError:
            self.close()
            raise

    def read(self, key: str) -> bytes:
        return os.pread(self.fds[key], 16384, 0)

    def sample(self) -> list:
        t = time.monotonic_ns()
        mem_stat = dict(line.split() for line in self.read("mem_stat").decode().splitlines())
        io = {line.split()[0]: line.rsplit("total=", 1)[1] for line in self.read("io_pressure").decode().splitlines()}
        return [
            t, self.id, int(self.read("cpu_usage")), int(self.read("cpu_user")), int(self.read("cpu_sys")),
            int(self.read("mem_usage")), int(mem_stat["rss"]), int(mem_stat["cache"]),
            int(mem_stat["inactive_file"]), int(mem_stat["total_inactive_file"]),
            int(io["some"]), int(io["full"]),
        ]

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds = {}


class ContainerSampler:
    def __init__(self, output_dir: Path, interval: float, name_regex: str):
        self.output_dir, self.interval, self.name_re = output_dir, interval, re.compile(name_regex)
        self.missed_ticks = 0

    def run(self, stop: threading.Event):
        # Inspect may block while a container starts; buffer its samples until resolved.
        tracked: dict[str, Container] = {}
        lookups: dict[str, Future] = {}
        ignored: set[str] = set()
        with (
            ThreadPoolExecutor(max_workers=4) as resolver,
            (self.output_dir / "container_samples.csv").open("w", newline="", buffering=1 << 20) as samples_file,
            (self.output_dir / "container_events.csv").open("w", newline="") as events_file,
        ):
            samples, events = csv.writer(samples_file), csv.writer(events_file)
            samples.writerow(CONTAINER_COLUMNS)
            events.writerow(["t_mono_ns", "event", "container_id", "name", "image"])
            next_tick = last_flush = time.monotonic()
            while not stop.is_set():
                try:
                    present = {d for d in os.listdir(DOCKER_CGROUP_DIR) if CONTAINER_ID_RE.fullmatch(d)}
                except FileNotFoundError:  # cgroup dir appears only once a container has run
                    present = set()
                for container_id in present - tracked.keys() - ignored:
                    try:
                        tracked[container_id] = Container(container_id)
                    except OSError:
                        continue
                    lookups[container_id] = resolver.submit(inspect_container, container_id)
                ignored &= present
                for container in tracked.values():
                    if container.stopped_ns is not None:
                        continue
                    try:
                        row = container.sample()
                    except (OSError, ValueError, KeyError):
                        container.close()
                        container.stopped_ns = time.monotonic_ns()
                        continue
                    if container.pending is None:
                        samples.writerow(row)
                    else:
                        container.pending.append(row)
                self._resolve(tracked, lookups, ignored, samples, events)
                now = time.monotonic()
                if now - last_flush >= 1.0:
                    samples_file.flush()
                    events_file.flush()
                    last_flush = now
                next_tick += self.interval
                if next_tick < now:
                    self.missed_ticks += int((now - next_tick) / self.interval) + 1
                    next_tick = now
                time.sleep(next_tick - now)
            wait(lookups.values())
            self._resolve(tracked, lookups, ignored, samples, events)
            for container in tracked.values():
                container.close()

    def _resolve(self, tracked: dict, lookups: dict, ignored: set, samples, events):
        for container_id, container in list(tracked.items()):
            if container_id in lookups and lookups[container_id].done():
                info = lookups.pop(container_id).result()
                if info is None or not self.name_re.search(info["name"]):
                    container.close()
                    del tracked[container_id]
                    ignored.add(container_id)
                    continue
                container.name, container.image = info["name"], info["image"]
                events.writerow([container.discovered_ns, "start", container_id, container.name, container.image])
                samples.writerows(container.pending)
                container.pending = None
            if container.pending is None and container.stopped_ns is not None:
                events.writerow([container.stopped_ns, "stop", container_id, container.name, container.image])
                del tracked[container_id]
                ignored.add(container_id)


class AgentProcessSampler:
    def __init__(self, output_dir: Path, interval: float, pid_file: Path):
        self.output_dir, self.interval, self.pid_file = output_dir, interval, pid_file
        self.missed_ticks = 0

    def sample(self, pid: int) -> list:
        t = time.monotonic_ns()
        values = {
            line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()
            if ":" in line and len(line.split()) >= 2 and line.split()[1].isdigit()
        }
        return [
            t, pid, values["Rss"], values["Pss"],
            values.get("Private_Clean", 0) + values.get("Private_Dirty", 0) + values.get("Private_Hugetlb", 0),
        ]

    def run(self, stop: threading.Event):
        pid = None
        with (self.output_dir / "agent_process_samples.csv").open("w", newline="", buffering=1 << 20) as f:
            writer = csv.writer(f)
            writer.writerow(AGENT_PROCESS_COLUMNS)
            next_tick = last_flush = time.monotonic()
            while not stop.is_set():
                if pid is None:
                    try:
                        pid = int(self.pid_file.read_text().strip())
                    except (FileNotFoundError, ValueError):
                        pass
                if pid is not None:
                    try:
                        writer.writerow(self.sample(pid))
                    except (FileNotFoundError, ProcessLookupError):  # agent exited mid-tick
                        pass
                now = time.monotonic()
                if now - last_flush >= 1.0:
                    f.flush()
                    last_flush = now
                next_tick += self.interval
                if next_tick < now:
                    self.missed_ticks += int((now - next_tick) / self.interval) + 1
                    next_tick = now
                time.sleep(next_tick - now)


class GpuSampler:
    def __init__(self, output_dir: Path, interval: float):
        sys.path.append(DCGM_BINDINGS)
        import dcgm_field_helpers
        import dcgm_fields
        import dcgm_structs
        import pydcgm

        self.output_dir, self.interval, self.gpu_entity = output_dir, interval, dcgm_fields.DCGM_FE_GPU
        tag = f"mswea-system-metrics-{os.getpid()}"
        self.handle = pydcgm.DcgmHandle(ipAddress="127.0.0.1", opMode=dcgm_structs.DCGM_OPERATION_MODE_AUTO)
        self.gpu_ids = self.handle.GetSystem().discovery.GetAllSupportedGpuIds()
        self.group = pydcgm.DcgmGroup(self.handle, groupName=tag, groupType=dcgm_structs.DCGM_GROUP_DEFAULT)
        self.field_group = pydcgm.DcgmFieldGroup(self.handle, name=tag, fieldIds=list(GPU_FIELDS))
        self.group.samples.WatchFields(self.field_group, int(interval * 1e6), 60.0, 0)
        self.values = dcgm_field_helpers.DcgmFieldValueEntityCollection(self.handle.handle, self.group.GetId())
        self.start_ts_us = time.time_ns() // 1000

    def run(self, stop: threading.Event):
        try:
            with (self.output_dir / "gpu_samples.csv").open("w", newline="", buffering=1 << 20) as f:
                writer = csv.writer(f)
                writer.writerow(["t_mono_ns", "gpu", "field", "value", "dcgm_ts_us"])
                while True:
                    stopping = stop.wait(max(self.interval, 1.0))
                    self.values.GetAllSinceLastCall(self.field_group)
                    offset_ns = time.monotonic_ns() - time.time_ns()
                    for gpu, fields in self.values.values.get(self.gpu_entity, {}).items():
                        for field_id, series in fields.items():
                            for v in series.values:
                                if not v.isBlank and v.ts >= self.start_ts_us:
                                    writer.writerow([v.ts * 1000 + offset_ns, gpu, GPU_FIELDS[field_id], v.value, v.ts])
                    self.values.EmptyValues()
                    f.flush()
                    if stopping:
                        break
        finally:
            self.group.samples.UnwatchFields(self.field_group)
            self.field_group.Delete()
            self.group.Delete()


def clock_pair() -> dict:
    return {"monotonic_ns": time.monotonic_ns(), "realtime_ns": time.time_ns()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gpu-interval", type=float, default=0.1)
    parser.add_argument("--cpu-interval", type=float, default=0.1)
    parser.add_argument("--agent-pid-file", type=Path)
    parser.add_argument("--container-name-regex", default=r"^minisweagent-")
    args = parser.parse_args()
    if args.gpu_interval < 0 or args.cpu_interval < 0:
        parser.error("intervals must be >= 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    stop_clock = {}

    def request_stop(*_):
        stop_clock.setdefault("clock", clock_pair())
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    samplers = []
    if args.gpu_interval > 0:
        samplers.append(GpuSampler(args.output_dir, args.gpu_interval))
    if args.cpu_interval > 0:
        samplers.append(ContainerSampler(args.output_dir, args.cpu_interval, args.container_name_regex))
    if args.cpu_interval > 0 and args.agent_pid_file is not None:
        samplers.append(AgentProcessSampler(args.output_dir, args.cpu_interval, args.agent_pid_file))
    errors = []

    def run(sampler):
        try:
            sampler.run(stop)
        except BaseException as e:
            errors.append(f"{type(sampler).__name__}: {e!r}")
            request_stop()
            raise

    threads = [threading.Thread(target=run, args=(s,), daemon=True) for s in samplers]
    for thread in threads:
        thread.start()
    meta = {
        "gpu_interval_s": args.gpu_interval, "cpu_interval_s": args.cpu_interval,
        "agent_pid_file": str(args.agent_pid_file) if args.agent_pid_file is not None else None,
        "container_name_regex": args.container_name_regex,
        "gpu_ids": next((s.gpu_ids for s in samplers if isinstance(s, GpuSampler)), []),
        "gpu_fields": {name: field_id for field_id, name in GPU_FIELDS.items()},
        "container_columns": CONTAINER_COLUMNS, "agent_process_columns": AGENT_PROCESS_COLUMNS,
        "container_cgroup_files": CONTAINER_FILES, "clock_start": clock_pair(),
        "hostname": socket.gethostname(), "pid": os.getpid(), "host_cpus": os.cpu_count(),
    }
    meta_path = args.output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"System metrics logger started (pid {os.getpid()}): {args.output_dir}", flush=True)
    while any(t.is_alive() for t in threads):
        for thread in threads:
            thread.join(0.2)
    clock_end = clock_pair()
    meta |= {
        "clock_stop": stop_clock.get("clock", clock_end), "clock_end": clock_end,
        "container_missed_ticks": next((s.missed_ticks for s in samplers if isinstance(s, ContainerSampler)), 0),
        "agent_process_missed_ticks": next((s.missed_ticks for s in samplers if isinstance(s, AgentProcessSampler)), 0),
        "errors": errors,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"System metrics logger stopped. Errors: {errors or 'none'}", flush=True)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()

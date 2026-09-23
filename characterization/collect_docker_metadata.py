#!/usr/bin/env python3
"""Record Docker daemon/client and immutable sandbox-image metadata for a run."""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

from system_metrics import DockerSocket

DAEMON_INFO_FIELDS = [
    "Name", "ServerVersion", "Driver", "CgroupDriver", "CgroupVersion", "KernelVersion",
    "OperatingSystem", "OSType", "Architecture", "NCPU", "MemTotal", "DockerRootDir",
    "SecurityOptions", "LiveRestoreEnabled",
]
DAEMON_VERSION_FIELDS = [
    "Version", "ApiVersion", "MinAPIVersion", "GitCommit", "GoVersion", "Os", "Arch", "KernelVersion", "BuildTime",
]
CLIENT_VERSION_FIELDS = ["Version", "ApiVersion", "GitCommit", "GoVersion", "Os", "Arch", "BuildTime"]
IMAGE_FIELDS = ["Id", "RepoTags", "RepoDigests", "Created", "Architecture", "Os", "Size", "VirtualSize"]


def docker_get(path: str) -> dict:
    conn = DockerSocket()
    conn.request("GET", path)
    response = conn.getresponse()
    body = response.read()
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"Docker GET {path} returned {response.status}: {body.decode(errors='replace')}")
    return json.loads(body)


def selected(data: dict, fields: list[str]) -> dict:
    return {field: data[field] for field in fields if field in data}


def docker_client(executable: str) -> tuple[dict | None, str | None]:
    result = subprocess.run([executable, "version", "--format", "{{json .Client}}"],
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        return None, result.stderr.strip() or result.stdout.strip()
    return selected(json.loads(result.stdout), CLIENT_VERSION_FIELDS), None


def image_references(events_path: Path) -> list[str]:
    if not events_path.exists():
        return []
    with events_path.open(newline="") as f:
        return sorted({row["image"] for row in csv.DictReader(f) if row.get("image")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    errors = []
    executable = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    client, client_error = docker_client(executable)
    if client_error:
        errors.append(f"Docker client: {client_error}")
    daemon = {}
    for endpoint, fields, key in [
        ("/version", DAEMON_VERSION_FIELDS, "version"), ("/info", DAEMON_INFO_FIELDS, "info"),
    ]:
        try:
            daemon[key] = selected(docker_get(endpoint), fields)
        except Exception as e:
            errors.append(f"Docker daemon {endpoint}: {e}")
    images = []
    for reference in image_references(args.metrics_dir / "container_events.csv"):
        try:
            images.append({"reference": reference} | selected(
                docker_get(f"/images/{urllib.parse.quote(reference, safe='')}/json"), IMAGE_FIELDS))
        except Exception as e:
            images.append({"reference": reference, "error": str(e)})
            errors.append(f"Docker image {reference}: {e}")
    args.output.write_text(json.dumps({
        "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "docker_executable": executable, "client": client, "daemon": daemon, "images": images, "errors": errors,
    }, indent=2))
    print(f"Wrote {args.output}")
    if errors:
        print(f"Docker metadata errors: {errors}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

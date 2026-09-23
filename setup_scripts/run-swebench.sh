#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'HELP'
Usage: bash run-swebench.sh [--instance ID ... | --all | --filter REGEX] [--slice START:STOP] [--workers N] [--shuffle]
                          [--gpu-interval SECONDS] [--cpu-interval SECONDS] [--no-docker-stats]
Runs SWE-Bench Verified, test split, using the existing local Qwen server.
Choose instances explicitly, or supply a slice. Default workers: one.
Examples:
  bash run-swebench.sh --instance '<INSTANCE_ID>'
  bash run-swebench.sh --slice 0:5 --workers 2
  bash run-swebench.sh --all --workers 1
Slices are zero-based and exclude STOP. They apply after filtering.
--shuffle uses seed 42 before filtering and slicing. Every invocation creates a fresh run directory.
GPU and cgroup CPU/controller sampling defaults to 0.1 seconds; zero disables that side.
Docker stats runs separately at Docker's native refresh cadence (500 ms in Docker 28.1.1).
--no-docker-stats disables only Docker stats. It does not disable step timings.
Docker container metadata is collected after logging stops.
Per-task cgroup summaries and per-step call timings are written to RUN_DIRECTORY/task_metrics.
MSWEA_PROJECT_ROOT overrides the project root; it must contain mini-swe-agent/ and envs/.
HELP
}
filter=''
selection=''
instances=()
slice=''
workers=1
gpu_interval=0.1
cpu_interval=0.1
docker_stats_enabled=1
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --instance)
            [[ -z "$selection" || "$selection" == --instance ]] || { echo 'Do not combine --instance with --all or --filter.' >&2; exit 1; }
            selection=--instance
            instances+=("${2:?--instance requires an instance ID}")
            shift 2 ;;
        --all|--filter)
            [[ -z "$selection" ]] || { echo 'Do not mix selectors.' >&2; exit 1; }
            selection="$1"
            if [[ "$1" == --all ]]; then filter=''; shift
            else filter="${2:?--filter requires a regex}"; shift 2; fi ;;
        --slice) slice="${2:?--slice requires START:STOP}"; shift 2 ;;
        --workers|-w) workers="${2:?--workers requires a positive integer}"; shift 2 ;;
        --shuffle) extra+=(--shuffle); shift ;;
        --gpu-interval) gpu_interval="${2:?--gpu-interval requires seconds}"; shift 2 ;;
        --cpu-interval) cpu_interval="${2:?--cpu-interval requires seconds}"; shift 2 ;;
        --no-docker-stats) docker_stats_enabled=0; shift ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
[[ -n "$selection" || -n "$slice" ]] || { usage >&2; exit 1; }
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'Workers must be a positive integer.' >&2; exit 1; }
[[ -z "$slice" || "$slice" =~ ^[0-9]*:[0-9]*$ ]] || { echo 'Use a slice such as 0:5.' >&2; exit 1; }
for interval in "$gpu_interval" "$cpu_interval"; do
    [[ "$interval" =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ ]] || { echo "Invalid interval: $interval" >&2; exit 1; }
done
if [[ "$selection" == --instance ]]; then
    for instance in "${instances[@]}"; do
        [[ "$instance" =~ ^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+$ ]] || { echo "Invalid instance ID: $instance" >&2; exit 1; }
    done
fi
project="${MSWEA_PROJECT_ROOT:-/mnt/raid0/jovans/yigit/agent-characterization}"
agent_root="$project/mini-swe-agent"
source "$agent_root/.venv/bin/activate"
cd "$agent_root"
if [[ "$selection" == --instance ]]; then
    filter=$("$agent_root/.venv/bin/python" -c 'import re,sys; print("^(?:"+"|".join(map(re.escape,sys.argv[1:]))+")$")' "${instances[@]}")
fi
curl --fail --silent --show-error --max-time 15 http://127.0.0.1:8000/v1/models |
    "$agent_root/.venv/bin/python" -c 'import json,sys; assert any(m["id"]=="Qwen/Qwen3-235B-A22B-Instruct-2507" for m in json.load(sys.stdin)["data"]), "Expected Qwen model is not served"'
mkdir -p "$project/swebench-runs"
run_dir=$(mktemp -d "$project/swebench-runs/qwen-run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
cat > "$run_dir/local-qwen.yaml" <<YAML
model:
  model_name: hosted_vllm/Qwen/Qwen3-235B-A22B-Instruct-2507
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: http://127.0.0.1:8000/v1
    num_retries: 0
YAML
metrics_dir="$run_dir/system_metrics"
docker_stats_dir="$run_dir/docker_stats"
mkdir -p "$metrics_dir"
agent_pid_file="$metrics_dir/agent.pid"
metrics_pid=''
docker_stats_pid=''
stop_metrics() {
    status=$?
    trap - EXIT
    if [[ -n "$metrics_pid" ]]; then kill -TERM "$metrics_pid" 2>/dev/null || true; fi
    if [[ -n "$docker_stats_pid" ]]; then kill -TERM "$docker_stats_pid" 2>/dev/null || true; fi
    if [[ -n "$metrics_pid" ]]; then
        wait "$metrics_pid" || echo "System metrics logger reported an error; see $metrics_dir/logger.log" >&2
    fi
    if [[ -n "$docker_stats_pid" ]]; then
        wait "$docker_stats_pid" || echo "Docker stats logger reported an error; see $run_dir/docker-stats.log" >&2
    fi
    if ! "$agent_root/.venv/bin/python" "$agent_root/characterization/collect_docker_metadata.py" \
        "$metrics_dir" "$run_dir/docker-metadata.json"; then
        echo 'Could not collect Docker metadata.' >&2
    fi
    if ! "$agent_root/.venv/bin/python" "$agent_root/characterization/analyze_task_metrics.py" "$run_dir"; then
        echo "Task analysis failed; raw trajectories and cgroup samples remain in $run_dir." >&2
    fi
    exit "$status"
}
trap stop_metrics EXIT
"$agent_root/.venv/bin/python" "$agent_root/characterization/system_metrics.py" "$metrics_dir" \
    --gpu-interval "$gpu_interval" --cpu-interval "$cpu_interval" --agent-pid-file "$agent_pid_file" \
    > "$metrics_dir/logger.log" 2>&1 &
metrics_pid=$!
if (( docker_stats_enabled )); then
    "$agent_root/.venv/bin/python" "$agent_root/characterization/docker_stats.py" "$docker_stats_dir" \
        --docker-executable "${MSWEA_DOCKER_EXECUTABLE:-docker}" > "$run_dir/docker-stats.log" 2>&1 &
    docker_stats_pid=$!
fi
for _ in {1..100}; do
    kill -0 "$metrics_pid" 2>/dev/null || { cat "$metrics_dir/logger.log" >&2; metrics_pid=''; exit 1; }
    if [[ -n "$docker_stats_pid" ]]; then
        kill -0 "$docker_stats_pid" 2>/dev/null || { cat "$run_dir/docker-stats.log" >&2; docker_stats_pid=''; exit 1; }
    fi
    if [[ -f "$metrics_dir/meta.json" ]] && { (( ! docker_stats_enabled )) || [[ -f "$docker_stats_dir/ready" ]]; }; then break; fi
    sleep 0.1
done
[[ -f "$metrics_dir/meta.json" ]] || { echo 'System metrics logger did not start.' >&2; exit 1; }
if (( docker_stats_enabled )); then
    [[ -f "$docker_stats_dir/ready" ]] || { echo "Docker stats logger did not start; see $run_dir/docker-stats.log" >&2; exit 1; }
fi
command=("$agent_root/.venv/bin/mini-extra" swebench
    -c "$agent_root/src/minisweagent/config/benchmarks/swebench.yaml"
    -c "$run_dir/local-qwen.yaml"
    -m hosted_vllm/Qwen/Qwen3-235B-A22B-Instruct-2507
    -o "$run_dir/results" --subset verified --split test
    --filter "$filter" --slice "$slice" --workers "$workers")
if [[ ${#extra[@]} -gt 0 ]]; then command+=("${extra[@]}"); fi
printf '%q ' "${command[@]}" > "$run_dir/agent-command.txt"
printf '\n' >> "$run_dir/agent-command.txt"
echo "Run directory: $run_dir"
echo "After inspecting the run, evaluate in Terminal 2 with:"
printf 'bash %q %q\n' "$agent_root/setup_scripts/evaluate-swebench.sh" "$run_dir"
"${command[@]}" &
agent_pid=$!
printf '%s\n' "$agent_pid" > "$agent_pid_file"
wait "$agent_pid"
echo "Runner finished. Inspect exit statuses in $run_dir/results."

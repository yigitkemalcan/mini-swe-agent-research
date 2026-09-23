#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash evaluate-swebench.sh RUN_DIRECTORY [WORKERS]'
    echo 'Evaluates predictions from run-swebench.sh. Default: one evaluation worker.'
    exit 0
fi
[[ $# -ge 1 && $# -le 2 ]] || { echo 'Usage: bash evaluate-swebench.sh RUN_DIRECTORY [WORKERS]' >&2; exit 1; }
run_dir=$(cd "$1" && pwd -P)
workers=${2:-1}
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'Workers must be a positive integer.' >&2; exit 1; }
[[ -f "$run_dir/results/preds.json" ]] || { echo "Missing predictions: $run_dir/results/preds.json" >&2; exit 1; }
project="${MSWEA_PROJECT_ROOT:-/mnt/raid0/jovans/yigit/agent-characterization}"
evaluator="$project/envs/swebench-eval"
source "$evaluator/bin/activate"
cd "$project"
"$evaluator/bin/python" - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
predictions = json.loads((root / "results/preds.json").read_text())
if not isinstance(predictions, dict) or not predictions:
    raise ValueError("Expected a nonempty predictions dictionary")
destination = root / "results/preds.swebench.jsonl"
destination.write_text("".join(json.dumps(prediction) + "\n" for prediction in predictions.values()))
print(f"Converted {len(predictions)} predictions: {destination}")
PY
evaluation_dir=$(mktemp -d "$run_dir/evaluation-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
run_id="$(basename "$run_dir")-$(basename "$evaluation_dir")"
printf '%s\n' "$run_id" > "$evaluation_dir/run-id.txt"
echo "Evaluation run ID: $run_id"
echo "Detailed evaluator logs: $project/logs/run_evaluation/$run_id"
echo "Evaluator summary JSON will be written under: $evaluation_dir"
"$evaluator/bin/swebench" eval verified -p "$run_dir/results/preds.swebench.jsonl" \
    -j "$workers" --run-id "$run_id" --report-dir "$evaluation_dir" 2>&1 | tee "$evaluation_dir/evaluator.log"

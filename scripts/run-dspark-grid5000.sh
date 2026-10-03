#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 /persistent/path/to/e5-small-100-seed42 /persistent/path/to/new-dspark-output" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

RUN_DIR=$1
OUTPUT_DIR=$2
if [[ "$RUN_DIR" != /* || "$OUTPUT_DIR" != /* ]]; then
  echo "RUN_DIR and OUTPUT_DIR must be absolute paths." >&2
  exit 2
fi
if [[ ! -f "$RUN_DIR/frozen_sample.json" || ! -f "$RUN_DIR/candidate_labels.csv" ]]; then
  echo "RUN_DIR must contain frozen_sample.json and candidate_labels.csv." >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR already exists; choose a fresh persistent path." >&2
  exit 2
fi

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "Use a clean, committed checkout so the run manifest identifies the executed code." >&2
  exit 2
fi
COMMIT_SHA=$(git -C "$PROJECT_ROOT" rev-parse HEAD)

TMP_PARENT=${TMPDIR:-/tmp}
if [[ ! -d "$TMP_PARENT" ]]; then
  echo "TMPDIR must point to an existing job-local directory." >&2
  exit 2
fi
TMP_PARENT=$(cd -- "$TMP_PARENT" && pwd)
OUTPUT_PARENT=$(dirname -- "$OUTPUT_DIR")
if [[ ! -d "$TMP_PARENT" || ! -d "$OUTPUT_PARENT" || ! -w "$OUTPUT_PARENT" ]]; then
  echo "TMPDIR and the existing persistent output parent must be directories; output parent must be writable." >&2
  exit 2
fi
python3 - "$TMP_PARENT" "$RUN_DIR" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

temporary_root = Path(sys.argv[1]).resolve()
for label, raw_path in (
    ("RUN_DIR", sys.argv[2]),
    ("OUTPUT_DIR", sys.argv[3]),
):
    path = Path(raw_path).resolve()
    if temporary_root == path or temporary_root in path.parents:
        raise SystemExit(f"{label} must be outside job-local temporary storage")
PY

MAX_CACHE_BYTES=$((20 * 1024 * 1024 * 1024))
MAX_OUTPUT_BYTES=$((1024 * 1024 * 1024))
python3 - "$TMP_PARENT" "$OUTPUT_PARENT" "$MAX_CACHE_BYTES" "$MAX_OUTPUT_BYTES" <<'PY'
import shutil
import sys

tmp_parent, output_parent = sys.argv[1:3]
cache_required, output_required = map(int, sys.argv[3:])
for label, path, required in (
    ("job-local temporary storage", tmp_parent, cache_required),
    ("persistent output storage", output_parent, output_required),
):
    free = shutil.disk_usage(path).free
    if free < required:
        raise SystemExit(
            f"{label} needs at least {required / 1024**3:.0f} GiB free; "
            f"found {free / 1024**3:.1f} GiB at {path}"
        )
PY

if ! command -v nvidia-smi >/dev/null || ! command -v setsid >/dev/null; then
  echo "nvidia-smi and setsid are required on the allocated Linux GPU node." >&2
  exit 2
fi
GPU_RECORD=$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits)
python3 - "$GPU_RECORD" <<'PY'
import sys

rows = [line.strip() for line in sys.argv[1].splitlines() if line.strip()]
if len(rows) != 1:
    raise SystemExit(f"Expected exactly one visible GPU; found {len(rows)}")
name, memory, capability = (item.strip() for item in rows[0].rsplit(",", maxsplit=2))
memory_mib = int(float(memory))
compute = tuple(int(part) for part in capability.split(".", maxsplit=1))
if memory_mib < 16_384 or compute < (8, 0):
    raise SystemExit(
        f"GPU gate requires >= 16384 MiB and compute capability >= 8.0; "
        f"found {memory_mib} MiB, capability {capability} ({name})"
    )
print(f"GPU preflight passed: {name}, {memory_mib} MiB, capability {capability}")
PY

JOB_TMP_ROOT=$(mktemp -d --tmpdir="$TMP_PARENT" georeset-dspark.XXXXXX)
case "$JOB_TMP_ROOT" in
  "$TMP_PARENT"/*) ;;
  *) echo "Temporary directory is outside TMPDIR; refusing cleanup." >&2; exit 2 ;;
esac
chmod 700 "$JOB_TMP_ROOT"
RUNNER_PID=
cleanup() {
  if [[ -n "${RUNNER_PID:-}" ]]; then
    RUNNER_PGID=$(ps -o pgid= -p "$RUNNER_PID" | tr -d '[:space:]' || true)
    SCRIPT_PGID=$(ps -o pgid= -p "$$" | tr -d '[:space:]' || true)
    if [[ -n "$RUNNER_PGID" && "$RUNNER_PGID" != "$SCRIPT_PGID" ]]; then
      kill -TERM -- "-$RUNNER_PGID" 2>/dev/null || true
    else
      kill -TERM "$RUNNER_PID" 2>/dev/null || true
    fi
    wait "$RUNNER_PID" 2>/dev/null || true
  fi
  rm -rf -- "$JOB_TMP_ROOT"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$JOB_TMP_ROOT/home" "$JOB_TMP_ROOT/hf" "$JOB_TMP_ROOT/xdg" \
  "$JOB_TMP_ROOT/uv-cache" "$JOB_TMP_ROOT/tmp"
export HOME="$JOB_TMP_ROOT/home"
export TMPDIR="$JOB_TMP_ROOT/tmp"
export HF_HOME="$JOB_TMP_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$JOB_TMP_ROOT/xdg"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export UV_CACHE_DIR="$JOB_TMP_ROOT/uv-cache"
export UV_PROJECT_ENVIRONMENT="$JOB_TMP_ROOT/venv"

JOB_START=$(date +%s)
JOB_BUDGET_SECONDS=$((55 * 60))
run_bounded() {
  setsid --wait "$@" &
  RUNNER_PID=$!
  local runner_pgid=
  for _ in {1..50}; do
    runner_pgid=$(ps -o pgid= -p "$RUNNER_PID" | tr -d '[:space:]')
    [[ -n "$runner_pgid" ]] && break
    if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
      if wait "$RUNNER_PID"; then
        RUNNER_PID=
        return 0
      else
        local completed_status=$?
        RUNNER_PID=
        return "$completed_status"
      fi
    fi
    sleep 0.1
  done
  SCRIPT_PGID=$(ps -o pgid= -p "$$" | tr -d '[:space:]')
  if [[ -z "$runner_pgid" || "$runner_pgid" == "$SCRIPT_PGID" ]]; then
    echo "Could not isolate a process group for the cache and time limits." >&2
    kill -TERM "$RUNNER_PID" 2>/dev/null || true
    wait "$RUNNER_PID" 2>/dev/null || true
    RUNNER_PID=
    return 2
  fi

  local run_status=0 wait_status
  while kill -0 "$RUNNER_PID" 2>/dev/null; do
    local cache_bytes elapsed_seconds
    cache_bytes=$(du -sb "$JOB_TMP_ROOT" | awk '{print $1}')
    elapsed_seconds=$(( $(date +%s) - JOB_START ))
    if (( cache_bytes > MAX_CACHE_BYTES )); then
      echo "Job-local model/runtime cache exceeded 20 GiB; stopping." >&2
      run_status=1
      break
    fi
    if (( elapsed_seconds >= JOB_BUDGET_SECONDS )); then
      echo "Run reached its 55-minute budget; stopping before the one-hour allocation ends." >&2
      run_status=124
      break
    fi
    sleep 2
  done
  if (( run_status != 0 )); then
    kill -TERM -- "-$runner_pgid" 2>/dev/null || true
    sleep 5
    kill -KILL -- "-$runner_pgid" 2>/dev/null || true
  fi
  if wait "$RUNNER_PID"; then
    wait_status=0
  else
    wait_status=$?
  fi
  if (( run_status == 0 )); then
    run_status=$wait_status
  fi
  local final_cache_bytes
  final_cache_bytes=$(du -sb "$JOB_TMP_ROOT" | awk '{print $1}')
  if (( run_status == 0 && final_cache_bytes > MAX_CACHE_BYTES )); then
    echo "Job-local model/runtime cache exceeded 20 GiB; stopping." >&2
    run_status=1
  fi
  RUNNER_PID=
  return "$run_status"
}

cd "$PROJECT_ROOT"
echo "Installing the locked DSpark runtime in job-local temporary storage."
run_bounded uv sync --locked --no-default-groups --extra dspark --python 3.12
run_bounded uv cache clean
echo "Running the frozen 100-row pilot; temporary cache limit is 20 GiB."
run_bounded uv run --locked georeset-pilot run-dspark \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-cache "$HF_HOME" \
  --computation-commit "$COMMIT_SHA" \
  --validation-commit "$COMMIT_SHA"

OUTPUT_BYTES=$(du -sb "$OUTPUT_DIR" | awk '{print $1}')
if (( OUTPUT_BYTES >= MAX_OUTPUT_BYTES )); then
  echo "Persistent evaluation outputs reached 1 GiB; expected less than 1 GiB." >&2
  exit 1
fi
echo "Run complete: $OUTPUT_BYTES persistent output bytes at $OUTPUT_DIR"

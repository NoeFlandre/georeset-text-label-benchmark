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
if [[ ! -f "$RUN_DIR/frozen_sample.json" || ! -f "$RUN_DIR/candidate_labels.csv" || ! -f "$RUN_DIR/manifest.json" ]]; then
  echo "RUN_DIR must contain frozen_sample.json, candidate_labels.csv, and manifest.json." >&2
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

source /etc/profile.d/lmod.sh 2>/dev/null || true
if ! type module >/dev/null 2>&1; then
  echo "The Grid'5000 Lmod module command is required on the allocated node." >&2
  exit 2
fi
if ! module load "${DS_DRIVER_MODULE:-nvidia-driver-libs/580}"; then
  echo "Could not load the CUDA 13 compatible NVIDIA driver libraries." >&2
  exit 2
fi
if ! module load "${DS_CUDA_MODULE:-cuda-toolkit/13.0.2}"; then
  echo "Could not load the pinned CUDA 13.0 toolkit." >&2
  exit 2
fi
if ! module load "${DS_UV_MODULE:-uv/0.10.12}"; then
  echo "Could not load uv from the Grid'5000 module environment." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null || ! command -v setsid >/dev/null \
  || ! command -v nvcc >/dev/null || ! command -v uv >/dev/null; then
  echo "nvidia-smi, setsid, CUDA nvcc, and uv are required on the allocated node." >&2
  exit 2
fi
NVCC_OUTPUT=$(nvcc --version)
if [[ "$NVCC_OUTPUT" != *"release 13.0"* ]]; then
  echo "The loaded CUDA toolkit must report release 13.0; refusing to install the runtime." >&2
  exit 2
fi
GPU_RECORD=$(nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits)
python3 - "$GPU_RECORD" <<'PY'
import re
import sys

rows = [line.strip() for line in sys.argv[1].splitlines() if line.strip()]
if len(rows) != 1:
    raise SystemExit(f"Expected exactly one visible GPU; found {len(rows)}")
name, memory, capability, driver_version = (
    item.strip() for item in rows[0].rsplit(",", maxsplit=3)
)
memory_mib = int(float(memory))
compute = tuple(int(part) for part in capability.split(".", maxsplit=1))
driver_match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", driver_version)
if driver_match is None:
    raise SystemExit(f"Could not parse the CUDA driver version: {driver_version}")
driver = tuple(int(part) for part in driver_match.groups())
if memory_mib < 16_384 or compute < (8, 0):
    raise SystemExit(
        f"GPU gate requires >= 16384 MiB and compute capability >= 8.0; "
        f"found {memory_mib} MiB, capability {capability} ({name})"
    )
if driver < (580, 65, 6):
    raise SystemExit(
        f"CUDA driver {driver_version} is below the minimum 580.65.06 required for CUDA 13"
    )
print(
    f"GPU preflight passed: {name}, {memory_mib} MiB, capability {capability}, "
    f"driver {driver_version}"
)
PY

CUDA_ROOT=$(dirname -- "$(dirname -- "$(command -v nvcc)")")
export CUDA_HOME="$CUDA_ROOT"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"

JOB_TMP_ROOT=$(mktemp -d --tmpdir="$TMP_PARENT" georeset-dspark.XXXXXX)
case "$JOB_TMP_ROOT" in
  "$TMP_PARENT"/*) ;;
  *) echo "Temporary directory is outside TMPDIR; refusing cleanup." >&2; exit 2 ;;
esac
chmod 700 "$JOB_TMP_ROOT"
RUNNER_PID=
RUNNER_PGID=
RUNNER_WAIT_STATUS=
source "$PROJECT_ROOT/scripts/run-bounded.sh"
runner_install_traps
runner_initialize

mkdir -p "$JOB_TMP_ROOT/home" "$JOB_TMP_ROOT/hf" "$JOB_TMP_ROOT/xdg" \
  "$JOB_TMP_ROOT/uv-cache" "$JOB_TMP_ROOT/tmp"
export HOME="$JOB_TMP_ROOT/home"
export TMPDIR="$JOB_TMP_ROOT/tmp"
export HF_HOME="$JOB_TMP_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$JOB_TMP_ROOT/xdg"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export FLASHINFER_WORKSPACE_BASE="$JOB_TMP_ROOT/flashinfer"
export UV_CACHE_DIR="$JOB_TMP_ROOT/uv-cache"
export UV_PROJECT_ENVIRONMENT="$JOB_TMP_ROOT/venv"

JOB_START=$(date +%s)
JOB_BUDGET_SECONDS=$((55 * 60))

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

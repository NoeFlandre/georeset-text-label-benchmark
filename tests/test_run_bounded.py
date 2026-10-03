"""Shell regressions for the Grid'5000 bounded process runner."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import pytest


def _runner_library() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts/run-bounded.sh"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Grid'5000 bounded runner library is missing")


def _driver(tmp_path: Path, *, body: str, ps_mode: str = "normal") -> tuple[Path, dict[str, str]]:
    shell = tmp_path / "driver.sh"
    shell.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
source "$RUNNER_LIBRARY"
RUNNER_PID=
RUNNER_PGID=
RUNNER_WAIT_STATUS=
JOB_TMP_ROOT=$(mktemp -d)
MAX_CACHE_BYTES=$((1024 * 1024 * 1024))
JOB_START=$(date +%s)
JOB_BUDGET_SECONDS=3600
CALLER_PID=$$
CALLER_PGID=$("$REAL_PS" -o pgid= -p "$CALLER_PID" | tr -d '[:space:]')
export CALLER_PID CALLER_PGID
SCRIPT_PGID=$CALLER_PGID
SIGNAL_LOG=$RUNNER_SIGNAL_LOG
kill() {
  if [[ "${1:-}" == "-TERM" || "${1:-}" == "-KILL" ]] &&
     [[ "${2:-}" == "--" && "${3:-}" == -* ]]; then
    local target=${3#-}
    if [[ "$target" == "$CALLER_PGID" ]]; then
      printf 'SHARED %s %s\\n' "$1" "$target" >> "$SIGNAL_LOG"
      return 97
    fi
    printf '%s %s\\n' "$1" "$target" >> "$SIGNAL_LOG"
  fi
  builtin kill "$@"
}
trap runner_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"""
        + body
        + "\n",
        encoding="utf-8",
    )
    shell.chmod(0o755)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    real_ps = shutil.which("ps")
    assert real_ps is not None
    (fake_bin / "ps").write_text(
        """#!/usr/bin/env bash
set -u
target_pid=
previous=
previous_arg=
for arg in "$@"; do
  if [[ "$previous_arg" == "-p" ]]; then target_pid=$arg; break; fi
  previous_arg=$arg
done
if [[ "$target_pid" != "${CALLER_PID:-}" && -n "$target_pid" ]]; then
  previous=$(cat "$PS_STATE" 2>/dev/null || true)
  case "${PS_MODE:-normal}" in
    fail-once-per-pid)
      if [[ "$target_pid" != "$previous" ]]; then
        printf '%s' "$target_pid" > "$PS_STATE"
        exit 1
      fi
      ;;
    same-group-once-per-pid)
      if [[ "$target_pid" != "$previous" ]]; then
        printf '%s' "$target_pid" > "$PS_STATE"
        printf ' %s\\n' "$CALLER_PGID" || exit $?
        printf 'injected\\n' > "$PS_INJECTION_LOG"
        exit 0
      fi
      ;;
  esac
fi
exec "$REAL_PS" "$@"
""",
        encoding="utf-8",
    )
    (fake_bin / "ps").chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "RUNNER_LIBRARY": str(_runner_library()),
            "REAL_PS": real_ps,
            "PS_MODE": ps_mode,
            "PS_STATE": str(tmp_path / "ps-state"),
            "PS_INJECTION_LOG": str(tmp_path / "ps-injection.log"),
            "RUNNER_SIGNAL_LOG": str(tmp_path / "signals.log"),
        }
    )
    return shell, environment


def _run_driver(
    shell: Path,
    environment: dict[str, str],
    *,
    timeout: float = 10,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        ["bash", str(shell)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_test_processes(process.pid, environment)
        pytest.fail("bounded shell regression exceeded its deadline")
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _kill_test_processes(driver_pid: int, environment: dict[str, str]) -> None:
    child_file = environment.get("CHILD_PID_FILE")
    if child_file and Path(child_file).is_file():
        try:
            child_pid = int(Path(child_file).read_text(encoding="utf-8"))
            child_group = os.getpgid(child_pid)
            if child_group != os.getpgid(driver_pid):
                os.killpg(child_group, signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    with suppress(ProcessLookupError):
        os.killpg(driver_pid, signal.SIGKILL)


def _assert_not_running(pid: int) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        stat_path = Path(f"/proc/{pid}/stat")
        if not stat_path.exists():
            return
        state = stat_path.read_text(encoding="utf-8").rsplit(")", maxsplit=1)[1].split()[0]
        if state in {"Z", "X"}:
            return
        time.sleep(0.05)
    pytest.fail(f"process {pid} was left running after process-group teardown")


def test_fast_successes_survive_transient_ps_failure(tmp_path: Path) -> None:
    shell, environment = _driver(
        tmp_path,
        body="for _ in {1..30}; do run_bounded true; done",
        ps_mode="fail-once-per-pid",
    )

    result = _run_driver(shell, environment)

    assert result.returncode == 0, result.stderr
    signal_log = Path(environment["RUNNER_SIGNAL_LOG"])
    assert "SHARED" not in (signal_log.read_text() if signal_log.exists() else "")


def test_fast_failure_preserves_child_exit_status_after_transient_ps_failure(
    tmp_path: Path,
) -> None:
    shell, environment = _driver(
        tmp_path,
        body="run_bounded bash -c 'exit 7'",
        ps_mode="fail-once-per-pid",
    )

    result = _run_driver(shell, environment)

    assert result.returncode == 7, result.stderr
    signal_log = Path(environment["RUNNER_SIGNAL_LOG"])
    assert "SHARED" not in (signal_log.read_text() if signal_log.exists() else "")


def test_runner_waits_for_a_process_group_distinct_from_its_own(tmp_path: Path) -> None:
    shell, environment = _driver(
        tmp_path,
        body="run_bounded sleep 0.3",
        ps_mode="same-group-once-per-pid",
    )

    result = _run_driver(shell, environment)

    assert result.returncode == 0, result.stderr
    assert Path(environment["PS_INJECTION_LOG"]).read_text(encoding="utf-8") == "injected\n"
    signal_log = Path(environment["RUNNER_SIGNAL_LOG"])
    assert "SHARED" not in (signal_log.read_text() if signal_log.exists() else "")


def test_timeout_terminates_the_owned_process_group_without_an_orphan(
    tmp_path: Path,
) -> None:
    child_pid_file = tmp_path / "child.pid"
    status_file = tmp_path / "status"
    body = f"""JOB_START=$(( $(date +%s) + 2 ))
JOB_BUDGET_SECONDS=0
set +e
run_bounded bash -c 'echo $$ > "{child_pid_file}"; exec sleep 30'
status=$?
set -e
printf '%s' "$status" > "{status_file}"
"""
    shell, environment = _driver(tmp_path, body=body)
    environment["CHILD_PID_FILE"] = str(child_pid_file)

    result = _run_driver(shell, environment, timeout=15)

    assert result.returncode == 0, result.stderr
    assert status_file.read_text(encoding="utf-8") == "124"
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    _assert_not_running(child_pid)
    signals = Path(environment["RUNNER_SIGNAL_LOG"]).read_text(encoding="utf-8")
    assert "-TERM " in signals
    assert "SHARED" not in signals


@pytest.mark.parametrize(
    ("signal_number", "expected_status"), [(signal.SIGTERM, 143), (signal.SIGINT, 130)]
)
def test_signal_cleanup_kills_term_ignoring_child_without_shared_group_signal(
    tmp_path: Path,
    signal_number: int,
    expected_status: int,
) -> None:
    child_pid_file = tmp_path / "child.pid"
    body = f'run_bounded bash -c \'trap "" TERM; echo $$ > "{child_pid_file}"; exec sleep 30\''
    shell, environment = _driver(tmp_path, body=body)
    environment["CHILD_PID_FILE"] = str(child_pid_file)
    process = subprocess.Popen(
        ["bash", str(shell)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while (
            not child_pid_file.is_file() and process.poll() is None and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert child_pid_file.is_file(), "runner child did not become ready"
        child_pid = int(child_pid_file.read_text(encoding="utf-8"))
        child_group = os.getpgid(child_pid)
        assert child_group != os.getpgid(process.pid)
        process.send_signal(signal_number)
        stdout, stderr = process.communicate(timeout=12)
    except subprocess.TimeoutExpired:
        _kill_test_processes(process.pid, environment)
        pytest.fail("signal cleanup waited indefinitely for a TERM-ignoring child")
    finally:
        if process.poll() is None:
            _kill_test_processes(process.pid, environment)

    assert process.returncode == expected_status, stderr or stdout
    _assert_not_running(child_pid)
    signals = Path(environment["RUNNER_SIGNAL_LOG"]).read_text(encoding="utf-8")
    assert f"-TERM {child_group}" in signals
    assert f"-KILL {child_group}" in signals
    assert "SHARED" not in signals

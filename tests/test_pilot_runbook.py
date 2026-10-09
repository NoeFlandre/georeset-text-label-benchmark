from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "pilot.md"
COMMIT = "a" * 40
ATTEMPT = "pilot-attempt"


def _bash_blocks_after(marker: str) -> list[str]:
    text = DOCS.read_text(encoding="utf-8").split(marker, 1)[1]
    return re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL)


def _write_executable(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _fake_tools(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    bin_dir = tmp_path / "bin"
    call_log = tmp_path / "calls.log"
    _write_executable(
        bin_dir / "hf",
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$PILOT_TEST_HF_LOG"\nexit 0\n',
    )
    _write_executable(
        bin_dir / "git",
        "#!/bin/sh\n"
        'if [ "$1" = status ]; then\n'
        '  case "$*" in\n'
        '    *"--untracked-files=all"*)\n'
        "      printf '%s' \"$PILOT_TEST_POST_SMOKE_GIT_STATUS\"\n"
        '      exit "${PILOT_TEST_POST_SMOKE_GIT_STATUS_EXIT:-0}";;\n'
        "  esac\n"
        '  case "$PILOT_TEST_GIT_STATUS" in\n'
        '    "?? "*) [ "$3" = "--untracked-files=no" ] && exit 0;;\n'
        "  esac\n"
        "  printf '%s' \"$PILOT_TEST_GIT_STATUS\"\n"
        '  exit "${PILOT_TEST_GIT_STATUS_EXIT:-0}"\n'
        "else\n"
        '  if [ -n "${PILOT_TEST_COMMIT_SEQUENCE_FILE:-}" ]; then\n'
        '    count=$(cat "${PILOT_TEST_COMMIT_SEQUENCE_FILE}.count" 2>/dev/null || printf 0)\n'
        "    count=$((count + 1))\n"
        '    printf \'%s\' "$count" > "${PILOT_TEST_COMMIT_SEQUENCE_FILE}.count"\n'
        '    sed -n "${count}p" "$PILOT_TEST_COMMIT_SEQUENCE_FILE"\n'
        "    exit 0\n"
        "  fi\n"
        "  printf '%s\\n' \"$PILOT_TEST_COMMIT\"\n"
        "fi\n",
    )
    _write_executable(
        bin_dir / "python",
        '#!/bin/sh\ncase "$*" in\n'
        '  *"import uuid"*) printf \'%s\\n\' "$PILOT_TEST_ATTEMPT";;\n'
        '  *) exec "$PILOT_TEST_REAL_PYTHON" "$@";;\n'
        "esac\n",
    )
    _write_executable(
        bin_dir / "uv",
        "#!/bin/sh\n{\n"
        "  printf 'CALL\\n'\n"
        "  printf '%s\\n' \"$@\"\n"
        "  printf 'HF_HOME=%s\\n' \"${HF_HOME-<unset>}\"\n"
        "  printf 'HF_HUB_CACHE=%s\\n' \"${HF_HUB_CACHE-<unset>}\"\n"
        "  printf 'END\\n'\n"
        '} >> "$PILOT_TEST_CALL_LOG"\n'
        'for arg in "$@"; do\n'
        '  [ "$arg" = "run-dspark-smoke" ] && exit "$PILOT_TEST_SMOKE_STATUS"\n'
        "done\nexit 0\n",
    )
    return bin_dir, call_log


def _environment(
    bin_dir: pathlib.Path, call_log: pathlib.Path, smoke_status: str
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "PILOT_TEST_ATTEMPT": ATTEMPT,
            "PILOT_TEST_CALL_LOG": str(call_log),
            "PILOT_TEST_COMMIT": COMMIT,
            "PILOT_TEST_GIT_STATUS": "",
            "PILOT_TEST_GIT_STATUS_EXIT": "0",
            "PILOT_TEST_HF_LOG": str(call_log.parent / "hf.log"),
            "PILOT_TEST_REAL_PYTHON": sys.executable,
            "PILOT_TEST_SMOKE_STATUS": smoke_status,
            "TMPDIR": str(call_log.parent),
            "HF_HOME": "/inherited/hf-home",
            "HF_HUB_CACHE": "/inherited/hf-hub-cache",
        }
    )
    return env


def _calls(call_log: pathlib.Path) -> list[str]:
    if not call_log.exists():
        return []
    return [part.split("END\n", 1)[0] for part in call_log.read_text().split("CALL\n")[1:]]


def _cache_path(call: str) -> str:
    arguments = call.split("HF_HOME=", 1)[0].splitlines()
    return arguments[arguments.index("--model-cache") + 1]


def _write_passing_smoke(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "dspark_metrics.json").write_text(
        json.dumps({"smoke_gate": {"passed": True}}), encoding="utf-8"
    )


def test_grid_wrapper_rejects_failed_cleanliness_check_before_recording_head(
    tmp_path: pathlib.Path,
) -> None:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    wrapper = tmp_path / "project/scripts/run-dspark-grid5000.sh"
    _write_executable(
        wrapper,
        (project_root / "scripts/run-dspark-grid5000.sh").read_text(encoding="utf-8"),
    )
    run_dir = tmp_path / "inputs"
    run_dir.mkdir()
    for name in ("frozen_sample.json", "candidate_labels.csv", "manifest.json"):
        (run_dir / name).write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "outputs/new-run"
    output_dir.parent.mkdir(parents=True)
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    fake_bin = tmp_path / "bin"
    git_log = tmp_path / "git.log"
    _write_executable(
        fake_bin / "git",
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$PILOT_TEST_GIT_LOG"\n'
        'if [[ " $* " == *" status "* ]]; then exit "${PILOT_TEST_GIT_STATUS_EXIT:-0}"; fi\n'
        'if [[ " $* " == *" rev-parse "* ]]; then printf \'%s\\n\' "$PILOT_TEST_COMMIT"; exit 0; fi\n'
        "exit 2\n",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "PILOT_TEST_COMMIT": COMMIT,
            "PILOT_TEST_GIT_LOG": str(git_log),
            "PILOT_TEST_GIT_STATUS_EXIT": "128",
            "TMPDIR": str(scratch_dir),
        }
    )

    result = subprocess.run(
        [str(wrapper), str(run_dir), str(output_dir)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    git_calls = git_log.read_text(encoding="utf-8").splitlines()
    assert result.returncode == 2
    assert "Could not verify a clean, committed checkout" in result.stderr
    assert any(" status " in f" {call} " for call in git_calls)
    assert all("rev-parse" not in call for call in git_calls)


def test_grid_wrapper_rejects_checkout_commit_mismatch_before_runtime(
    tmp_path: pathlib.Path,
) -> None:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    wrapper = tmp_path / "project/scripts/run-dspark-grid5000.sh"
    _write_executable(
        wrapper,
        (project_root / "scripts/run-dspark-grid5000.sh").read_text(encoding="utf-8"),
    )
    run_dir = tmp_path / "inputs"
    run_dir.mkdir()
    for name in ("frozen_sample.json", "candidate_labels.csv", "manifest.json"):
        (run_dir / name).write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "outputs/new-run"
    output_dir.parent.mkdir(parents=True)
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    fake_bin = tmp_path / "bin"
    _write_executable(
        fake_bin / "git",
        "#!/usr/bin/env bash\n"
        'if [[ " $* " == *" status "* ]]; then exit 0; fi\n'
        'if [[ " $* " == *" rev-parse "* ]]; then printf \'%s\\n\' "$PILOT_TEST_COMMIT"; exit 0; fi\n'
        "exit 2\n",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "PILOT_TEST_COMMIT": COMMIT,
            "TMPDIR": str(scratch_dir),
        }
    )

    result = subprocess.run(
        [str(wrapper), str(run_dir), str(output_dir), "--expected-commit", "b" * 40],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "does not match expected commit" in result.stderr


def test_grid_wrapper_usage_names_the_geographic_run_directory() -> None:
    wrapper = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run-dspark-grid5000.sh"
    usage = next(
        line for line in wrapper.read_text(encoding="utf-8").splitlines() if "Usage:" in line
    )

    assert "pilot-100-seed42" in usage
    assert "e5-small" not in wrapper.read_text(encoding="utf-8")

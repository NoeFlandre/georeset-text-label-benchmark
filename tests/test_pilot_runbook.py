from __future__ import annotations

import hashlib
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


def _smoke_wrapper(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, dict[str, str]]:
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
    (tmp_path / "outputs").mkdir()
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
    return wrapper, run_dir, env


def _write_smoke_evidence(path: pathlib.Path, *, commit: str = COMMIT) -> pathlib.Path:
    path.mkdir(parents=True)
    manifest = {
        "computation_commit": commit,
        "validation_commit": commit,
        "run_scope": {"kind": "bounded-smoke", "processed_row_count": 8},
    }
    (path / "dspark_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "dspark_metrics.json").write_text(
        json.dumps({"smoke_gate": {"passed": True}}), encoding="utf-8"
    )
    return path


def _run_wrapper(
    wrapper: pathlib.Path,
    run_dir: pathlib.Path,
    env: dict[str, str],
    *extra: str,
    smoke: bool = False,
) -> subprocess.CompletedProcess[str]:
    output_dir = run_dir.parent / "outputs/new-run"
    arguments = [str(run_dir), str(output_dir), *(["--smoke"] if smoke else []), *extra]
    return subprocess.run(
        [str(wrapper), *arguments],
        cwd=run_dir.parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_grid_wrapper_requires_smoke_evidence_for_full_runs(tmp_path: pathlib.Path) -> None:
    wrapper, run_dir, env = _smoke_wrapper(tmp_path)

    result = _run_wrapper(wrapper, run_dir, env)

    assert result.returncode == 2
    assert "Full DSpark runs require --smoke-evidence" in result.stderr


def test_grid_wrapper_rejects_smoke_evidence_on_smoke_runs(tmp_path: pathlib.Path) -> None:
    wrapper, run_dir, env = _smoke_wrapper(tmp_path)
    evidence = _write_smoke_evidence(tmp_path / "evidence")

    result = _run_wrapper(wrapper, run_dir, env, "--smoke-evidence", str(evidence), smoke=True)

    assert result.returncode == 2
    assert "--smoke-evidence applies only to full runs" in result.stderr


def test_grid_wrapper_rejects_smoke_evidence_from_another_commit(tmp_path: pathlib.Path) -> None:
    wrapper, run_dir, env = _smoke_wrapper(tmp_path)
    evidence = _write_smoke_evidence(tmp_path / "evidence", commit="b" * 40)

    result = _run_wrapper(wrapper, run_dir, env, "--smoke-evidence", str(evidence))

    assert result.returncode == 2
    assert "must record this checkout as both its computation and validation commit" in (
        result.stderr
    )


def test_grid_wrapper_rejects_smoke_evidence_inside_the_output_directory(
    tmp_path: pathlib.Path,
) -> None:
    wrapper, run_dir, env = _smoke_wrapper(tmp_path)
    inside_output = tmp_path / "outputs/new-run/smoke"

    result = _run_wrapper(wrapper, run_dir, env, "--smoke-evidence", str(inside_output))

    assert result.returncode == 2
    assert "must be outside the full-run output directory" in result.stderr


def test_grid_wrapper_rejects_relative_or_missing_smoke_evidence(
    tmp_path: pathlib.Path,
) -> None:
    wrapper, run_dir, env = _smoke_wrapper(tmp_path)

    relative = _run_wrapper(wrapper, run_dir, env, "--smoke-evidence", "smoke")
    missing = _run_wrapper(
        wrapper, run_dir, env, "--smoke-evidence", str(tmp_path / "missing-smoke")
    )

    assert relative.returncode == 2
    assert "--smoke-evidence must be an absolute path." in relative.stderr
    assert missing.returncode == 2
    assert "must name an existing smoke output directory" in missing.stderr


def test_grid_wrapper_usage_names_smoke_evidence() -> None:
    wrapper = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run-dspark-grid5000.sh"
    usage = next(
        line for line in wrapper.read_text(encoding="utf-8").splitlines() if "Usage:" in line
    )

    assert "--smoke-evidence /persistent/path/to/passing-smoke-output" in usage


def _smoke_precheck_program(wrapper: pathlib.Path) -> str:
    """Return the stdlib-only evidence pre-check embedded in the wrapper, verbatim."""
    text = wrapper.read_text(encoding="utf-8")
    match = re.search(
        r"python3 - \"\$RUN_DIR\" \"\$OUTPUT_DIR\" \"\$SMOKE_EVIDENCE\" \"\$COMMIT_SHA\" <<'PY'\n"
        r"(.*?)\nPY\n",
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _precheck_evidence(
    run_dir: pathlib.Path, evidence: pathlib.Path, *, gate_passed: bool = True
) -> None:
    evidence.mkdir(exist_ok=True)
    frozen = {
        name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        for name in ("frozen_sample.json", "candidate_labels.csv")
    }
    (evidence / "dspark_predictions.parquet").write_bytes(b"predictions")
    (evidence / "dspark_metrics.json").write_text(
        json.dumps({"smoke_gate": {"passed": gate_passed}}), encoding="utf-8"
    )
    manifest = {
        "computation_commit": COMMIT,
        "validation_commit": COMMIT,
        "frozen_manifest_sha256": hashlib.sha256(
            (run_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "run_scope": {"kind": "bounded-smoke", "processed_row_count": 8},
        "outputs_sha256": {
            **frozen,
            "dspark_predictions.parquet": hashlib.sha256(b"predictions").hexdigest(),
            "dspark_metrics.json": hashlib.sha256(
                (evidence / "dspark_metrics.json").read_bytes()
            ).hexdigest(),
        },
    }
    (evidence / "dspark_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_grid_wrapper_precheck_accepts_matching_passing_evidence(tmp_path: pathlib.Path) -> None:
    wrapper, run_dir, _ = _smoke_wrapper(tmp_path)
    program = _smoke_precheck_program(wrapper)
    evidence = tmp_path / "evidence"
    _precheck_evidence(run_dir, evidence)
    output_dir = tmp_path / "outputs/new-run"

    def check() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-", str(run_dir), str(output_dir), str(evidence), COMMIT],
            input=program,
            capture_output=True,
            text=True,
            check=False,
        )

    assert check().returncode == 0

    (evidence / "dspark_predictions.parquet").write_bytes(b"tampered")
    tampered = check()
    assert tampered.returncode == 2
    assert "output hash recorded for dspark_predictions.parquet" in tampered.stderr

    _precheck_evidence(run_dir, evidence, gate_passed=False)
    failed = check()
    assert failed.returncode == 2
    assert "does not record a passing smoke gate" in failed.stderr

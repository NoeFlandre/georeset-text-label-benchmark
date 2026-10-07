from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys


DOCS = Path(__file__).resolve().parents[1] / "docs" / "pilot.md"
COMMIT = "a" * 40
ATTEMPT = "pilot-attempt"


def _bash_blocks_after(marker: str) -> list[str]:
    text = DOCS.read_text(encoding="utf-8").split(marker, 1)[1]
    return re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL)


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _fake_tools(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    call_log = tmp_path / "calls.log"
    _write_executable(bin_dir / "hf", "#!/bin/sh\nexit 0\n")
    _write_executable(
        bin_dir / "git",
        '#!/bin/sh\nprintf \'%s\\n\' "$PILOT_TEST_COMMIT"\n',
    )
    _write_executable(
        bin_dir / "python",
        "#!/bin/sh\ncase \"$*\" in\n"
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
        "} >> \"$PILOT_TEST_CALL_LOG\"\n"
        'for arg in "$@"; do\n'
        '  [ "$arg" = "run-dspark-smoke" ] && exit "$PILOT_TEST_SMOKE_STATUS"\n'
        "done\nexit 0\n",
    )
    return bin_dir, call_log


def _environment(bin_dir: Path, call_log: Path, smoke_status: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "PILOT_TEST_ATTEMPT": ATTEMPT,
            "PILOT_TEST_CALL_LOG": str(call_log),
            "PILOT_TEST_COMMIT": COMMIT,
            "PILOT_TEST_REAL_PYTHON": sys.executable,
            "PILOT_TEST_SMOKE_STATUS": smoke_status,
            "HF_HOME": "/inherited/hf-home",
            "HF_HUB_CACHE": "/inherited/hf-hub-cache",
        }
    )
    return env


def _calls(call_log: Path) -> list[str]:
    if not call_log.exists():
        return []
    return [part.split("END\n", 1)[0] for part in call_log.read_text().split("CALL\n")[1:]]


def _cache_path(call: str) -> str:
    arguments = call.split("HF_HOME=", 1)[0].splitlines()
    return arguments[arguments.index("--model-cache") + 1]


def _write_passing_smoke(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "dspark_metrics.json").write_text(
        json.dumps({"smoke_gate": {"passed": True}}), encoding="utf-8"
    )


def test_direct_runbook_stops_after_failed_smoke_with_stale_passing_metrics(
    tmp_path: Path,
) -> None:
    snippet = _bash_blocks_after("Run the smoke and full inference commands only inside")[0]
    bin_dir, call_log = _fake_tools(tmp_path)
    run_root = tmp_path / "artifacts/source/pilot/runs"
    _write_passing_smoke(
        run_root / f"lfm2.5-2.6b-dspark-smoke-8-seed42-{COMMIT}"
    )
    _write_passing_smoke(
        run_root / f"lfm2.5-2.6b-dspark-smoke-8-seed42-{COMMIT}-{ATTEMPT}"
    )

    result = subprocess.run(
        ["bash", "-c", snippet],
        cwd=tmp_path,
        env=_environment(bin_dir, call_log, "23"),
        check=False,
        capture_output=True,
        text=True,
    )

    calls = _calls(call_log)
    assert result.returncode != 0
    assert len(calls) == 1
    assert "run-dspark-smoke" in calls[0]


def test_direct_runbook_uses_distinct_cache_roots_and_ignores_inherited_cache_paths(
    tmp_path: Path,
) -> None:
    snippet = _bash_blocks_after("Run the smoke and full inference commands only inside")[0]
    bin_dir, call_log = _fake_tools(tmp_path)
    smoke_out = tmp_path / "artifacts/source/pilot/runs" / (
        f"lfm2.5-2.6b-dspark-smoke-8-seed42-{COMMIT}-{ATTEMPT}"
    )
    _write_passing_smoke(smoke_out)

    result = subprocess.run(
        ["bash", "-c", snippet],
        cwd=tmp_path,
        env=_environment(bin_dir, call_log, "0"),
        check=False,
        capture_output=True,
        text=True,
    )

    calls = _calls(call_log)
    assert result.returncode == 0
    assert len(calls) == 2
    assert _cache_path(calls[0]).endswith("/smoke")
    assert _cache_path(calls[1]).endswith("/full")
    assert _cache_path(calls[0]) != _cache_path(calls[1])
    assert all("HF_HOME=<unset>" in call for call in calls)
    assert all("HF_HUB_CACHE=<unset>" in call for call in calls)


def test_grid_runbook_stops_after_failed_smoke_with_stale_passing_metrics(
    tmp_path: Path,
) -> None:
    blocks = _bash_blocks_after("### Grid" + chr(0x2019) + "5000 one-GPU execution")
    snippet = "\n".join(blocks[:2]).replace("/path/to/persistent", str(tmp_path))
    bin_dir, call_log = _fake_tools(tmp_path)
    wrapper = tmp_path / "scripts/run-dspark-grid5000.sh"
    _write_executable(
        wrapper,
        "#!/bin/sh\n{\n"
        "  printf 'CALL\\n'\n"
        "  printf '%s\\n' \"$@\"\n"
        "  printf 'HF_HOME=%s\\n' \"${HF_HOME-<unset>}\"\n"
        "  printf 'HF_HUB_CACHE=%s\\n' \"${HF_HUB_CACHE-<unset>}\"\n"
        "  printf 'END\\n'\n"
        "} >> \"$PILOT_TEST_CALL_LOG\"\n"
        'for arg in "$@"; do\n'
        '  [ "$arg" = "--smoke" ] && exit "$PILOT_TEST_SMOKE_STATUS"\n'
        "done\nexit 0\n",
    )
    _write_passing_smoke(
        tmp_path / "pilot/runs" / f"lfm2.5-2.6b-dspark-smoke-8-seed42-{COMMIT}"
    )
    _write_passing_smoke(
        tmp_path
        / "pilot/runs"
        / f"lfm2.5-2.6b-dspark-smoke-8-seed42-{COMMIT}-{ATTEMPT}"
    )

    result = subprocess.run(
        ["bash", "-c", snippet],
        cwd=tmp_path,
        env=_environment(bin_dir, call_log, "23"),
        check=False,
        capture_output=True,
        text=True,
    )

    calls = _calls(call_log)
    assert result.returncode != 0
    assert len(calls) == 1
    assert "--smoke" in calls[0]

"""Command-line dispatch tests without network access."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import Any, cast

import pytest

from georeset_text_label_benchmark import cli


def test_cli_run_uses_default_source_and_prints_summary(monkeypatch, capsys, tmp_path) -> None:
    expected = {"status": "complete", "stages": {"retained_overlap_rows": 3}}
    received: dict[str, object] = {}
    monkeypatch.setattr(cli.DescriptionSource, "from_hub", lambda: "test-source")

    def fake_run(source, output, *, computation_commit, validation_commit):
        received["source"] = source
        received["output"] = output
        received["computation_commit"] = computation_commit
        received["validation_commit"] = validation_commit
        return expected

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    output = tmp_path / "result"

    commit = "a" * 40
    assert (
        cli.main(
            [
                "run",
                "--output",
                str(output),
                "--computation-commit",
                commit,
                "--validation-commit",
                commit,
            ]
        )
        == 0
    )
    assert received == {
        "source": "test-source",
        "output": output,
        "computation_commit": commit,
        "validation_commit": commit,
    }
    printed = capsys.readouterr().out
    assert json.loads(printed) == expected
    assert printed == json.dumps(expected, indent=2, sort_keys=True) + "\n"


def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main([])


def test_cli_requires_both_provenance_commits() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["run"])


def test_cli_parser_help_and_argument_contract() -> None:
    parser = cli._parser()
    run_parser = cast(Any, parser._subparsers)._group_actions[0].choices["run"]
    commit = "a" * 40

    assert parser.format_help() == (
        "usage: georeset-benchmark [-h] {run} ...\n\n"
        "positional arguments:\n"
        "  {run}\n"
        "    run       compute overlap from pinned public Hub snapshots\n\n"
        "options:\n"
        "  -h, --help  show this help message and exit\n"
    )
    assert run_parser.format_help() == (
        "usage: georeset-benchmark run [-h] [--output OUTPUT] --computation-commit\n"
        "                              COMPUTATION_COMMIT --validation-commit\n"
        "                              VALIDATION_COMMIT\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --output OUTPUT       new directory for overlap.parquet, summary.json, and\n"
        "                        manifest.json\n"
        "  --computation-commit COMPUTATION_COMMIT\n"
        "                        40-character Git commit SHA of the code used to\n"
        "                        generate the artifacts\n"
        "  --validation-commit VALIDATION_COMMIT\n"
        "                        40-character Git commit SHA of the code used to\n"
        "                        validate the artifacts\n"
    )
    assert parser.parse_args(
        ["run", "--computation-commit", commit, "--validation-commit", commit]
    ) == argparse.Namespace(
        command="run",
        output=Path("artifacts/description-eunis-overlap"),
        computation_commit=commit,
        validation_commit=commit,
    )


def test_module_entry_point_delegates_to_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "main", lambda: 0)

    with pytest.raises(SystemExit, match="0"):
        runpy.run_module("georeset_text_label_benchmark.__main__", run_name="__main__")

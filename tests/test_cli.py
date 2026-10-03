"""Command-line dispatch tests without network access."""

from __future__ import annotations

import json

import pytest

from georeset_text_label_benchmark import cli


def test_cli_run_uses_default_source_and_prints_summary(monkeypatch, capsys, tmp_path) -> None:
    expected = {"status": "complete", "stages": {"retained_overlap_rows": 3}}
    received: dict[str, object] = {}
    monkeypatch.setattr(cli.DescriptionSource, "from_hub", lambda: "test-source")

    def fake_run(source, output):
        received["source"] = source
        received["output"] = output
        return expected

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    output = tmp_path / "result"

    assert cli.main(["run", "--output", str(output)]) == 0
    assert received == {"source": "test-source", "output": output}
    assert json.loads(capsys.readouterr().out) == expected


def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main([])

"""Tests for the repository's own fail-closed code-quality gates."""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from georeset_text_label_benchmark.quality import mutation
from georeset_text_label_benchmark.quality.crap import (
    _complexity,
    _coverage_file,
    _file_failures,
    _function_crap,
    _function_nodes,
    _source_files,
    _statement_lines,
)
from georeset_text_label_benchmark.quality.crap import (
    main as check_crap,
)
from georeset_text_label_benchmark.quality.mutation import (
    EQUIVALENT_MUTANTS,
    _equivalent_survivors,
    _failures,
    _killed_count,
    _parse_results,
    _report_results,
)
from georeset_text_label_benchmark.quality.mutation import (
    main as check_mutations,
)


def test_mutmut_copies_docs_directory_needed_by_runbook_tests() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert "docs/" in config["tool"]["mutmut"]["also_copy"]


def _function(source: str) -> ast.FunctionDef:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def f(items):\n return [item for item in items if item > 0]\n", 3),
        ("def f(a, b):\n return a and b\n", 2),
        ("def f(a, b):\n return a if b else None\n", 2),
        (
            "def f(value):\n match value:\n  case 1: return 1\n  case 2: return 2\n  case _: return 0\n",
            3,
        ),
        ("def f(value):\n match value:\n  case _: return 0\n", 1),
    ],
)
def test_complexity_uses_conventional_decision_points(source: str, expected: int) -> None:
    assert _complexity(_function(source)) == expected


def test_function_discovery_and_scoring_ignore_nested_scopes() -> None:
    tree = ast.parse("def outer():\n return 1\n def inner():\n  return 2\n return 3\n")
    functions = _function_nodes(tree)

    assert [function.name for function in functions] == ["outer", "inner"]
    assert _statement_lines(functions[0]) == {2, 5}
    assert _complexity(functions[0]) == 1
    assert _function_crap(functions[0], {2, 5}) == 1


def test_crap_formula_penalizes_uncovered_statements() -> None:
    function = _function("def f(value):\n    if value:\n        return 1\n    return 0\n")

    assert _function_crap(function, {2}) == pytest.approx(3.185185185185185)


def test_file_failure_report_is_ordered_exact_and_reads_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_file = tmp_path / "module.py"
    source_file.write_text(
        "def zed(value):\n"
        "    if value:\n"
        "        return 1\n"
        "    return 0\n"
        "def alpha():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    original_read_text = Path.read_text
    encodings: list[str | None] = []

    def record_read(path: Path, *args: Any, **kwargs: Any) -> str:
        encodings.append(kwargs.get("encoding"))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_read)
    coverage = {
        "files": {source_file.as_posix(): {"executed_lines": []}},
    }

    failures = _file_failures(source_file, coverage)

    assert encodings == ["utf-8"]
    assert failures == [f"{source_file}:1 zed: CRAP=6.000"]
    assert capsys.readouterr().out == (
        f"{source_file}:1 zed: CRAP=6.000\n{source_file}:5 alpha: CRAP=2.000\n"
    )


def test_crap_main_reads_the_coverage_report_as_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "module.py"
    source_file.write_text("def helper():\n    return 1\n", encoding="utf-8")
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps({"files": {source_file.as_posix(): {"executed_lines": [2]}}}),
        encoding="utf-8",
    )
    original_read_text = Path.read_text
    read_encodings: list[tuple[Path, str | None]] = []

    def record_read(path: Path, *args: Any, **kwargs: Any) -> str:
        read_encodings.append((path, kwargs.get("encoding")))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_read)

    assert check_crap(report, (source_dir,)) == 0
    assert read_encodings == [(report, "utf-8"), (source_file, "utf-8")]


def test_coverage_file_rejects_missing_file_and_missing_mapping() -> None:
    with pytest.raises(ValueError, match=r"^coverage report does not contain src/missing\.py$"):
        _coverage_file({"files": {}}, Path("src/missing.py"))
    with pytest.raises(ValueError, match=r"^coverage report is missing its files mapping$"):
        _coverage_file({}, Path("src/missing.py"))


def test_source_file_discovery_fails_closed_on_missing_or_empty_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"^owned source directory missing:") as caught:
        _source_files([tmp_path / "second", tmp_path / "first"])
    assert str(caught.value) == (
        f"owned source directory missing: {tmp_path / 'first'}, {tmp_path / 'second'}"
    )
    with pytest.raises(ValueError, match=r"^no owned Python source files found for CRAP scoring$"):
        _source_files([tmp_path])


def test_crap_gate_fails_at_exactly_six_even_with_full_coverage(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "complex.py"
    source_file.write_text(
        "def six(items):\n"
        "    for item in items:\n"
        "        if item:\n"
        "            pass\n"
        "    while False:\n"
        "        break\n"
        "    if items:\n"
        "        pass\n"
        "    if not items:\n"
        "        pass\n"
        "    return len(items)\n",
        encoding="utf-8",
    )
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps({"files": {source_file.as_posix(): {"executed_lines": list(range(2, 12))}}}),
        encoding="utf-8",
    )

    assert check_crap(report, (source_dir,)) == 1


def test_crap_gate_rejects_a_report_missing_an_owned_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "module.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain"):
        check_crap(report, (source_dir,))


def test_crap_gate_passes_a_fully_covered_simple_source_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "module.py"
    source_file.write_text("def helper():\n    return 1\n", encoding="utf-8")
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps({"files": {source_file.as_posix(): {"executed_lines": [2]}}}),
        encoding="utf-8",
    )

    assert check_crap(report, (source_dir,)) == 0


def test_crap_gate_prints_function_score_and_threshold_diagnostic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "module.py"
    source_file.write_text(
        "def helper(value):\n    if value:\n        return 1\n    return 0\n", encoding="utf-8"
    )
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps({"files": {source_file.as_posix(): {"executed_lines": []}}}), encoding="utf-8"
    )

    assert check_crap(report, (source_dir,)) == 1
    assert capsys.readouterr().out == (
        f"{source_file}:1 helper: CRAP=6.000\n1 function(s) must have CRAP strictly below 6\n"
    )


def test_mutation_parser_records_every_status_and_rejects_unknown_lines() -> None:
    output = "mutant_a: killed\nmutant_b: survived\nmutant_c: no tests\nmutant_d: timeout\n"

    assert _parse_results(output) == {
        "mutant_a": "killed",
        "mutant_b": "survived",
        "mutant_c": "no tests",
        "mutant_d": "timeout",
    }
    with pytest.raises(ValueError, match="unrecognized"):
        _parse_results("future mutant state")


@pytest.mark.parametrize(
    ("output", "error"),
    [
        ("", "mutmut returned no mutation results"),
        ("same: killed\nsame: killed", "mutmut returned duplicate mutant identifiers"),
    ],
)
def test_mutation_parser_rejects_empty_or_duplicate_results(output: str, error: str) -> None:
    with pytest.raises(ValueError, match=r"^mutmut returned") as caught:
        _parse_results(output)
    assert str(caught.value) == error


def test_mutation_gate_rejects_every_status_other_than_killed() -> None:
    assert _failures({"killed": "killed"}, {}) == []
    assert _failures({"z": "timeout", "a": "survived", "b": "no tests"}, {}) == [
        "a: survived",
        "b: no tests",
        "z: timeout",
    ]


@pytest.mark.parametrize("name", list(EQUIVALENT_MUTANTS))
def test_mutation_gate_waives_only_documented_equivalent_survivors(name: str) -> None:
    exemption = EQUIVALENT_MUTANTS[name]
    assert exemption.rationale
    assert len(exemption.fingerprint) == 64
    assert _failures({name: "survived"}, {name: exemption.fingerprint}) == []
    assert _failures({name: "survived"}, {name: "0" * 64}) == [f"{name}: survived"]
    assert _failures({name: "no tests"}, {name: exemption.fingerprint}) == [f"{name}: no tests"]


def test_mutation_summary_distinguishes_killed_and_exact_equivalent_results() -> None:
    first = next(iter(EQUIVALENT_MUTANTS))
    second = next(reversed(EQUIVALENT_MUTANTS))
    results = {"killed": "killed", first: "survived", second: "killed"}

    assert _killed_count(results) == 2
    assert _equivalent_survivors(results, {first: EQUIVALENT_MUTANTS[first].fingerprint}) == [first]


def test_mutation_gate_reports_exact_equivalence_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    name = next(iter(EQUIVALENT_MUTANTS))
    exemption = EQUIVALENT_MUTANTS[name]

    assert _report_results({name: "survived"}, {name: exemption.fingerprint}) == 0
    assert capsys.readouterr().out == (
        "Mutation results: 0/1 killed\n"
        f"Equivalent mutant documented: {name}: {exemption.rationale} "
        f"(diff sha256: {exemption.fingerprint})\n"
    )

    assert _report_results({name: "killed"}, {}) == 0
    assert capsys.readouterr().out == "Mutation results: 1/1 killed\n"


def test_mutation_gate_reads_fingerprint_for_surviving_reviewed_patch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    name = "example.x_gate__mutmut_1"
    diff = (
        f"# {name}: survived\n--- src/example.py\n+++ src/example.py\n"
        "@@ -14,3 +14,3 @@\n def gate(value):\n-    return False\n+    return bool(value)\n"
    )
    fingerprint = mutation._mutation_fingerprint(diff, name)
    monkeypatch.setattr(
        mutation,
        "REVIEWED_EXEMPTIONS",
        {name: mutation.MutationExemption(fingerprint, "fixture reviewed as equivalent")},
    )
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(args: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((args, kwargs))
        output = f"{name}: survived\n" if args[1] == "results" else diff
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(mutation.subprocess, "run", fake_run)

    assert check_mutations() == 0
    subprocess_kwargs = {"capture_output": True, "text": True, "check": False}
    assert calls == [
        (["mutmut", "results", "--all=true"], subprocess_kwargs),
        (["mutmut", "show", name], subprocess_kwargs),
    ]
    assert capsys.readouterr().out.endswith(f"(diff sha256: {fingerprint})\n")


def test_mutation_fingerprint_binds_the_reviewed_patch_and_hunk_location() -> None:
    name = "example.x_gate__mutmut_1"
    reviewed = (
        f"# {name}: survived\n--- src/example.py\n+++ src/example.py\n"
        "@@ -14,3 +14,3 @@\n def gate(value):\n-    return False\n+    return bool(value)\n"
    )
    shifted = reviewed.replace("@@ -14,3 +14,3 @@", "@@ -114,3 +114,3 @@")
    changed = reviewed.replace("+    return bool(value)", "+    return value")

    assert mutation._mutation_fingerprint(reviewed, name) == (
        "3433e77824bf453d1c51a26790c539ace180f67c49c378a546d73c12305183fc"
    )
    assert mutation._mutation_fingerprint(reviewed, name) != mutation._mutation_fingerprint(
        shifted, name
    )
    assert mutation._mutation_fingerprint(reviewed, name) != mutation._mutation_fingerprint(
        changed, name
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("--- src/example.py", True),
        ("+++ src/example.py", True),
        ("@@ -1 +1 @@", True),
        ("-removed", True),
        ("+added", True),
        (" context", True),
        ("---src/example.py", False),
        ("+++src/example.py", False),
        ("\\ No newline at end of file", False),
    ],
)
def test_valid_mutation_diff_line_checks_headers_and_patch_lines(line: str, expected: bool) -> None:
    assert mutation._valid_mutation_diff_line(line) is expected


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["+added"], True),
        (["-removed"], True),
        (["--- old", "+++ new"], False),
        ([" context"], False),
        ([], False),
    ],
)
def test_mutation_diff_requires_added_or_removed_code(lines: list[str], expected: bool) -> None:
    assert mutation._has_changed_diff_content(lines) is expected


def test_mutation_fingerprint_rejects_invalid_or_incomplete_diffs() -> None:
    name = "example.x_gate__mutmut_1"
    malformed = [
        ("", "unexpected header"),
        (
            f"# {name}: killed\n--- src/example.py\n+++ src/example.py\n"
            "@@ -14,3 +14,3 @@\n-return False\n+return True\n",
            "unexpected header",
        ),
        (
            f"# {name}: survived\n--- src/example.py\n+++ src/example.py\n"
            "@@ -14,3 +14,3 @@\n\\ No newline at end of file\n",
            "invalid diff",
        ),
        (
            f"# {name}: survived\n--- src/example.py\n+++ src/example.py\n"
            "@@ -14,3 +14,3 @@\n unchanged line\n",
            "incomplete diff",
        ),
        (
            f"# {name}: survived\n+++ src/example.py\n"
            "@@ -14,3 +14,3 @@\n-return False\n+return True\n",
            "incomplete diff",
        ),
    ]

    for diff, message in malformed:
        with pytest.raises(ValueError, match=message):
            mutation._mutation_fingerprint(diff, name)


def test_mutation_gate_requires_the_exact_reviewed_diff_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "example.x_gate__mutmut_1"
    reviewed = (
        f"# {name}: survived\n--- src/example.py\n+++ src/example.py\n"
        "@@ -14,3 +14,3 @@\n def gate(value):\n-    return False\n+    return bool(value)\n"
    )
    fingerprint = mutation._mutation_fingerprint(reviewed, name)
    monkeypatch.setattr(
        mutation,
        "REVIEWED_EXEMPTIONS",
        {name: mutation.MutationExemption(fingerprint, "fixture reviewed as equivalent")},
    )

    assert _failures({name: "survived"}, {name: fingerprint}) == []
    assert _failures({name: "survived"}, {name: "0" * 64}) == [f"{name}: survived"]
    assert _failures({name: "survived"}, {}) == [f"{name}: survived"]


@pytest.mark.parametrize("status", ["no tests", "timeout", "unknown"])
def test_mutation_fingerprint_never_waives_invalid_statuses(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    name = "example.x_gate__mutmut_1"
    fingerprint = "a" * 64
    monkeypatch.setattr(
        mutation,
        "REVIEWED_EXEMPTIONS",
        {name: mutation.MutationExemption(fingerprint, "fixture reviewed as equivalent")},
    )

    assert _failures({name: status}, {name: fingerprint}) == [f"{name}: {status}"]


def test_survivor_names_includes_reviewed_and_unreviewed_mutants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mutation,
        "REVIEWED_EXEMPTIONS",
        {"reviewed": mutation.MutationExemption("a" * 64, "fixture")},
    )

    assert mutation._survivor_names(
        {"reviewed": "survived", "killed": "killed", "unreviewed": "survived"}
    ) == ["reviewed", "unreviewed"]


def test_mutation_gate_fails_when_mutmut_command_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "georeset_text_label_benchmark.quality.mutation.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="mutmut failed"),
    )

    assert check_mutations() == 1
    assert capsys.readouterr().out == "mutmut failed\n"


def test_mutation_gate_fails_closed_when_survivor_diff_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    name = next(iter(EQUIVALENT_MUTANTS))

    def fake_run(args: list[str], **_kwargs: Any) -> SimpleNamespace:
        if args[1] == "results":
            return SimpleNamespace(returncode=0, stdout=f"{name}: survived\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="show denied")

    monkeypatch.setattr(mutation.subprocess, "run", fake_run)

    assert check_mutations() == 1
    assert capsys.readouterr().out == f"mutmut show failed for {name}: show denied\n"


@pytest.mark.parametrize(
    ("stdout", "expected", "reported"),
    [
        ("only: killed\n", 0, "Mutation results: 1/1 killed\n"),
        (
            "alive: survived\n",
            1,
            None,
        ),
        (
            "untested: no tests\n",
            1,
            "Mutation results: 0/1 killed\nUnresolved mutation results:\n  untested: no tests\n",
        ),
        ("", 1, "mutmut returned no mutation results\n"),
    ],
)
def test_mutation_gate_checks_run_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stdout: str,
    expected: int,
    reported: str | None,
) -> None:
    patch = (
        "# alive: survived\n--- src/example.py\n+++ src/example.py\n"
        "@@ -14,3 +14,3 @@\n def gate(value):\n-    return False\n+    return bool(value)\n"
    )
    fingerprint = mutation._mutation_fingerprint(patch, "alive")

    def fake_run(args: list[str], **_kwargs: Any) -> SimpleNamespace:
        output = stdout if args[1] == "results" else patch
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(mutation.subprocess, "run", fake_run)

    assert check_mutations() == expected
    actual = capsys.readouterr().out
    if reported is None:
        reported = (
            "Unresolved mutation patch for alive:\n"
            f"{patch}"
            f"Mutation fingerprint: {fingerprint}\n"
            "Mutation results: 0/1 killed\n"
            "Unresolved mutation results:\n"
            "  alive: survived\n"
        )
    assert actual == reported


def test_mutation_gate_invokes_mutmut_with_all_results_and_reports_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="exact: killed\n", stderr="")

    monkeypatch.setattr("georeset_text_label_benchmark.quality.mutation.subprocess.run", fake_run)

    assert check_mutations() == 0
    assert calls == [
        (
            (["mutmut", "results", "--all=true"],),
            {"capture_output": True, "text": True, "check": False},
        )
    ]
    assert capsys.readouterr().out == "Mutation results: 1/1 killed\n"


def test_mutation_gate_reports_patch_and_fingerprint_for_unreviewed_survivor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    name = "example.x_gate__mutmut_1"
    patch = (
        f"# {name}: survived\n--- src/example.py\n+++ src/example.py\n"
        "@@ -14,3 +14,3 @@\n def gate(value):\n-    return False\n+    return bool(value)\n"
    )
    fingerprint = mutation._mutation_fingerprint(patch, name)

    def fake_run(args: list[str], **kwargs: Any) -> SimpleNamespace:
        output = f"{name}: survived\n" if args[1] == "results" else patch
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(mutation.subprocess, "run", fake_run)

    assert check_mutations() == 1
    output = capsys.readouterr().out
    assert patch in output
    assert f"Mutation fingerprint: {fingerprint}" in output
    assert f"{name}: survived" in output

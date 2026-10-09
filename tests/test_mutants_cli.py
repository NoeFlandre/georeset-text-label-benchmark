"""Exact-output tests for the pilot CLI and candidate-label validation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, cast

import pytest

from georeset_text_label_benchmark.pilot import cli
from georeset_text_label_benchmark.pilot.sampling import build_candidate_labels

COMMIT = "a" * 40
VALIDATION_COMMIT = "b" * 40
LOWER_SHA256 = "0123456789abcdef" * 4


def _subcommand(name: str) -> argparse.ArgumentParser:
    commands = cast(Any, cli._parser()._subparsers)._group_actions[0]
    return cast(argparse.ArgumentParser, commands.choices[name])


def _exact(message: str) -> str:
    return rf"^{re.escape(message)}$"


def _row(code: str = "T11", name: str = "Temperate forest") -> dict[str, Any]:
    return {"eunis_code": code, "eunis_name": name}


def _taxonomy(
    name: str = "Temperate forest", description: str = "Forest description."
) -> dict[str, str]:
    return {
        "name": name,
        "description": description,
        "classification_release": "terrestrial-2021",
        "classification_source": "EEA EUNIS terrestrial habitat classification",
        "source_sha256": LOWER_SHA256,
        "license": "CC-BY-4.0",
    }


# --- argparse help and argument contract -----------------------------------------------


def test_freeze_help_lists_exactly_its_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "80")

    assert _subcommand("freeze").format_help() == (
        "usage: georeset-pilot freeze [-h] --labelled-parquet LABELLED_PARQUET\n"
        "                             --pipeline-manifest PIPELINE_MANIFEST\n"
        "                             [--candidate-csv CANDIDATE_CSV]\n"
        "                             [--output-dir OUTPUT_DIR] [--per-group PER_GROUP]\n"
        "                             [--seed SEED]\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --labelled-parquet LABELLED_PARQUET\n"
        "  --pipeline-manifest PIPELINE_MANIFEST\n"
        "  --candidate-csv CANDIDATE_CSV\n"
        "  --output-dir OUTPUT_DIR\n"
        "  --per-group PER_GROUP\n"
        "  --seed SEED\n"
    )


def test_run_dspark_help_lists_exactly_its_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "80")

    assert _subcommand("run-dspark").format_help() == (
        "usage: georeset-pilot run-dspark [-h] [--run-dir RUN_DIR]\n"
        "                                 [--output-dir OUTPUT_DIR]\n"
        "                                 [--model-cache MODEL_CACHE]\n"
        "                                 [--model-cache-seed MODEL_CACHE_SEED]\n"
        "                                 --computation-commit COMPUTATION_COMMIT\n"
        "                                 --validation-commit VALIDATION_COMMIT\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --run-dir RUN_DIR\n"
        "  --output-dir OUTPUT_DIR\n"
        "  --model-cache MODEL_CACHE\n"
        "  --model-cache-seed MODEL_CACHE_SEED\n"
        "  --computation-commit COMPUTATION_COMMIT\n"
        "  --validation-commit VALIDATION_COMMIT\n"
    )


def test_run_dspark_smoke_help_lists_exactly_its_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "80")

    assert _subcommand("run-dspark-smoke").format_help() == (
        "usage: georeset-pilot run-dspark-smoke [-h] [--run-dir RUN_DIR] --output-dir\n"
        "                                       OUTPUT_DIR [--model-cache MODEL_CACHE]\n"
        "                                       [--model-cache-seed MODEL_CACHE_SEED]\n"
        "                                       --computation-commit COMPUTATION_COMMIT\n"
        "                                       --validation-commit VALIDATION_COMMIT\n\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --run-dir RUN_DIR\n"
        "  --output-dir OUTPUT_DIR\n"
        "  --model-cache MODEL_CACHE\n"
        "  --model-cache-seed MODEL_CACHE_SEED\n"
        "  --computation-commit COMPUTATION_COMMIT\n"
        "  --validation-commit VALIDATION_COMMIT\n"
    )


def test_freeze_parses_every_flag_into_its_own_field() -> None:
    args = cli._parser().parse_args(
        [
            "freeze",
            "--labelled-parquet",
            "labelled.parquet",
            "--pipeline-manifest",
            "manifest.json",
            "--candidate-csv",
            "candidates.csv",
            "--output-dir",
            "run-out",
            "--per-group",
            "7",
            "--seed",
            "9",
        ]
    )

    assert args == argparse.Namespace(
        command="freeze",
        labelled_parquet=Path("labelled.parquet"),
        pipeline_manifest=Path("manifest.json"),
        candidate_csv=Path("candidates.csv"),
        output_dir=Path("run-out"),
        per_group=7,
        seed=9,
    )
    assert type(args.per_group) is int
    assert type(args.seed) is int


def test_freeze_defaults_name_the_pinned_candidate_table_and_pilot_run_directory() -> None:
    args = cli._parser().parse_args(
        ["freeze", "--labelled-parquet", "labelled.parquet", "--pipeline-manifest", "m.json"]
    )

    assert args == argparse.Namespace(
        command="freeze",
        labelled_parquet=Path("labelled.parquet"),
        pipeline_manifest=Path("m.json"),
        candidate_csv=Path("pilot_data/eunis_candidate_labels.csv"),
        output_dir=Path("artifacts/pilot-100-seed42"),
        per_group=50,
        seed=42,
    )
    assert type(args.per_group) is int


def test_freeze_error_names_both_missing_inputs_on_the_freeze_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli._parser().parse_args(["freeze"])

    assert error.value.code == 2
    assert capsys.readouterr().err.splitlines()[-1] == (
        "georeset-pilot freeze: error: the following arguments are required: "
        "--labelled-parquet, --pipeline-manifest"
    )


def test_main_without_a_command_reports_the_command_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([])

    assert error.value.code == 2
    assert capsys.readouterr().err.splitlines()[-1] == (
        "georeset-pilot: error: the following arguments are required: command"
    )


def test_main_rejects_an_unknown_command_with_the_valid_choices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["freeze-typo"])

    assert error.value.code == 2
    # argparse quotes the choices differently across CPython 3.12 patch releases; accept both exact forms.
    assert capsys.readouterr().err.splitlines()[-1] in {
        "georeset-pilot: error: argument command: invalid choice: 'freeze-typo' "
        "(choose from freeze, run-dspark, run-dspark-smoke)",
        "georeset-pilot: error: argument command: invalid choice: 'freeze-typo' "
        "(choose from 'freeze', 'run-dspark', 'run-dspark-smoke')",
    }


def test_run_dspark_defaults_name_the_pilot_run_and_model_cache() -> None:
    args = cli._parser().parse_args(
        [
            "run-dspark",
            "--computation-commit",
            COMMIT,
            "--validation-commit",
            VALIDATION_COMMIT,
        ]
    )

    assert args == argparse.Namespace(
        command="run-dspark",
        run_dir=Path("artifacts/pilot-100-seed42"),
        output_dir=None,
        model_cache=Path(".cache/model-dspark"),
        model_cache_seed=None,
        computation_commit=COMMIT,
        validation_commit=VALIDATION_COMMIT,
    )


def test_run_dspark_smoke_defaults_name_the_pilot_run_and_model_cache() -> None:
    args = cli._parser().parse_args(
        [
            "run-dspark-smoke",
            "--output-dir",
            "smoke-out",
            "--computation-commit",
            COMMIT,
            "--validation-commit",
            VALIDATION_COMMIT,
        ]
    )

    assert args == argparse.Namespace(
        command="run-dspark-smoke",
        run_dir=Path("artifacts/pilot-100-seed42"),
        output_dir=Path("smoke-out"),
        model_cache=Path(".cache/model-dspark"),
        model_cache_seed=None,
        computation_commit=COMMIT,
        validation_commit=VALIDATION_COMMIT,
    )


# --- main() dispatch and output ---------------------------------------------------------


def test_main_freeze_passes_inputs_in_order_and_prints_sorted_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_freeze(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"zeta": 1, "alpha": "café"}

    monkeypatch.setattr(cli, "freeze_sample", fake_freeze)

    exit_code = cli.main(
        [
            "freeze",
            "--labelled-parquet",
            "labelled.parquet",
            "--pipeline-manifest",
            "manifest.json",
            "--candidate-csv",
            "candidates.csv",
            "--output-dir",
            "run-out",
            "--per-group",
            "7",
            "--seed",
            "9",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            (
                Path("labelled.parquet"),
                Path("candidates.csv"),
                Path("manifest.json"),
                Path("run-out"),
            ),
            {"per_group": 7, "seed": 9},
        )
    ]
    assert capsys.readouterr().out == '{\n  "alpha": "café",\n  "zeta": 1\n}\n'


def test_main_run_dspark_exits_zero_even_when_the_smoke_gate_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"smoke_gate": {"passed": False}}

    monkeypatch.setattr(cli, "run_dspark_pilot", fake_run)

    exit_code = cli.main(
        [
            "run-dspark",
            "--computation-commit",
            COMMIT,
            "--validation-commit",
            VALIDATION_COMMIT,
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == '{\n  "smoke_gate": {\n    "passed": false\n  }\n}\n'


def test_main_smoke_exit_status_is_one_exactly_when_the_gate_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli, "run_dspark_pilot", lambda *args, **kwargs: {"smoke_gate": {"passed": False}}
    )
    arguments = [
        "run-dspark-smoke",
        "--output-dir",
        "smoke-out",
        "--computation-commit",
        COMMIT,
        "--validation-commit",
        VALIDATION_COMMIT,
    ]

    assert cli.main(arguments) == 1
    assert capsys.readouterr().out == '{\n  "smoke_gate": {\n    "passed": false\n  }\n}\n'


# --- candidate label validation ----------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "row"),
    [
        ("eunis_code", {"eunis_name": "Temperate forest"}),
        ("eunis_code", {"eunis_code": "", "eunis_name": "Temperate forest"}),
        ("eunis_code", {"eunis_code": 11, "eunis_name": "Temperate forest"}),
        ("eunis_name", {"eunis_code": "T11"}),
        ("eunis_name", {"eunis_code": "T11", "eunis_name": ""}),
        ("eunis_name", {"eunis_code": "T11", "eunis_name": None}),
    ],
)
def test_observed_rows_need_non_empty_string_code_and_name(field: str, row: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=_exact(f"{field} must be a non-empty string")):
        build_candidate_labels([row], {"T11": _taxonomy()})


def test_observed_rows_with_one_code_must_agree_on_its_name() -> None:
    rows = [_row(name="Temperate forest"), _row(name="Other")]

    with pytest.raises(ValueError, match=_exact("inconsistent EUNIS name for T11")):
        build_candidate_labels(rows, {"T11": _taxonomy()})


def test_repeated_observations_with_the_same_name_yield_one_candidate() -> None:
    rows = [_row(), _row()]

    candidates = build_candidate_labels(rows, {"T11": _taxonomy()})

    assert [candidate["eunis_code"] for candidate in candidates] == ["T11"]


def test_candidates_are_sorted_by_code_regardless_of_row_order() -> None:
    rows = [_row("T11", "Temperate forest"), _row("MA221", "Saltmarsh"), _row("ZZ9", "Other")]
    taxonomy = {
        "T11": _taxonomy("Temperate forest"),
        "MA221": _taxonomy("Saltmarsh"),
        "ZZ9": _taxonomy("Other"),
    }

    candidates = build_candidate_labels(rows, taxonomy)

    assert [candidate["eunis_code"] for candidate in candidates] == ["MA221", "T11", "ZZ9"]


def test_taxonomy_must_cover_exactly_the_observed_codes() -> None:
    message = "taxonomy codes do not match observed EUNIS codes"
    with pytest.raises(ValueError, match=_exact(message)):
        build_candidate_labels([_row()], {})
    with pytest.raises(ValueError, match=_exact(message)):
        build_candidate_labels([_row()], {"T11": _taxonomy(), "MA221": _taxonomy("Saltmarsh")})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", None),
        ("name", ""),
        ("description", None),
        ("classification_release", ""),
        ("classification_source", 7),
        ("source_sha256", None),
        ("license", ""),
    ],
)
def test_taxonomy_entries_need_non_empty_string_fields(field: str, value: Any) -> None:
    details = {**_taxonomy(), field: value}

    with pytest.raises(ValueError, match=_exact(f"{field} must be a non-empty string")):
        build_candidate_labels([_row()], {"T11": details})


def test_taxonomy_name_must_match_the_observed_polygon_name_exactly() -> None:
    with pytest.raises(ValueError, match=_exact("EUNIS name mismatch for T11")):
        build_candidate_labels(
            [_row(name="Temperate forest")], {"T11": _taxonomy("Temperate  forest")}
        )


@pytest.mark.parametrize("description", [" ", "\t", "\n  \n"])
def test_taxonomy_description_must_contain_text(description: str) -> None:
    with pytest.raises(ValueError, match=_exact("description must be non-empty for T11")):
        build_candidate_labels([_row()], {"T11": _taxonomy(description=description)})


@pytest.mark.parametrize(
    "source_hash",
    [
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "X" * 64,
        "g" * 64,
        LOWER_SHA256[:-1] + "F",
    ],
)
def test_source_sha256_must_be_exactly_64_lowercase_hex_digits(source_hash: str) -> None:
    details = {**_taxonomy(), "source_sha256": source_hash}

    with pytest.raises(
        ValueError, match=_exact("source_sha256 must be a lowercase SHA-256 for T11")
    ):
        build_candidate_labels([_row()], {"T11": details})


def test_source_sha256_accepts_a_lowercase_digest_and_keeps_it_verbatim() -> None:
    candidates = build_candidate_labels([_row()], {"T11": _taxonomy()})

    assert candidates[0]["source_sha256"] == LOWER_SHA256


def test_candidate_record_holds_exactly_the_documented_fields() -> None:
    candidates = build_candidate_labels([_row()], {"T11": _taxonomy()})

    assert candidates == [
        {
            "eunis_code": "T11",
            "eunis_name": "Temperate forest",
            "eunis_description": "Forest description.",
            "candidate_text": "Temperate forest\nForest description.",
            "classification_release": "terrestrial-2021",
            "classification_source": "EEA EUNIS terrestrial habitat classification",
            "source_sha256": LOWER_SHA256,
            "license": "CC-BY-4.0",
        }
    ]

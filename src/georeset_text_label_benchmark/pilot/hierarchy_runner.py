"""Validate historical pilot artifacts and publish derived hierarchy metrics."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from georeset_text_label_benchmark.pilot.hierarchy import hierarchical_report
from georeset_text_label_benchmark.pilot.publication import publish_directory, staged_directory
from georeset_text_label_benchmark.pilot.runner import (
    _sha256_json,
    _validate_commit,
    _write_json_exclusive,
    sha256_file,
)

PREDICTIONS_NAME = "dspark_predictions.parquet"
METRICS_NAME = "dspark_metrics.json"
RUN_MANIFEST_NAME = "dspark_manifest.json"
HIERARCHY_METRICS_NAME = "hierarchical_metrics.json"
HIERARCHY_TABLE_NAME = "hierarchical_metrics.csv"
HIERARCHY_CARD_NAME = "hierarchical_dataset_card.md"
HIERARCHY_MANIFEST_NAME = "hierarchical_manifest.json"
_CSV_FIELDS = (
    "level",
    "group",
    "sample_n",
    "eligible_n",
    "n",
    "excluded_shallower_reference_count",
    "valid_count",
    "failed_count",
    "truncated_count",
    "unmapped_count",
    "unmapped_gold_count",
    "unmapped_prediction_count",
    "correct_count",
    "accuracy",
    "valid_mapped_count",
    "valid_only_accuracy",
    "macro_f1",
    "multiclass_mcc",
    "label_universe",
)


def score_hierarchical_run(
    *,
    predictions_path: Path,
    run_manifest_path: Path,
    frozen_sample_path: Path,
    frozen_manifest_path: Path,
    candidate_csv_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    scoring_commit: str,
) -> dict[str, Any]:
    """Score an existing full run without changing its predictions or fine metrics."""
    _validate_commit(scoring_commit, "scoring_commit")
    inputs = _input_paths(
        predictions_path,
        run_manifest_path,
        frozen_sample_path,
        frozen_manifest_path,
        candidate_csv_path,
        taxonomy_path,
    )
    _validate_input_files(inputs)
    _validate_output_directory(output_dir)
    hashes = _input_hashes(inputs)
    sample = _read_json(frozen_sample_path)
    frozen_manifest = _read_json(frozen_manifest_path)
    run_manifest = _read_json(run_manifest_path)
    taxonomy = _read_json(taxonomy_path)
    candidates = _read_candidate_sources(candidate_csv_path, sample, hashes)
    rows = _read_predictions(predictions_path)
    _validate_source_artifacts(sample, frozen_manifest, run_manifest, hashes, rows)

    report = _report_for_run(sample, rows, candidates, taxonomy)
    fine_metrics = _read_json(inputs[METRICS_NAME])
    _validate_original_leaf_metrics(report, fine_metrics)
    report["original_leaf"]["historical_metrics_sha256"] = hashes[METRICS_NAME]
    report["original_leaf"]["historical_metrics_match"] = True
    report["provenance"] = _report_provenance(
        hashes, sample, run_manifest, scoring_commit, report["classifications"]
    )
    return _write_outputs(output_dir, report, hashes, sample, run_manifest, scoring_commit)


def _input_hashes(inputs: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in inputs.items()}


def _report_for_run(
    sample: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, str]],
    taxonomy: Mapping[str, Any],
) -> dict[str, Any]:
    selected_rows = sample["selected_rows"]
    groups, gold, predicted, statuses, versions = _row_vectors(selected_rows, predictions)
    return hierarchical_report(
        groups=groups,
        gold_codes=gold,
        predicted_codes=predicted,
        parse_statuses=statuses,
        source_versions=versions,
        candidate_codes=sample["candidate_labels"]["codes"],
        candidate_sources=candidates,
        taxonomy=taxonomy,
    )


def _row_vectors(
    sample_rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> tuple[list[str], list[str], list[str | None], list[str], list[str]]:
    groups = []
    gold = []
    predicted = []
    statuses = []
    versions = []
    for sample_row, prediction in zip(sample_rows, predictions, strict=True):
        groups.append(sample_row["decision"])
        gold.append(sample_row["eunis_code"])
        predicted.append(_prediction_code(prediction))
        statuses.append(_prediction_status(prediction))
        versions.append(sample_row["eunis_source_version"])
    return groups, gold, predicted, statuses, versions


def _input_paths(
    predictions: Path,
    run_manifest: Path,
    sample: Path,
    frozen_manifest: Path,
    candidate_csv: Path,
    taxonomy: Path,
) -> dict[str, Path]:
    return {
        PREDICTIONS_NAME: predictions,
        RUN_MANIFEST_NAME: run_manifest,
        METRICS_NAME: run_manifest.parent / METRICS_NAME,
        "frozen_sample.json": sample,
        "frozen_manifest.json": frozen_manifest,
        "candidate_labels.csv": candidate_csv,
        "taxonomy_mapping.json": taxonomy,
    }


def _validate_original_leaf_metrics(
    report: Mapping[str, Any], historical_metrics: Mapping[str, Any]
) -> None:
    original = report["original_leaf"]["overall"]
    recorded = historical_metrics.get("overall")
    if not isinstance(recorded, Mapping):
        raise ValueError("historical DSpark metrics have no overall leaf report")
    keys = (
        "sample_count",
        "candidate_class_count",
        "prediction_count",
        "invalid_output_count",
        "coverage",
        "top1_accuracy",
        "macro_f1_all_candidates",
    )
    if any(original.get(key) != recorded.get(key) for key in keys):
        raise ValueError("recomputed original leaf metrics differ from historical DSpark metrics")


def _validate_input_files(inputs: Mapping[str, Path]) -> None:
    for path in inputs.values():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required historical input is not a regular file: {path}")


def _validate_output_directory(output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"hierarchy output directory already exists: {output_dir}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _read_candidate_sources(
    path: Path, sample: Mapping[str, Any], hashes: Mapping[str, str]
) -> dict[str, dict[str, str]]:
    candidate_info = sample.get("candidate_labels")
    if not isinstance(candidate_info, Mapping) or candidate_info.get("sha256") != hashes.get(
        "candidate_labels.csv"
    ):
        raise ValueError("candidate CSV checksum does not match the frozen sample")
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            return _candidate_sources_from_rows(reader.fieldnames, list(reader), candidate_info)
    except OSError as error:
        raise ValueError(f"cannot read candidate CSV: {path}") from error


def _candidate_sources_from_rows(
    fields: Sequence[str] | None,
    rows: Sequence[Mapping[str, str | None]],
    candidate_info: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    _validate_candidate_csv_fields(fields)
    codes: list[str] = []
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        code = row.get("eunis_code")
        if not code or code in result:
            raise ValueError("candidate CSV codes must be non-empty and unique")
        result[code] = _candidate_metadata(row, code)
        codes.append(code)
    if codes != candidate_info.get("codes"):
        raise ValueError("candidate CSV code list does not match the frozen sample")
    return result


def _validate_candidate_csv_fields(fields: Sequence[str] | None) -> None:
    required = {
        "eunis_code",
        "classification_release",
        "classification_source",
        "source_sha256",
    }
    if fields is None or not required.issubset(fields):
        raise ValueError("candidate CSV is missing classification provenance columns")


def _candidate_metadata(row: Mapping[str, str | None], code: str) -> dict[str, str]:
    metadata = {
        key: row.get(key) or ""
        for key in ("classification_release", "classification_source", "source_sha256")
    }
    if any(not value for value in metadata.values()):
        raise ValueError(f"candidate taxonomy provenance is incomplete: {code}")
    return metadata


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    try:
        rows = pq.read_table(path).to_pylist()
    except (OSError, pa.ArrowException) as error:
        raise ValueError(f"cannot read prediction Parquet: {path}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("historical prediction Parquet must contain row objects")
    return rows


def _validate_source_artifacts(
    sample: Mapping[str, Any],
    frozen_manifest: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    hashes: Mapping[str, str],
    predictions: Sequence[Mapping[str, Any]],
) -> None:
    frozen_rows = _sample_rows(sample)
    sample_ids = _sample_ids(frozen_rows)
    selection = sample.get("selection")
    if not isinstance(selection, Mapping) or selection.get("sample_ids_sha256") != _sha256_json(
        sample_ids
    ):
        raise ValueError("frozen sample ID digest is inconsistent")
    _validate_frozen_manifest(frozen_manifest, hashes, selection)
    _validate_run_manifest(run_manifest, hashes, selection, len(frozen_rows))
    _validate_prediction_identity(predictions, frozen_rows, sample_ids)


def _sample_rows(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = sample.get("selected_rows")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("frozen sample must contain selected rows")
    return rows


def _sample_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    ids = []
    for row in rows:
        value = row.get("sample_id")
        if not isinstance(value, str) or not value:
            raise ValueError("frozen sample IDs must be unique non-empty strings")
        ids.append(value)
    if len(set(ids)) != len(ids):
        raise ValueError("frozen sample IDs must be unique non-empty strings")
    return ids


def _validate_frozen_manifest(
    manifest: Mapping[str, Any], hashes: Mapping[str, str], selection: Mapping[str, Any]
) -> None:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping):
        raise ValueError("frozen sample manifest has no output checksums")
    expected = {
        "frozen_sample.json": hashes["frozen_sample.json"],
        "candidate_labels.csv": hashes["candidate_labels.csv"],
    }
    if any(outputs.get(name) != digest for name, digest in expected.items()):
        raise ValueError("frozen sample output checksum mismatch")
    if manifest.get("sample_ids_sha256") != selection.get("sample_ids_sha256"):
        raise ValueError("frozen sample manifest ID digest mismatch")


def _validate_run_manifest(
    manifest: Mapping[str, Any],
    hashes: Mapping[str, str],
    selection: Mapping[str, Any],
    row_count: int,
) -> None:
    _validate_run_output_hashes(manifest, hashes)
    _validate_run_sample_reference(manifest, hashes, selection)
    _validate_full_pilot_scope(manifest, selection, row_count)


def _validate_run_output_hashes(manifest: Mapping[str, Any], hashes: Mapping[str, str]) -> None:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping):
        raise ValueError("DSpark manifest has no output checksums")
    expected_outputs = {
        PREDICTIONS_NAME: hashes[PREDICTIONS_NAME],
        METRICS_NAME: hashes[METRICS_NAME],
    }
    if any(outputs.get(name) != digest for name, digest in expected_outputs.items()):
        raise ValueError("DSpark output checksum mismatch")


def _validate_run_sample_reference(
    manifest: Mapping[str, Any], hashes: Mapping[str, str], selection: Mapping[str, Any]
) -> None:
    if manifest.get("frozen_manifest_sha256") != hashes["frozen_manifest.json"]:
        raise ValueError("DSpark manifest points to a different frozen sample manifest")
    if manifest.get("sample") != selection:
        raise ValueError("DSpark manifest sample metadata does not match frozen input")


def _validate_full_pilot_scope(
    manifest: Mapping[str, Any], selection: Mapping[str, Any], row_count: int
) -> None:
    scope = manifest.get("run_scope")
    if not isinstance(scope, Mapping) or scope.get("kind") != "full-pilot":
        raise ValueError("hierarchical retrospective scoring requires a full pilot run")
    if scope.get("processed_row_count") != row_count or scope.get(
        "processed_sample_ids_sha256"
    ) != selection.get("sample_ids_sha256"):
        raise ValueError("DSpark manifest run scope does not match the frozen sample")


def _validate_prediction_identity(
    predictions: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[str],
) -> None:
    _validate_prediction_order(predictions, sample_ids)
    for prediction, frozen in zip(predictions, sample_rows, strict=True):
        _validate_prediction_fields(prediction, frozen)


def _validate_prediction_order(
    predictions: Sequence[Mapping[str, Any]], sample_ids: Sequence[str]
) -> None:
    if [row.get("sample_id") for row in predictions] != list(sample_ids):
        raise ValueError("prediction rows do not match frozen sample order")


def _validate_prediction_fields(prediction: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    for field in ("decision", "eunis_code", "eunis_source_version"):
        if prediction.get(field) != frozen.get(field):
            raise ValueError(f"prediction row does not match frozen {field}")


def _prediction_code(row: Mapping[str, Any]) -> str | None:
    value = row.get("parsed_eunis_code")
    return value if isinstance(value, str) else None


def _prediction_status(row: Mapping[str, Any]) -> str:
    value = row.get("parse_status")
    return value if isinstance(value, str) else "failed"


def _report_provenance(
    hashes: Mapping[str, str],
    sample: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    scoring_commit: str,
    classifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selection = sample["selection"]
    return {
        "scoring_protocol": "eunis-hierarchy-v1",
        "scoring_commit": scoring_commit,
        "historical_computation_commit": run_manifest.get("computation_commit"),
        "historical_validation_commit": run_manifest.get("validation_commit"),
        "sample_ids_sha256": selection["sample_ids_sha256"],
        "sample_size": selection["sample_size"],
        "decision_counts": dict(
            sorted(Counter(row["decision"] for row in sample["selected_rows"]).items())
        ),
        "inputs_sha256": dict(sorted(hashes.items())),
        "taxonomy_classifications": list(classifications),
    }


def _write_outputs(
    output_dir: Path,
    report: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    sample: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    scoring_commit: str,
) -> dict[str, Any]:
    with staged_directory(
        output_dir.parent, prefix=f".{output_dir.name}.staging-", preserve_on_error=True
    ) as staging:
        metrics_path = staging / HIERARCHY_METRICS_NAME
        table_path = staging / HIERARCHY_TABLE_NAME
        card_path = staging / HIERARCHY_CARD_NAME
        manifest_path = staging / HIERARCHY_MANIFEST_NAME
        _write_json_exclusive(metrics_path, report)
        _write_csv(table_path, report["rows"])
        _write_dataset_card(card_path, report)
        manifest = _output_manifest(
            report,
            input_hashes,
            sample,
            run_manifest,
            scoring_commit,
            metrics_path,
            table_path,
            card_path,
        )
        _write_json_exclusive(manifest_path, manifest)
        publish_directory(staging, output_dir)
    return {
        "output_dir": str(output_dir),
        "sample_size": sample["selection"]["sample_size"],
        "classification_count": len(report["classifications"]),
        "level_group_rows": len(report["rows"]),
        "files_sha256": {
            name: sha256_file(output_dir / name)
            for name in (
                HIERARCHY_METRICS_NAME,
                HIERARCHY_TABLE_NAME,
                HIERARCHY_CARD_NAME,
                HIERARCHY_MANIFEST_NAME,
            )
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False, separators=(",", ":"))
                    if key == "label_universe"
                    else row[key]
                    for key in _CSV_FIELDS
                }
            )


def _write_dataset_card(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "## Retrospective hierarchical EUNIS evaluation",
        "",
        "The original EUNIS leaf predictions and reference labels are unchanged. "
        "Levels 1 through 3 include every row; level 4 includes only references with a level 4 target, "
        "and reports shallower exclusions explicitly. Invalid and truncated predictions remain "
        "in each eligible denominator.",
        "",
        "### Official ancestor levels",
        "",
        "| Level | Group | n | Excluded shallower | Valid | Failed | Truncated | Unmapped | Accuracy | "
        "Valid-only accuracy | Macro-F1 | Multiclass MCC |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {level} | {group} | {n} | {excluded} | {valid} | {failed} | {truncated} | "
            "{unmapped} | {accuracy} | {valid_accuracy} | {macro_f1} | {mcc} |".format(
                level=row["level"],
                group=row["group"],
                n=row["n"],
                excluded=row["excluded_shallower_reference_count"],
                valid=row["valid_count"],
                failed=row["failed_count"],
                truncated=row["truncated_count"],
                unmapped=row["unmapped_count"],
                accuracy=_format_metric(row["accuracy"]),
                valid_accuracy=_format_metric(row["valid_only_accuracy"]),
                macro_f1=_format_metric(row["macro_f1"]),
                mcc=_format_metric(row["multiclass_mcc"]),
            )
        )
    leaf = report["original_leaf"]
    lines.extend(
        [
            "",
            "### Original leaf-label results",
            "",
            "| Group | n | Valid predictions | Invalid outputs | Accuracy | Coverage | Macro-F1 |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.append(_leaf_card_row("overall", leaf["overall"]))
    lines.extend(_leaf_group_card_row(row) for row in leaf["by_decision"])
    lines.extend(
        [
            "",
            "Ancestor scoring uses the source-specific official EEA terrestrial 2021 and marine "
            "2022 mappings. Label universes and input checksums are in `hierarchical_metrics.json` "
            "and `hierarchical_manifest.json`.",
            "",
        ]
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def _leaf_card_row(group: str, row: Mapping[str, Any]) -> str:
    return (
        f"| {group} | {row['sample_count']} | {row['prediction_count']} | "
        f"{row['invalid_output_count']} | {_format_metric(row['top1_accuracy'])} | "
        f"{_format_metric(row['coverage'])} | {_format_metric(row['macro_f1_all_candidates'])} |"
    )


def _leaf_group_card_row(row: Mapping[str, Any]) -> str:
    return (
        f"| {row['group']} | {row['sample_count']} | {row['prediction_count']} | "
        f"{row['sample_count'] - row['prediction_count']} | {_format_metric(row['top1_accuracy'])} | "
        f"{_format_metric(row['coverage'])} | — |"
    )


def _format_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


def _output_manifest(
    report: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    sample: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    scoring_commit: str,
    metrics_path: Path,
    table_path: Path,
    card_path: Path,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "protocol_version": report["protocol_version"],
        "scoring_commit": scoring_commit,
        "historical_computation_commit": run_manifest.get("computation_commit"),
        "historical_validation_commit": run_manifest.get("validation_commit"),
        "sample_ids_sha256": sample["selection"]["sample_ids_sha256"],
        "input_sha256": dict(sorted(input_hashes.items())),
        "taxonomy_classifications": report["classifications"],
        "outputs_sha256": {
            HIERARCHY_METRICS_NAME: sha256_file(metrics_path),
            HIERARCHY_TABLE_NAME: sha256_file(table_path),
            HIERARCHY_CARD_NAME: sha256_file(card_path),
        },
    }

"""Freeze an audit-ready sample and run pinned E5 zero-shot candidate ranking."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from georeset_text_label_benchmark.pilot.embeddings import encode_texts, rank_candidates
from georeset_text_label_benchmark.pilot.metrics import (
    class_breakdown,
    classification_report,
    group_breakdown,
)
from georeset_text_label_benchmark.pilot.protocol import (
    BATCH_SIZE,
    CANDIDATE_LABELS_SHA256,
    MAX_LENGTH,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    OVERLAP_DATASET,
    OVERLAP_PARQUET_SHA256,
    OVERLAP_REVISION,
    SAMPLE_SEED,
    SAMPLE_SIZE,
)
from georeset_text_label_benchmark.pilot.publication import (
    ensure_publication_supported,
    find_staged_directory,
    publish_files,
    staged_directory,
)
from georeset_text_label_benchmark.pilot.sampling import (
    build_candidate_labels,
    select_distinct_sample,
)

SOURCE_COLUMNS = (
    "source_pbf",
    "osm_type",
    "osm_id",
    "description_identity",
    "tag_key",
    "sentence_index",
    "sentence",
    "text_sha256",
    "language_code",
    "eunis_code",
    "eunis_name",
)
CANDIDATE_FIELDS = (
    "eunis_code",
    "eunis_name",
    "eunis_description",
    "candidate_text",
    "classification_release",
    "classification_source",
    "source_sha256",
    "license",
)
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EXPECTED_SOURCE_ROWS = 224_789
EXPECTED_CANDIDATES = 158
OUTPUT_FILES = ("predictions.parquet", "metrics.json", "manifest.json")


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_staged_hashes(
    staging: Path,
    manifest: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    label: str,
) -> None:
    hashes = manifest.get("outputs_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"retained {label} staging manifest has no output hashes: {staging}")
    _validate_staged_hash_group(staging, hashes, input_paths, label, "input")
    _validate_staged_hash_group(staging, hashes, output_paths, label, "output")


def _validate_staged_hash_group(
    staging: Path,
    hashes: Mapping[str, Any],
    paths: Mapping[str, Path],
    label: str,
    kind: str,
) -> None:
    for name, path in paths.items():
        if hashes.get(name) != sha256_file(path):
            raise ValueError(f"retained {label} staging {kind} hash mismatch for {name}: {staging}")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _read_taxonomy(path: Path) -> dict[str, dict[str, str]]:
    taxonomy: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _validate_taxonomy_header(reader.fieldnames)
        for row in reader:
            code, details = _validated_taxonomy_row(row, taxonomy)
            taxonomy[code] = details
    return taxonomy


def _validate_taxonomy_header(fields: Sequence[str] | None) -> None:
    if tuple(fields or ()) != CANDIDATE_FIELDS:
        raise ValueError("candidate CSV schema does not match the pinned pilot schema")


def _validated_taxonomy_row(
    row: Mapping[str, str], existing: Mapping[str, Any]
) -> tuple[str, dict[str, str]]:
    code = row["eunis_code"]
    if not code or code in existing:
        raise ValueError(f"missing or duplicate candidate code: {code}")
    expected_text = f"{row['eunis_name']}\n{row['eunis_description']}"
    if row["candidate_text"] != expected_text:
        raise ValueError(f"candidate text mismatch for {code}")
    return code, {
        "name": row["eunis_name"],
        "description": row["eunis_description"],
        "classification_release": row["classification_release"],
        "classification_source": row["classification_source"],
        "source_sha256": row["source_sha256"],
        "license": row["license"],
    }


def _read_source_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    parquet = pq.ParquetFile(path)
    rows: list[dict[str, Any]] = []
    for batch in parquet.iter_batches(columns=list(SOURCE_COLUMNS), batch_size=16_384):
        rows.extend(batch.to_pylist())
    return rows, parquet.metadata.num_rows


def _sample_metadata(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "row_count": len(rows),
        "unique_sentence_hash_count": len({row["text_sha256"] for row in rows}),
        "unique_polygon_count": len(
            {(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in rows}
        ),
        "observed_eunis_class_count": len({row["eunis_code"] for row in rows}),
    }


def _validate_freeze_inputs(
    source_parquet: Path,
    candidate_csv: Path,
    expected_source_sha256: str,
    expected_candidate_sha256: str | None,
    source_revision: str,
) -> tuple[str, str]:
    if source_revision != OVERLAP_REVISION:
        raise ValueError("source revision is not the pinned overlap snapshot")
    source_hash = sha256_file(source_parquet)
    if source_hash != expected_source_sha256:
        raise ValueError("overlap Parquet SHA-256 does not match the pinned input")
    candidate_hash = sha256_file(candidate_csv)
    if expected_candidate_sha256 is not None and candidate_hash != expected_candidate_sha256:
        raise ValueError("candidate CSV SHA-256 does not match the pinned input")
    return source_hash, candidate_hash


def _validate_source_row_count(parquet_row_count: int) -> None:
    if parquet_row_count != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"expected {EXPECTED_SOURCE_ROWS} overlap rows, got {parquet_row_count}")


def _validate_candidate_count(candidates: Sequence[Mapping[str, Any]]) -> None:
    if len(candidates) != EXPECTED_CANDIDATES:
        raise ValueError(f"expected {EXPECTED_CANDIDATES} candidate classes")


def freeze_sample(
    source_parquet: Path,
    candidate_csv: Path,
    output_dir: Path,
    *,
    expected_source_sha256: str = OVERLAP_PARQUET_SHA256,
    expected_candidate_sha256: str | None = CANDIDATE_LABELS_SHA256,
    source_revision: str = OVERLAP_REVISION,
    size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> dict[str, Any]:
    """Write immutable selected rows and candidate text before predictions."""
    if output_dir.exists():
        raise FileExistsError(f"pilot directory already exists: {output_dir}")
    source_hash, candidate_hash = _validate_freeze_inputs(
        source_parquet,
        candidate_csv,
        expected_source_sha256,
        expected_candidate_sha256,
        source_revision,
    )
    source_rows, parquet_row_count = _read_source_rows(source_parquet)
    _validate_source_row_count(parquet_row_count)
    taxonomy = _read_taxonomy(candidate_csv)
    candidates = build_candidate_labels(source_rows, taxonomy)
    _validate_candidate_count(candidates)
    selected = select_distinct_sample(source_rows, size=size, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_output = output_dir / "candidate_labels.csv"
    shutil.copyfile(candidate_csv, candidate_output)
    sample = _frozen_sample(
        source_revision, source_hash, candidate_hash, source_rows, candidates, selected, size, seed
    )
    _write_json_exclusive(output_dir / "frozen_sample.json", sample)
    return _freeze_summary(output_dir, size, len(candidates), sample)


def _selection_metadata(
    selected: Sequence[Mapping[str, Any]], size: int, seed: int
) -> dict[str, Any]:
    ids = [row["sample_id"] for row in selected]
    return {
        "method": (
            "Sort occurrences by SHA256(seed + ':' + stable occurrence ID), then retain "
            "the first rows with unseen exact text hashes and unseen polygon keys."
        ),
        "seed": seed,
        "sample_size": size,
        "sample_ids_sha256": _sha256_json(ids),
        "unique_sentence_hash_count": len({row["text_sha256"] for row in selected}),
        "unique_polygon_count": len(
            {(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in selected}
        ),
    }


def _frozen_sample(
    source_revision: str,
    source_hash: str,
    candidate_hash: str,
    source_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    size: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "source": {
            "dataset": OVERLAP_DATASET,
            "revision": source_revision,
            "file": "overlap.parquet",
            "sha256": source_hash,
            "license": "OpenStreetMap ODbL-1.0; European Environment Agency CC-BY-4.0",
        },
        "source_coverage": _sample_metadata(source_rows),
        "selection": _selection_metadata(selected, size, seed),
        "candidate_labels": {
            "file": "candidate_labels.csv",
            "sha256": candidate_hash,
            "count": len(candidates),
            "codes": [row["eunis_code"] for row in candidates],
            "text_method": "Exact pinned EUNIS English name, newline, exact EEA description.",
        },
        "selected_rows": list(selected),
    }


def _freeze_summary(
    output_dir: Path, size: int, candidate_count: int, sample: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "sample_size": size,
        "candidate_count": candidate_count,
        "sample_ids_sha256": sample["selection"]["sample_ids_sha256"],
        "frozen_sample_sha256": sha256_file(output_dir / "frozen_sample.json"),
    }


def _validate_commit(commit: str, name: str) -> None:
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError(f"{name} must be a 40-character lowercase Git commit SHA")


def _validate_frozen_source(source: Mapping[str, Any]) -> None:
    if source["dataset"] != OVERLAP_DATASET or source["revision"] != OVERLAP_REVISION:
        raise ValueError("frozen sample uses a different overlap dataset revision")
    if source["sha256"] != OVERLAP_PARQUET_SHA256:
        raise ValueError("frozen sample uses a different overlap Parquet hash")


def _validate_frozen_rows(rows: list[dict[str, Any]], selection: Mapping[str, Any]) -> None:
    _validate_frozen_row_count(rows, selection)
    ids = [row["sample_id"] for row in rows]
    _require_unique_frozen_values(ids)
    _require_unique_frozen_values([row["text_sha256"] for row in rows])
    polygons = [(row["source_pbf"], row["osm_type"], row["osm_id"]) for row in rows]
    _require_unique_frozen_values(polygons)
    _validate_frozen_ids(ids, selection)
    _validate_frozen_selection(rows, selection)


def _validate_frozen_row_count(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any]
) -> None:
    if len(rows) != selection["sample_size"]:
        raise ValueError("frozen sample row count does not match its declaration")


def _require_unique_frozen_values(values: Sequence[Any]) -> None:
    if len(set(values)) != len(values):
        raise ValueError("frozen sample must have unique IDs, sentence hashes, and polygons")


def _validate_frozen_ids(ids: Sequence[str], selection: Mapping[str, Any]) -> None:
    if _sha256_json(ids) != selection["sample_ids_sha256"]:
        raise ValueError("frozen sample ID checksum mismatch")


def _validate_frozen_selection(rows: list[dict[str, Any]], selection: Mapping[str, Any]) -> None:
    expected = select_distinct_sample(rows, size=selection["sample_size"], seed=selection["seed"])
    if expected != rows:
        raise ValueError("frozen rows do not match the deterministic sampling protocol")


def _read_frozen_sample(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen_path = run_dir / "frozen_sample.json"
    with frozen_path.open(encoding="utf-8") as stream:
        sample = json.load(stream)
    _validate_frozen_source(sample["source"])
    rows = sample["selected_rows"]
    _validate_frozen_rows(rows, sample["selection"])
    return sample, rows


def _download_model(cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        cache_dir=str(cache_dir),
        allow_patterns=list(MODEL_FILES),
        token=False,
    )
    return Path(local_path)


def _load_model(model_dir: Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    if model.config.model_type != "bert" or model.config.hidden_size != 384:
        raise ValueError("downloaded checkpoint does not match multilingual-e5-small")
    model.to(torch.device("cpu"))
    model.eval()
    return tokenizer, model


def _model_inventory(model_dir: Path) -> list[dict[str, Any]]:
    inventory = []
    for name in MODEL_FILES:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"required pinned model file is missing: {name}")
        inventory.append({"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return inventory


def _candidate_rows(run_dir: Path, sample: Mapping[str, Any]) -> list[dict[str, str]]:
    candidate_path = run_dir / sample["candidate_labels"]["file"]
    if sha256_file(candidate_path) != sample["candidate_labels"]["sha256"]:
        raise ValueError("candidate label file checksum mismatch")
    taxonomy = _read_taxonomy(candidate_path)
    candidates = _candidate_records(taxonomy)
    _validate_candidate_references(candidates, sample)
    return candidates


def read_frozen_pilot_inputs(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    """Read and validate the existing frozen sample and candidate vocabulary."""
    sample, rows = _read_frozen_sample(run_dir)
    candidates = _candidate_rows(run_dir, sample)
    return sample, rows, candidates


def _candidate_records(taxonomy: Mapping[str, Mapping[str, str]]) -> list[dict[str, str]]:
    candidates = [
        {
            "eunis_code": code,
            "eunis_name": details["name"],
            "candidate_text": f"{details['name']}\n{details['description']}",
            "classification_release": details["classification_release"],
            "classification_source": details["classification_source"],
            "source_sha256": details["source_sha256"],
            "license": details["license"],
        }
        for code, details in sorted(taxonomy.items())
    ]
    _validate_candidate_count(candidates)
    return candidates


def _validate_candidate_references(
    candidates: Sequence[Mapping[str, str]], sample: Mapping[str, Any]
) -> None:
    _validate_candidate_code_list(candidates, sample)
    names = {row["eunis_code"]: row["eunis_name"] for row in candidates}
    _validate_sampled_candidate_names(sample, names)


def _validate_candidate_code_list(
    candidates: Sequence[Mapping[str, str]], sample: Mapping[str, Any]
) -> None:
    if [row["eunis_code"] for row in candidates] != sample["candidate_labels"]["codes"]:
        raise ValueError("candidate code list does not match the frozen protocol")


def _validate_sampled_candidate_names(sample: Mapping[str, Any], names: Mapping[str, str]) -> None:
    for row in sample["selected_rows"]:
        if names.get(row["eunis_code"]) != row["eunis_name"]:
            raise ValueError(f"frozen gold name mismatch for {row['eunis_code']}")


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, str]],
    rankings: Sequence[Sequence[tuple[str, float]]],
) -> list[dict[str, Any]]:
    names = {row["eunis_code"]: row["eunis_name"] for row in candidates}
    return [
        _prediction_row(row, ranking, names) for row, ranking in zip(rows, rankings, strict=True)
    ]


def _prediction_row(
    row: Mapping[str, Any], ranking: Sequence[tuple[str, float]], names: Mapping[str, str]
) -> dict[str, Any]:
    codes = [code for code, _ in ranking]
    scores = [score for _, score in ranking]
    top1 = codes[0]
    return {
        **row,
        "gold_eunis_code": row["eunis_code"],
        "gold_eunis_name": row["eunis_name"],
        "top1_eunis_code": top1,
        "top1_eunis_name": names[top1],
        "top1_cosine": scores[0],
        "top5_eunis_codes": codes,
        "top5_eunis_names": [names[code] for code in codes],
        "top5_cosine_scores": scores,
        "correct_top1": top1 == row["eunis_code"],
        "correct_top5": row["eunis_code"] in codes,
    }


def _confusion_pairs(gold: Sequence[str], top1: Sequence[str]) -> list[dict[str, Any]]:
    counts = Counter((actual, predicted) for actual, predicted in zip(gold, top1, strict=True))
    mistakes = [
        ((actual, predicted), count)
        for (actual, predicted), count in counts.items()
        if actual != predicted
    ]
    mistakes.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [
        {"gold_eunis_code": actual, "predicted_eunis_code": predicted, "count": count}
        for (actual, predicted), count in mistakes[:10]
    ]


def _metrics_payload(
    predictions: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    gold, top1, top5 = _prediction_metric_values(predictions)
    codes = [row["eunis_code"] for row in candidates]
    return {
        "overall": classification_report(gold, top1, top5, codes),
        "by_eunis_class": _named_class_breakdown(gold, top1, top5, candidates, codes),
        "by_language": group_breakdown(
            [row["language_code"] for row in predictions], gold, top1, top5
        ),
        "top_confusions": _confusion_pairs(gold, top1),
        "scope_note": (
            "All rows are sampled from the existing positive Description/EUNIS overlap. "
            "The EUNIS assignment is polygon-level context, not sentence-level truth; "
            "its scientific validation remains unconfirmed."
        ),
    }


def _prediction_metric_values(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[Sequence[str]]]:
    gold = [row["gold_eunis_code"] for row in predictions]
    top1 = [row["top1_eunis_code"] for row in predictions]
    top5 = [row["top5_eunis_codes"] for row in predictions]

    return gold, top1, top5


def _named_class_breakdown(
    gold: Sequence[str],
    top1: Sequence[str],
    top5: Sequence[Sequence[str]],
    candidates: Sequence[Mapping[str, str]],
    codes: Sequence[str],
) -> list[dict[str, Any]]:
    names = {row["eunis_code"]: row["eunis_name"] for row in candidates}
    classes = class_breakdown(gold, top1, top5, codes)
    for item in classes:
        item["eunis_name"] = names[item["eunis_code"]]
    return classes


def _runtime_metadata() -> dict[str, Any]:
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "pyarrow": version("pyarrow"),
        "inference_device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def _write_outputs(
    run_dir: Path,
    predictions: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> tuple[Path, Path]:
    prediction_path = run_dir / "predictions.parquet"
    pq.write_table(pa.Table.from_pylist(list(predictions)), prediction_path, compression="zstd")
    metrics_path = run_dir / "metrics.json"
    _write_json_exclusive(metrics_path, metrics)
    return prediction_path, metrics_path


def run_pilot(
    run_dir: Path,
    model_cache_dir: Path,
    *,
    computation_commit: str,
    validation_commit: str,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """Run inference only after the source rows and candidate labels are frozen."""
    _validate_run_options(computation_commit, validation_commit, batch_size, max_length)
    sample, rows = _read_frozen_sample(run_dir)
    candidates = _candidate_rows(run_dir, sample)
    staging = _find_staged_run(run_dir)
    if staging is not None:
        return _resume_staged_run(
            run_dir,
            staging,
            sample,
            candidates,
            computation_commit,
            validation_commit,
            batch_size,
            max_length,
        )
    ensure_publication_supported(run_dir)
    download_start = perf_counter()
    model_dir = _download_model(model_cache_dir)
    model_download_seconds = perf_counter() - download_start
    model_inventory = _model_inventory(model_dir)
    load_start = perf_counter()
    tokenizer, model = _load_model(model_dir)
    model_load_seconds = perf_counter() - load_start
    inference_start = perf_counter()
    sentence_start = perf_counter()
    sentence_vectors = encode_texts(
        [row["sentence"] for row in rows],
        tokenizer,
        model,
        prefix="query",
        batch_size=batch_size,
        max_length=max_length,
    )
    sentence_encoding_seconds = perf_counter() - sentence_start
    candidate_start = perf_counter()
    candidate_vectors = encode_texts(
        [row["candidate_text"] for row in candidates],
        tokenizer,
        model,
        prefix="passage",
        batch_size=batch_size,
        max_length=max_length,
    )
    candidate_encoding_seconds = perf_counter() - candidate_start
    scoring_start = perf_counter()
    rankings = rank_candidates(
        sentence_vectors,
        candidate_vectors,
        [row["eunis_code"] for row in candidates],
        top_k=5,
    )
    scoring_seconds = perf_counter() - scoring_start
    cpu_inference_seconds = perf_counter() - inference_start
    prediction_rows = _prediction_rows(rows, candidates, rankings)
    metrics = _metrics_payload(prediction_rows, candidates)
    with staged_directory(run_dir, prefix=".pilot-output-", preserve_on_error=True) as staging:
        staged_prediction, staged_metrics = _write_outputs(staging, prediction_rows, metrics)
        manifest = _build_manifest(
            run_dir,
            sample,
            candidates,
            model_inventory,
            staged_prediction,
            staged_metrics,
            computation_commit,
            validation_commit,
            batch_size,
            max_length,
            model_download_seconds,
            model_load_seconds,
            sentence_encoding_seconds,
            candidate_encoding_seconds,
            scoring_seconds,
            cpu_inference_seconds,
        )
        _write_json_exclusive(staging / "manifest.json", manifest)
        publish_files(staging, run_dir, OUTPUT_FILES)
    prediction_path = run_dir / "predictions.parquet"
    manifest_path = run_dir / "manifest.json"
    return {
        "run_dir": str(run_dir),
        "sample_count": len(prediction_rows),
        "candidate_count": len(candidates),
        "metrics": metrics["overall"],
        "cpu_inference_seconds": cpu_inference_seconds,
        "model_download_seconds": model_download_seconds,
        "model_load_seconds": model_load_seconds,
        "predictions_sha256": sha256_file(prediction_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _validate_run_inputs(
    run_dir: Path,
    computation_commit: str,
    validation_commit: str,
    batch_size: int,
    max_length: int,
) -> None:
    _validate_run_options(computation_commit, validation_commit, batch_size, max_length)
    _validate_no_existing_outputs(run_dir)


def _validate_run_options(
    computation_commit: str,
    validation_commit: str,
    batch_size: int,
    max_length: int,
) -> None:
    _validate_commit(computation_commit, "computation_commit")
    _validate_commit(validation_commit, "validation_commit")
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch size and max length must be positive")


def _validate_no_existing_outputs(run_dir: Path) -> None:
    if any((run_dir / name).exists() for name in OUTPUT_FILES):
        raise FileExistsError("pilot prediction outputs already exist")


def _find_staged_run(run_dir: Path) -> Path | None:
    staging = find_staged_directory(run_dir, prefix=".pilot-output-")
    if staging is None or (run_dir / "manifest.json").exists():
        _validate_no_existing_outputs(run_dir)
    return staging


def _resume_staged_run(
    run_dir: Path,
    staging: Path,
    sample: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
    computation_commit: str,
    validation_commit: str,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    manifest = _read_staged_manifest(staging, "E5")
    _validate_staged_run_protocol(
        staging,
        manifest,
        sample,
        candidates,
        computation_commit,
        validation_commit,
        batch_size,
        max_length,
    )
    _validate_staged_run_hashes(run_dir, staging, manifest)
    ensure_publication_supported(run_dir)
    publish_files(staging, run_dir, OUTPUT_FILES)
    return _staged_run_summary(run_dir, staging, sample, candidates, manifest)


def _read_staged_manifest(staging: Path, model_name: str) -> Mapping[str, Any]:
    path = staging / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"retained {model_name} staging directory is incomplete or unreadable: {staging}"
        ) from error
    if not isinstance(manifest, Mapping):
        raise ValueError(f"retained {model_name} staging manifest must be a JSON object: {staging}")
    return manifest


def _validate_staged_run_protocol(
    staging: Path,
    manifest: Mapping[str, Any],
    sample: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
    computation_commit: str,
    validation_commit: str,
    batch_size: int,
    max_length: int,
) -> None:
    settings = manifest.get("settings")
    expected = {
        "computation_commit": computation_commit,
        "validation_commit": validation_commit,
        "source": sample["source"],
        "sample": sample["selection"],
        "candidate_labels": _candidate_provenance(sample, candidates),
    }
    settings_valid = isinstance(settings, Mapping) and {
        "batch_size": settings.get("batch_size"),
        "maximum_tokens": settings.get("maximum_tokens"),
    } == {"batch_size": batch_size, "maximum_tokens": max_length}
    if {key: manifest.get(key) for key in expected} != expected or not settings_valid:
        raise ValueError(
            f"retained E5 staging manifest does not match this frozen run and settings: {staging}"
        )


def _validate_staged_run_hashes(run_dir: Path, staging: Path, manifest: Mapping[str, Any]) -> None:
    _validate_staged_hashes(
        staging,
        manifest,
        {name: run_dir / name for name in ("frozen_sample.json", "candidate_labels.csv")},
        {name: staging / name for name in ("predictions.parquet", "metrics.json")},
        "E5",
    )


def _staged_run_summary(
    run_dir: Path,
    staging: Path,
    sample: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = json.loads((staging / "metrics.json").read_text(encoding="utf-8"))
    timings = manifest["timings_seconds"]
    return {
        "run_dir": str(run_dir),
        "sample_count": sample["selection"]["sample_size"],
        "candidate_count": len(candidates),
        "metrics": metrics["overall"],
        "cpu_inference_seconds": timings["total_cpu_inference"],
        "model_download_seconds": timings["model_download"],
        "model_load_seconds": timings["model_load_setup"],
        "predictions_sha256": sha256_file(run_dir / "predictions.parquet"),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
    }


def _build_manifest(
    run_dir: Path,
    sample: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
    model_inventory: Sequence[Mapping[str, Any]],
    prediction_path: Path,
    metrics_path: Path,
    computation_commit: str,
    validation_commit: str,
    batch_size: int,
    max_length: int,
    model_download_seconds: float,
    model_load_seconds: float,
    sentence_encoding_seconds: float,
    candidate_encoding_seconds: float,
    scoring_seconds: float,
    cpu_inference_seconds: float,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "computation_commit": computation_commit,
        "validation_commit": validation_commit,
        "source": sample["source"],
        "source_coverage": sample["source_coverage"],
        "sample": sample["selection"],
        "candidate_labels": _candidate_provenance(sample, candidates),
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "license": "MIT",
            "files": list(model_inventory),
            "method": "query/passage-prefixed mean pooling with attention mask and L2 normalization",
        },
        "settings": {
            "query_prefix": "query: ",
            "candidate_prefix": "passage: ",
            "maximum_tokens": max_length,
            "batch_size": batch_size,
            "candidate_count": len(candidates),
            "candidate_text": "EUNIS English name + newline + authoritative EEA description",
            "scoring": "cosine similarity of L2-normalized embeddings; top-5 ties sorted by code",
            "training_or_finetuning": False,
            "gold_label_used_in_input": False,
        },
        "runtime": _runtime_metadata(),
        "timings_seconds": {
            "model_download": model_download_seconds,
            "model_load_setup": model_load_seconds,
            "sentence_encoding_cpu": sentence_encoding_seconds,
            "candidate_encoding_cpu": candidate_encoding_seconds,
            "similarity_ranking_cpu": scoring_seconds,
            "total_cpu_inference": cpu_inference_seconds,
            "cpu_inference_excludes_model_download_and_load": True,
        },
        "outputs_sha256": {
            "frozen_sample.json": sha256_file(run_dir / "frozen_sample.json"),
            "candidate_labels.csv": sha256_file(run_dir / "candidate_labels.csv"),
            "predictions.parquet": sha256_file(prediction_path),
            "metrics.json": sha256_file(metrics_path),
        },
        "limitations": [
            "A 100-sentence pilot is not a comprehensive evaluation.",
            "EUNIS labels are polygon-level and may not describe the exact sentence evidence.",
            "Final scientific validation of the EUNIS reference assignments is unconfirmed.",
            "Candidate names and descriptions are English; source sentences span languages.",
            "Results are zero-shot label ranking, not model training or fine-tuning.",
        ],
    }


def _candidate_provenance(
    sample: Mapping[str, Any], candidates: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    """Share the same pinned EEA attribution in both pilot run manifests."""
    releases = sorted({row["classification_release"] for row in candidates})
    citations = sorted({row["classification_source"] for row in candidates})
    taxonomy_hashes = sorted({row["source_sha256"] for row in candidates})
    return {
        **sample["candidate_labels"],
        "classification_releases": releases,
        "classification_sources": citations,
        "taxonomy_archive_sha256": taxonomy_hashes,
        "license": "CC-BY-4.0, European Environment Agency",
    }

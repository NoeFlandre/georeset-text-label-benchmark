"""Run the frozen 100-row pilot as direct, single-label LFM2.5 DSpark generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from georeset_text_label_benchmark.pilot import dspark
from georeset_text_label_benchmark.pilot.metrics import (
    single_label_class_breakdown,
    single_label_group_breakdown,
    single_label_report,
)
from georeset_text_label_benchmark.pilot.protocol import (
    CANDIDATE_LABELS_SHA256,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    SAMPLE_SEED,
    SAMPLE_SIZE,
)
from georeset_text_label_benchmark.pilot.publication import (
    ensure_publication_supported,
    find_staged_directory,
    publish_directory,
    staged_directory,
)
from georeset_text_label_benchmark.pilot.runner import (
    _candidate_provenance,
    _sha256_json,
    _validate_commit,
    _validate_staged_hashes,
    _write_json_exclusive,
    read_frozen_pilot_inputs,
    sha256_file,
)

MIN_GPU_MEMORY_MIB = 16 * 1024
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DRIVER_VERSION = (580, 65, 6)
MIN_DRIVER_VERSION_TEXT = "580.65.06"
MIN_CACHE_FREE_BYTES = 8 * 1024**3
MAX_CACHE_TREE_METADATA_BYTES = 16 * 1024**2
PREDICTIONS_NAME = "dspark_predictions.parquet"
METRICS_NAME = "dspark_metrics.json"
MANIFEST_NAME = "dspark_manifest.json"
_clock = perf_counter


def run_dspark_pilot(
    run_dir: Path,
    *,
    computation_commit: str,
    validation_commit: str,
    output_dir: Path | None = None,
    model_cache_dir: Path = Path(".cache/model-dspark"),
    model_cache_seed_dir: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the frozen DSpark pilot or its fixed eight-row readiness smoke."""
    destination = _output_destination(run_dir, output_dir, smoke)
    _validate_run_options(run_dir, destination, computation_commit, validation_commit)
    staging = _find_staged_dspark_run(destination)
    sample, frozen_rows, candidates = read_frozen_pilot_inputs(run_dir)
    _validate_published_sample(sample, frozen_rows)
    frozen_e5_manifest_sha256 = _validate_frozen_e5_manifest(run_dir, sample)
    rows = _rows_for_run(frozen_rows, smoke)
    smoke_gate = None
    run_scope = _run_scope(sample, rows, smoke)
    if staging is not None:
        return _resume_staged_dspark_run(
            run_dir,
            destination,
            staging,
            sample,
            candidates,
            computation_commit,
            validation_commit,
            frozen_e5_manifest_sha256,
            run_scope,
        )
    ensure_publication_supported(destination.parent)
    gpu = _require_supported_gpu()
    cuda_preflight = _runtime_compatibility_preflight(gpu)
    cache_dir = _prepare_model_cache(model_cache_dir)
    if model_cache_seed_dir is not None:
        _seed_model_cache_from_hub_cache(model_cache_seed_dir, cache_dir)
    tokenizer_start = _clock()
    tokenizer = dspark.load_tokenizer()
    tokenizer_seconds = _clock() - tokenizer_start
    encoded, prompt_hashes = _encode_frozen_prompts(tokenizer, rows, candidates)
    template_hash = dspark.template_sha256(tokenizer)
    engine_start = _clock()
    engine = dspark.SGLangEngine(dspark.engine_kwargs(), dspark.SAMPLING)
    engine_seconds = _clock() - engine_start
    generation_start = _clock()
    try:
        generated = asyncio.run(_generate_rows(engine, encoded))
    finally:
        engine.shutdown()
    generation_seconds = _clock() - generation_start
    prediction_rows = _prediction_rows(rows, candidates, encoded, prompt_hashes, generated)
    metrics = _metrics_payload(prediction_rows, candidates)
    if smoke:
        smoke_gate = _smoke_gate(prediction_rows)
        metrics["smoke_gate"] = smoke_gate
    with staged_directory(
        destination.parent,
        prefix=f".{destination.name}.staging-",
        preserve_on_error=True,
    ) as staging:
        staged_prediction = staging / PREDICTIONS_NAME
        staged_metrics = staging / METRICS_NAME
        staged_manifest = staging / MANIFEST_NAME
        pq.write_table(pa.Table.from_pylist(prediction_rows), staged_prediction, compression="zstd")
        _write_json_exclusive(staged_metrics, metrics)
        manifest = _build_manifest(
            run_dir,
            sample,
            run_scope,
            smoke_gate,
            candidates,
            staged_prediction,
            staged_metrics,
            computation_commit,
            validation_commit,
            gpu,
            cache_dir,
            template_hash,
            tokenizer_seconds,
            engine_seconds,
            generation_seconds,
            engine.version,
            frozen_e5_manifest_sha256,
            cuda_preflight,
            model_cache_seed_dir,
        )
        _write_json_exclusive(staged_manifest, manifest)
        publish_directory(staging, destination)
    prediction_path = destination / PREDICTIONS_NAME
    manifest_path = destination / MANIFEST_NAME
    result = {
        "run_dir": str(destination),
        "frozen_input_dir": str(run_dir),
        "sample_count": len(prediction_rows),
        "candidate_count": len(candidates),
        "metrics": metrics["overall"],
        "generation_seconds": generation_seconds,
        "model_load_seconds": tokenizer_seconds + engine_seconds,
        "predictions_sha256": sha256_file(prediction_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    if smoke_gate is not None:
        result["run_scope"] = run_scope
        result["smoke_gate"] = smoke_gate
    return result


def _output_destination(run_dir: Path, output_dir: Path | None, smoke: bool) -> Path:
    """Use a caller-supplied path or a distinct default name for each run scope."""
    if output_dir is not None:
        return output_dir
    name = "lfm2.5-2.6b-dspark-smoke-8-seed42" if smoke else "lfm2.5-2.6b-dspark-100-seed42"
    return run_dir.parent / name


def _rows_for_run(
    frozen_rows: Sequence[Mapping[str, Any]], smoke: bool
) -> Sequence[Mapping[str, Any]]:
    """Keep full-run order or select bounded, language-diverse smoke rows."""
    return _select_smoke_rows(frozen_rows) if smoke else frozen_rows


def _select_smoke_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Choose up to eight rows with distinct languages, then fill in frozen order."""
    first_by_language = {}
    for row in rows:
        language = row.get("language_code")
        first_by_language.setdefault(language, row)
    selected = list(first_by_language.values())[: dspark.MAX_SMOKE_ROWS]
    if len(selected) < dspark.MAX_SMOKE_ROWS:
        selected_ids = {row["sample_id"] for row in selected}
        selected.extend(_fill_smoke_rows(rows, selected_ids, dspark.MAX_SMOKE_ROWS - len(selected)))
    return selected


def _fill_smoke_rows(
    rows: Sequence[Mapping[str, Any]], selected_ids: set[str], count: int
) -> list[Mapping[str, Any]]:
    """Fill a language-diverse smoke selection in frozen sample order."""
    selected = []
    for row in rows:
        if row["sample_id"] not in selected_ids:
            selected.append(row)
            if len(selected) == count:
                break
    return selected


def _smoke_gate(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require six EOS-finished final codes and reject every truncated row."""
    valid = _count_parse_status(predictions, "valid")
    truncated = _count_parse_status(predictions, "truncated")
    valid_eos_stopped = _count_valid_eos_stopped_answers(predictions)
    minimum_valid = min(dspark.MIN_SMOKE_VALID_OUTPUTS, len(predictions))
    passed = (
        len(predictions) == dspark.MAX_SMOKE_ROWS
        and valid >= minimum_valid
        and valid_eos_stopped >= minimum_valid
        and truncated == 0
    )
    return {
        "passed": passed,
        "row_count": len(predictions),
        "valid_final_answer_count": valid,
        "minimum_valid_final_answers": minimum_valid,
        "truncated_count": truncated,
        "valid_eos_stopped_answer_count": valid_eos_stopped,
        "minimum_eos_stopped_answers": minimum_valid,
    }


def _count_parse_status(predictions: Sequence[Mapping[str, Any]], status: str) -> int:
    """Count rows with one exact parse status."""
    return sum(row["parse_status"] == status for row in predictions)


def _count_valid_eos_stopped_answers(predictions: Sequence[Mapping[str, Any]]) -> int:
    """Count valid parser results whose raw text retains the matched EOS token."""
    return sum(_is_valid_eos_stopped_answer(row) for row in predictions)


def _is_valid_eos_stopped_answer(row: Mapping[str, Any]) -> bool:
    """Check that one parsed final code was visibly stopped by the pinned EOS."""
    raw_output = row.get("raw_output")
    return (
        row.get("parse_status") == "valid"
        and row.get("finish_reason") == "stop"
        and isinstance(raw_output, str)
        and raw_output.endswith(dspark.TARGET_EOS_TOKEN)
    )


def _run_scope(
    sample: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], smoke: bool
) -> dict[str, Any]:
    ids = [str(row["sample_id"]) for row in rows]
    scope: dict[str, Any] = {
        "kind": "bounded-smoke" if smoke else "full-pilot",
        "frozen_sample_size": sample["selection"]["sample_size"],
        "processed_row_count": len(rows),
        "processed_sample_ids_sha256": _sha256_json(ids),
    }
    if smoke:
        scope["selection_method"] = (
            "Take the first sample-order row for each distinct language_code, up to eight, "
            "then fill from the frozen sample order."
        )
        scope["processed_sample_ids"] = ids
    return scope


async def _generate_rows(
    engine: Any, encoded: Sequence[tuple[list[int], int]]
) -> list[tuple[Mapping[str, Any], float]]:
    generated = []
    for input_ids, _ in encoded:
        start = _clock()
        output = await engine.generate(input_ids)
        generated.append((output, _clock() - start))
    return generated


def _encode_frozen_prompts(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, str]],
) -> tuple[list[tuple[list[int], int]], list[str]]:
    encoded = []
    prompt_hashes = []
    for row in rows:
        prompt = dspark.build_prompt(row["sentence"], candidates)
        input_ids = dspark.encode_prompt(tokenizer, prompt)
        dspark.validate_context_length(len(input_ids))
        encoded.append((input_ids, len(input_ids)))
        prompt_hashes.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    return encoded, prompt_hashes


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, str]],
    encoded: Sequence[tuple[list[int], int]],
    prompt_hashes: Sequence[str],
    generated: Sequence[tuple[Mapping[str, Any], float]],
) -> list[dict[str, Any]]:
    names = {row["eunis_code"]: row["eunis_name"] for row in candidates}
    codes = list(names)
    return [
        _prediction_row(row, tokens, prompt_hash, result, elapsed, codes, names)
        for row, (_, tokens), prompt_hash, (result, elapsed) in zip(
            rows, encoded, prompt_hashes, generated, strict=True
        )
    ]


def _prediction_row(
    row: Mapping[str, Any],
    prompt_tokens: int,
    prompt_hash: str,
    output: Mapping[str, Any],
    elapsed_seconds: float,
    candidate_codes: Sequence[str],
    candidate_names: Mapping[str, str],
) -> dict[str, Any]:
    meta = dspark.output_metadata(output)
    raw_output = str(output.get("text", ""))
    finish_reason = dspark.output_finish_reason(output)
    parsed = dspark.parse_label(
        raw_output,
        finish_reason=finish_reason,
        candidate_codes=candidate_codes,
    )
    generated_tokens = int(meta.get("completion_tokens") or 0)
    return {
        **row,
        "gold_eunis_code": row["eunis_code"],
        "gold_eunis_name": row["eunis_name"],
        "parsed_eunis_code": parsed.code,
        "parsed_eunis_name": candidate_names.get(parsed.code) if parsed.code else None,
        "correct_top1": parsed.code == row["eunis_code"] if parsed.code else None,
        "parse_status": parsed.status,
        "parse_error": parsed.error,
        "raw_output": raw_output,
        "prompt_sha256": prompt_hash,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "finish_reason": finish_reason,
        "accepted_drafts": _optional_int(meta.get("spec_num_correct_drafts")),
        "proposed_drafts": _optional_int(meta.get("spec_num_proposed_drafts")),
        "generation_seconds": elapsed_seconds,
    }


def _metrics_payload(
    predictions: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    gold, predicted, languages = _metric_values(predictions)
    codes = [row["eunis_code"] for row in candidates]
    return {
        "overall": single_label_report(gold, predicted, codes),
        "by_eunis_class": _named_class_metrics(gold, predicted, candidates, codes),
        "by_language": single_label_group_breakdown(languages, gold, predicted),
        "invalid_output_reasons": _invalid_output_reasons(predictions),
        "comparison_note": (
            "Compare top1_accuracy and macro_f1_all_candidates with the E5 run. This direct-label "
            "generation produces one code, so it has no top-5 ranking metric."
        ),
        "scope_note": (
            "This reports agreement with existing polygon-level EUNIS assignments, not sentence "
            "ground truth. The fixed sample contains positive overlap rows only."
        ),
    }


def _metric_values(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str | None], list[str | None]]:
    return (
        [row["gold_eunis_code"] for row in predictions],
        [row["parsed_eunis_code"] for row in predictions],
        [row["language_code"] for row in predictions],
    )


def _named_class_metrics(
    gold: Sequence[str],
    predicted: Sequence[str | None],
    candidates: Sequence[Mapping[str, str]],
    codes: Sequence[str],
) -> list[dict[str, Any]]:
    classes = single_label_class_breakdown(gold, predicted, codes)
    names = {row["eunis_code"]: row["eunis_name"] for row in candidates}
    for item in classes:
        item["eunis_name"] = names[item["eunis_code"]]
    return classes


def _invalid_output_reasons(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reasons = Counter(row["parse_error"] for row in predictions if row["parse_error"])
    return [{"reason": reason, "count": count} for reason, count in sorted(reasons.items())]


def _build_manifest(
    run_dir: Path,
    sample: Mapping[str, Any],
    run_scope: Mapping[str, Any],
    smoke_gate: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, str]],
    prediction_path: Path,
    metrics_path: Path,
    computation_commit: str,
    validation_commit: str,
    gpu: Mapping[str, Any],
    cache_dir: Path,
    template_hash: str,
    tokenizer_seconds: float,
    engine_seconds: float,
    generation_seconds: float,
    engine_version: str,
    frozen_e5_manifest_sha256: str,
    cuda_preflight: Mapping[str, Any],
    model_cache_seed_dir: Path | None,
) -> dict[str, Any]:
    generation_config = _generation_config(template_hash, sample)
    return {
        "format_version": 1,
        "computation_commit": computation_commit,
        "validation_commit": validation_commit,
        "source": sample["source"],
        "frozen_e5_manifest_sha256": frozen_e5_manifest_sha256,
        "source_coverage": sample["source_coverage"],
        "sample": sample["selection"],
        "run_scope": dict(run_scope),
        "smoke_gate": dict(smoke_gate) if smoke_gate is not None else None,
        "candidate_labels": _candidate_provenance(sample, candidates),
        "model": {
            "repository": dspark.TARGET_MODEL,
            "revision": dspark.TARGET_REVISION,
            "license": "LFM1.0",
            "context_tokens": dspark.MODEL_CONTEXT_TOKENS,
        },
        "draft": {
            "repository": dspark.DRAFT_MODEL,
            "revision": dspark.DRAFT_REVISION,
            "license": "LFM1.0",
            "parameters": 327_700_000,
            "context_tokens": dspark.RUNTIME_CONTEXT_TOKENS,
        },
        "generation_config": generation_config,
        "generation_config_sha256": _sha256_json(generation_config),
        "runtime": {
            **_runtime_metadata(),
            "engine_version": engine_version,
            "gpu": dict(gpu),
            "cuda_preflight": dict(cuda_preflight),
            "model_cache": str(cache_dir),
            "model_cache_seed": _cache_seed_provenance(model_cache_seed_dir),
        },
        "timings_seconds": {
            "tokenizer_download_and_load": tokenizer_seconds,
            "sglang_target_and_draft_load": engine_seconds,
            "generation_total": generation_seconds,
            "generation_includes_sequential_requests": run_scope["processed_row_count"],
        },
        "outputs_sha256": {
            "frozen_sample.json": sha256_file(run_dir / "frozen_sample.json"),
            "candidate_labels.csv": sha256_file(run_dir / "candidate_labels.csv"),
            PREDICTIONS_NAME: sha256_file(prediction_path),
            METRICS_NAME: sha256_file(metrics_path),
        },
        "limitations": [
            "The source is the fixed 100-sentence positive-overlap pilot.",
            "Gold labels are existing polygon-level EUNIS assignments, not sentence-level truth.",
            "The 158 candidates are the frozen EEA vocabulary; they omit built and intensive-cropland classes.",
            "The sample is occurrence-weighted before unique-text and unique-polygon filtering.",
            "Compare top-1 and macro-F1 only with E5; direct generation has no top-5 ranking.",
            "Generation uses the target model card's sampled decoding settings.",
            "The DSpark vendor parity benchmarks use greedy decoding; this sampled run does not claim greedy parity.",
            *(
                [
                    "This is an eight-row readiness smoke and its metrics are not a full pilot result."
                ]
                if run_scope["kind"] == "bounded-smoke"
                else []
            ),
        ],
    }


def _generation_config(template_hash: str, sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt_version": dspark.PROMPT_VERSION,
        "prompt_instructions_sha256": dspark.PROMPT_SHA256,
        "tokenizer_chat_template_sha256": template_hash,
        "chat_template_kwargs": dspark.CHAT_TEMPLATE_KWARGS,
        "sampling": dspark.SAMPLING,
        "engine": dspark.ENGINE_ARGS,
        "setting_provenance": dspark.GENERATION_SETTING_PROVENANCE,
        "runtime_context_limit_tokens": dspark.RUNTIME_CONTEXT_TOKENS,
        "maximum_new_tokens": dspark.MAX_NEW_TOKENS,
        "candidate_count": len(sample["candidate_labels"]["codes"]),
        "candidate_csv_sha256": CANDIDATE_LABELS_SHA256,
        "sample_ids_sha256": sample["selection"]["sample_ids_sha256"],
        "gold_label_used_in_prompt": False,
        "embeddings_shortlisting": False,
    }


def _validate_run_inputs(
    run_dir: Path, output_dir: Path, computation_commit: str, validation_commit: str
) -> None:
    _validate_run_options(run_dir, output_dir, computation_commit, validation_commit)
    _validate_output_is_available(output_dir)


def _validate_run_options(
    run_dir: Path, output_dir: Path, computation_commit: str, validation_commit: str
) -> None:
    _validate_commit(computation_commit, "computation_commit")
    _validate_commit(validation_commit, "validation_commit")
    resolved_run_dir = run_dir.resolve()
    resolved_output_dir = output_dir.resolve()
    if resolved_output_dir == resolved_run_dir or resolved_run_dir in resolved_output_dir.parents:
        raise ValueError("DSpark output directory must be separate from frozen E5 inputs")


def _validate_output_is_available(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError("DSpark output directory already exists; choose a fresh path")


def _find_staged_dspark_run(destination: Path) -> Path | None:
    staging = find_staged_directory(destination.parent, prefix=f".{destination.name}.staging-")
    if staging is None or (destination / MANIFEST_NAME).exists():
        _validate_output_is_available(destination)
    return staging


def _resume_staged_dspark_run(
    run_dir: Path,
    destination: Path,
    staging: Path,
    sample: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
    computation_commit: str,
    validation_commit: str,
    frozen_e5_manifest_sha256: str,
    run_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = _read_staged_dspark_manifest(staging)
    _validate_staged_dspark_protocol(
        staging,
        manifest,
        sample,
        candidates,
        computation_commit,
        validation_commit,
        frozen_e5_manifest_sha256,
        run_scope,
    )
    _validate_staged_dspark_hashes(run_dir, staging, manifest)
    metrics = _read_staged_dspark_metrics(staging)
    _validate_staged_smoke_gate(staging, manifest, run_scope, metrics)
    ensure_publication_supported(destination.parent)
    publish_directory(staging, destination)
    return _staged_dspark_summary(destination, run_dir, staging, sample, candidates, manifest)


def _read_staged_dspark_manifest(staging: Path) -> Mapping[str, Any]:
    try:
        manifest = json.loads((staging / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"retained DSpark staging directory is incomplete or unreadable: {staging}"
        ) from error
    if not isinstance(manifest, Mapping):
        raise ValueError(f"retained DSpark staging manifest must be a JSON object: {staging}")
    return manifest


def _validate_staged_dspark_protocol(
    staging: Path,
    manifest: Mapping[str, Any],
    sample: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
    computation_commit: str,
    validation_commit: str,
    frozen_e5_manifest_sha256: str,
    run_scope: Mapping[str, Any] | None = None,
) -> None:
    expected = {
        "computation_commit": computation_commit,
        "validation_commit": validation_commit,
        "source": sample["source"],
        "sample": sample["selection"],
        "candidate_labels": _candidate_provenance(sample, candidates),
        "frozen_e5_manifest_sha256": frozen_e5_manifest_sha256,
    }
    if run_scope is not None:
        expected["run_scope"] = dict(run_scope)
    if {key: manifest.get(key) for key in expected} != expected:
        raise ValueError(
            f"retained DSpark staging manifest does not match this frozen run: {staging}"
        )


def _validate_staged_dspark_hashes(
    run_dir: Path, staging: Path, manifest: Mapping[str, Any]
) -> None:
    _validate_staged_hashes(
        staging,
        manifest,
        {name: run_dir / name for name in ("frozen_sample.json", "candidate_labels.csv")},
        {name: staging / name for name in (PREDICTIONS_NAME, METRICS_NAME)},
        "DSpark",
    )


def _read_staged_dspark_metrics(staging: Path) -> Mapping[str, Any]:
    try:
        metrics = json.loads((staging / METRICS_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"retained DSpark staging metrics are incomplete or unreadable: {staging}"
        ) from error
    if not isinstance(metrics, Mapping):
        raise ValueError(f"retained DSpark staging metrics must be a JSON object: {staging}")
    return metrics


def _validate_staged_smoke_gate(
    staging: Path,
    manifest: Mapping[str, Any],
    run_scope: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    if run_scope is None or run_scope.get("kind") != "bounded-smoke":
        return None
    return _validate_staged_smoke_gate_payload(staging, manifest, run_scope, metrics)


def _validate_staged_smoke_gate_payload(
    staging: Path,
    manifest: Mapping[str, Any],
    run_scope: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    smoke_gate = metrics.get("smoke_gate")
    if not isinstance(smoke_gate, Mapping):
        raise ValueError(f"retained DSpark staging smoke gate is missing or malformed: {staging}")
    _validate_smoke_gate_passed(smoke_gate, staging)
    if dict(smoke_gate) != manifest.get("smoke_gate"):
        raise ValueError(
            f"retained DSpark staging smoke gate does not match its manifest: {staging}"
        )
    if smoke_gate.get("row_count") != run_scope.get("processed_row_count"):
        raise ValueError(f"retained DSpark staging smoke gate does not match its scope: {staging}")
    predictions = _read_staged_dspark_predictions(staging)
    _validate_staged_smoke_prediction_identity(staging, run_scope, predictions)
    computed_gate = _smoke_gate(predictions)
    if dict(smoke_gate) != computed_gate:
        raise ValueError(
            f"retained DSpark staging smoke gate does not match staged predictions: {staging}"
        )
    return computed_gate


def _validate_staged_smoke_prediction_identity(
    staging: Path,
    run_scope: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> None:
    """Require retained smoke outputs to preserve their frozen sample ordering."""
    sample_ids = [row.get("sample_id") for row in predictions]
    expected_sample_ids = run_scope.get("processed_sample_ids")
    if not isinstance(expected_sample_ids, list) or sample_ids != expected_sample_ids:
        raise ValueError(f"retained DSpark staging predictions do not match smoke scope: {staging}")
    _validate_staged_smoke_prediction_count_and_hash(staging, run_scope, predictions, sample_ids)


def _validate_staged_smoke_prediction_count_and_hash(
    staging: Path,
    run_scope: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[Any],
) -> None:
    if len(predictions) != run_scope.get("processed_row_count"):
        raise ValueError(f"retained DSpark staging predictions do not match smoke scope: {staging}")
    if _sha256_json(sample_ids) != run_scope.get("processed_sample_ids_sha256"):
        raise ValueError(f"retained DSpark staging predictions do not match smoke scope: {staging}")


def _read_staged_dspark_predictions(staging: Path) -> list[dict[str, Any]]:
    """Read prediction rows used to revalidate a retained smoke transaction."""
    try:
        predictions = pq.read_table(staging / PREDICTIONS_NAME).to_pylist()
    except (OSError, pa.ArrowException) as error:
        raise ValueError(
            f"retained DSpark staging predictions are incomplete or unreadable: {staging}"
        ) from error
    if not all(isinstance(row, dict) for row in predictions):
        raise ValueError(f"retained DSpark staging predictions must contain row objects: {staging}")
    return predictions


def _validate_smoke_gate_passed(smoke_gate: Mapping[str, Any], staging: Path) -> None:
    """Reject a retained smoke record that did not meet its readiness threshold."""
    if not isinstance(smoke_gate.get("passed"), bool):
        raise ValueError(f"retained DSpark staging smoke gate is missing or malformed: {staging}")
    if smoke_gate.get("passed") is not True:
        raise ValueError(f"retained DSpark staging smoke gate did not pass: {staging}")


def _staged_dspark_summary(
    destination: Path,
    run_dir: Path,
    staging: Path,
    sample: Mapping[str, Any],
    candidates: Sequence[Mapping[str, str]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = _read_staged_dspark_metrics(staging)
    timings = manifest["timings_seconds"]
    result = {
        "run_dir": str(destination),
        "frozen_input_dir": str(run_dir),
        "sample_count": manifest.get("run_scope", {}).get(
            "processed_row_count", sample["selection"]["sample_size"]
        ),
        "candidate_count": len(candidates),
        "metrics": metrics["overall"],
        "generation_seconds": timings["generation_total"],
        "model_load_seconds": timings["tokenizer_download_and_load"]
        + timings["sglang_target_and_draft_load"],
        "predictions_sha256": sha256_file(destination / PREDICTIONS_NAME),
        "manifest_sha256": sha256_file(destination / MANIFEST_NAME),
    }
    run_scope = manifest.get("run_scope")
    if isinstance(run_scope, Mapping) and run_scope.get("kind") == "bounded-smoke":
        result["run_scope"] = dict(run_scope)
        result["smoke_gate"] = _validate_staged_smoke_gate(staging, manifest, run_scope, metrics)
    return result


def _validate_published_sample(
    sample: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    _validate_sample_selection(sample["selection"], rows)
    _validate_sample_digests(sample)


def _validate_frozen_e5_manifest(run_dir: Path, sample: Mapping[str, Any]) -> str:
    """Bind all frozen rows and candidates to the published E5 output manifest."""
    manifest_path = run_dir / "manifest.json"
    manifest = _read_frozen_e5_manifest(manifest_path)
    _verify_frozen_input_hashes(run_dir, manifest)
    _verify_frozen_e5_provenance(manifest, sample)
    _verify_frozen_e5_model(manifest)
    return sha256_file(manifest_path)


def _read_frozen_e5_manifest(manifest_path: Path) -> Mapping[str, Any]:
    if not manifest_path.is_file():
        raise ValueError("frozen E5 manifest.json is required to verify the published inputs")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("frozen E5 manifest.json is unreadable") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("frozen E5 manifest.json must contain a JSON object")
    return manifest


def _verify_frozen_input_hashes(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping):
        raise ValueError("frozen E5 manifest has no outputs_sha256 object")
    for filename in ("frozen_sample.json", "candidate_labels.csv"):
        actual = sha256_file(run_dir / filename)
        if outputs.get(filename) != actual:
            raise ValueError(f"E5 manifest hash mismatch for {filename}")


def _verify_frozen_e5_provenance(manifest: Mapping[str, Any], sample: Mapping[str, Any]) -> None:
    if manifest.get("source") != sample.get("source"):
        raise ValueError("frozen E5 manifest source does not match frozen_sample.json")
    if manifest.get("sample") != sample.get("selection"):
        raise ValueError("frozen E5 manifest sample does not match frozen_sample.json")
    _verify_frozen_candidate_provenance(manifest, sample)


def _verify_frozen_candidate_provenance(
    manifest: Mapping[str, Any], sample: Mapping[str, Any]
) -> None:
    source_candidates = manifest.get("candidate_labels")
    sample_candidates = sample.get("candidate_labels")
    if not isinstance(source_candidates, Mapping) or not isinstance(sample_candidates, Mapping):
        raise ValueError("frozen E5 manifest candidate provenance is missing")
    _verify_frozen_candidate_fields(source_candidates, sample_candidates)


def _verify_frozen_candidate_fields(
    source_candidates: Mapping[str, Any], sample_candidates: Mapping[str, Any]
) -> None:
    for key in ("file", "sha256", "count", "codes"):
        if source_candidates.get(key) != sample_candidates.get(key):
            raise ValueError("frozen E5 manifest candidate provenance does not match frozen inputs")


def _verify_frozen_e5_model(manifest: Mapping[str, Any]) -> None:
    model = manifest.get("model")
    if not isinstance(model, Mapping) or (
        model.get("repository") != MODEL_REPOSITORY or model.get("revision") != MODEL_REVISION
    ):
        raise ValueError("frozen inputs are not from the pinned E5 model run")


def _validate_sample_selection(
    selection: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    if selection["sample_size"] != SAMPLE_SIZE or len(rows) != SAMPLE_SIZE:
        raise ValueError("DSpark evaluation requires the published 100-row pilot sample")
    if selection["seed"] != SAMPLE_SEED:
        raise ValueError("frozen sample seed does not match the published pilot")


def _validate_sample_digests(sample: Mapping[str, Any]) -> None:
    selection = sample["selection"]
    if selection["sample_ids_sha256"] != dspark.EXPECTED_SAMPLE_IDS_SHA256:
        raise ValueError("frozen sample IDs do not match the published pilot")
    if sample["candidate_labels"]["sha256"] != CANDIDATE_LABELS_SHA256:
        raise ValueError("candidate CSV hash does not match the published pilot")


def _require_supported_gpu() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("DSpark inference requires a visible NVIDIA CUDA GPU and nvidia-smi")
    record = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.splitlines()[0]
    return _validate_gpu_record(record)


def _validate_gpu_record(record: str) -> dict[str, Any]:
    name, memory_mib, capability, driver_version = _parse_gpu_record(record)
    _validate_gpu_capacity(memory_mib, capability)
    _verify_minimum_cuda_driver(driver_version)
    return {
        "name": name,
        "memory_mib": memory_mib,
        "compute_capability": capability,
        "driver_version": driver_version,
    }


def _parse_gpu_record(record: str) -> tuple[str, int, str, str]:
    name, memory, capability, driver_version = (
        part.strip() for part in record.rsplit(",", maxsplit=3)
    )
    memory_mib = int(float(memory))
    return name, memory_mib, capability, driver_version


def _validate_gpu_capacity(memory_mib: int, capability: str) -> None:
    compute = tuple(int(part) for part in capability.split(".", maxsplit=1))
    if memory_mib < MIN_GPU_MEMORY_MIB or compute < MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"DSpark GPU gate requires >= {MIN_GPU_MEMORY_MIB} MiB and compute capability "
            f">= 8.0; found {memory_mib} MiB, capability {capability}"
        )


def _version_tuple(value: str, label: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(part) for part in value.split("."))
    except ValueError as error:
        raise RuntimeError(f"{label} is malformed: {value}") from error
    if len(parts) < 2:
        raise RuntimeError(f"{label} is malformed: {value}")
    return parts


def _runtime_compatibility_preflight(gpu: Mapping[str, Any]) -> dict[str, Any]:
    """Check pinned CUDA runtime and compile a tiny FlashInfer kernel before weights load."""
    driver_version = str(gpu.get("driver_version", ""))
    _verify_minimum_cuda_driver(driver_version)
    nvcc_release = _read_nvcc_release()

    import torch

    torch_cuda = str(torch.version.cuda or "")
    _verify_cuda_runtime(nvcc_release, torch_cuda, torch.cuda.is_available())
    sglang_version, flashinfer_version, torch_version = _verify_runtime_package_versions()
    smoke = _run_flashinfer_smoke(torch)

    return {
        "status": "passed",
        "driver_version": driver_version,
        "nvcc_release": nvcc_release,
        "torch_version": torch_version,
        "torch_cuda": torch_cuda,
        "sglang_version": sglang_version,
        "flashinfer_python_version": flashinfer_version,
        "flashinfer_jit_smoke": smoke,
    }


def _verify_minimum_cuda_driver(driver_version: str) -> None:
    driver = _version_tuple(driver_version, "CUDA driver version")
    if driver < MIN_DRIVER_VERSION:
        raise RuntimeError(
            f"CUDA driver {driver_version} is below the minimum {MIN_DRIVER_VERSION_TEXT}"
        )


def _read_nvcc_release() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("CUDA 13.0 nvcc is required for the FlashInfer runtime")
    nvcc_output = subprocess.run(
        [nvcc, "--version"], capture_output=True, check=True, text=True
    ).stdout
    release_match = re.search(r"release\s+(\d+\.\d+)", nvcc_output)
    if release_match is None:
        raise RuntimeError("could not parse the CUDA release from nvcc --version")
    return release_match.group(1)


def _verify_cuda_runtime(nvcc_release: str, torch_cuda: str, is_available: bool) -> None:
    if nvcc_release != torch_cuda:
        raise RuntimeError(f"nvcc CUDA {nvcc_release} does not match PyTorch CUDA {torch_cuda}")
    if nvcc_release != "13.0":
        raise RuntimeError(f"DSpark runtime requires CUDA 13.0; found CUDA {nvcc_release}")
    if not is_available:
        raise RuntimeError("PyTorch cannot access the visible CUDA GPU")


def _verify_runtime_package_versions() -> tuple[str, str, str]:
    sglang_version = version("sglang")
    flashinfer_version = version("flashinfer-python")
    torch_version = version("torch")
    if sglang_version != dspark.SGLANG_VERSION:
        raise RuntimeError(f"SGLang {sglang_version} does not match pinned {dspark.SGLANG_VERSION}")
    if flashinfer_version != dspark.FLASHINFER_VERSION:
        raise RuntimeError(
            f"FlashInfer {flashinfer_version} does not match pinned {dspark.FLASHINFER_VERSION}"
        )
    return sglang_version, flashinfer_version, torch_version


def _run_flashinfer_smoke(torch: Any) -> dict[str, Any]:
    from flashinfer.prefill import single_prefill_with_kv_cache  # ty: ignore[unresolved-import]

    query = torch.zeros((1, 1, 64), device="cuda", dtype=torch.float16)
    key = torch.zeros((1, 1, 64), device="cuda", dtype=torch.float16)
    value = torch.ones((1, 1, 64), device="cuda", dtype=torch.float16)
    try:
        result = single_prefill_with_kv_cache(query, key, value, causal=False)
        torch.cuda.synchronize()
        if not torch.allclose(result, value):
            raise RuntimeError("FlashInfer smoke kernel returned an unexpected result")
    except Exception as error:
        raise RuntimeError("FlashInfer CUDA JIT smoke test failed before model loading") from error
    return {
        "kernel": "single_prefill_with_kv_cache",
        "dtype": "float16",
        "shape": [1, 1, 64],
        "result": "passed",
    }


def _prepare_model_cache(model_cache_dir: Path) -> Path:
    if "HF_HOME" not in os.environ:
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(model_cache_dir.resolve())
    cache_name = os.environ.get("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    cache_dir = Path(cache_name).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_dir).free
    if free_bytes < MIN_CACHE_FREE_BYTES:
        required_gib = MIN_CACHE_FREE_BYTES // 1024**3
        free_gib = free_bytes / 1024**3
        raise OSError(f"model cache needs {required_gib} GiB free; found {free_gib:.1f} GiB")
    return cache_dir


def _seed_model_cache_from_hub_cache(source_hub_cache: Path, target_hub_cache: Path) -> list[str]:
    """Seed pinned Hub snapshots from a trusted cache without copying model weights."""
    source, target = _validate_model_cache_seed_roots(source_hub_cache, target_hub_cache)
    if not source.is_dir():
        raise RuntimeError(f"model cache seed is not a readable Hub cache directory: {source}")
    target.mkdir(parents=True, exist_ok=True)
    repositories = _pinned_cache_repositories()
    folder_names = _pinned_cache_folder_names(repositories)
    collisions = _cache_folder_collisions(target, folder_names)
    if collisions:
        raise RuntimeError(
            "model cache seed requires an empty pinned-repository cache; existing entries: "
            + ", ".join(collisions)
        )

    with tempfile.TemporaryDirectory(dir=target) as stage_name:
        stage = Path(stage_name)
        _stage_pinned_repositories(source, stage, repositories)
        _install_staged_cache_folders(stage, target, folder_names)
    return [revision for _repository, revision in repositories]


def _validate_model_cache_seed_roots(source: Path, target: Path) -> tuple[Path, Path]:
    """Resolve cache roots and forbid either cache from nesting inside the other."""
    source_root = source.expanduser().resolve()
    target_root = target.expanduser().resolve()
    if (
        source_root == target_root
        or source_root.is_relative_to(target_root)
        or target_root.is_relative_to(source_root)
    ):
        raise ValueError("model cache seed and writable cache must be separate directories")
    return source_root, target_root


def _pinned_cache_repositories() -> tuple[tuple[str, str], ...]:
    """List only the exact model and draft repositories supported by this runner."""
    return (
        (dspark.TARGET_MODEL, dspark.TARGET_REVISION),
        (dspark.DRAFT_MODEL, dspark.DRAFT_REVISION),
    )


def _pinned_cache_folder_names(repositories: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """Map pinned repository IDs to their Hub cache folder names."""
    return tuple(
        f"models--{repository.replace('/', '--')}" for repository, _revision in repositories
    )


def _cache_folder_collisions(target: Path, folder_names: Sequence[str]) -> list[str]:
    """Find pinned cache folders that would overwrite existing job-local data."""
    collisions = []
    for name in folder_names:
        candidate = target / name
        if candidate.exists() or candidate.is_symlink():
            collisions.append(name)
    return collisions


def _stage_pinned_repositories(
    source: Path, stage: Path, repositories: Sequence[tuple[str, str]]
) -> None:
    """Build and locally validate each pinned repo before installing any of them."""
    for repository, revision in repositories:
        _seed_pinned_repository(source, stage, repository, revision)
        _validate_staged_repository(stage, repository, revision)


def _install_staged_cache_folders(stage: Path, target: Path, folder_names: Sequence[str]) -> None:
    """Install already validated repository folders atomically into the job cache."""
    for folder_name in folder_names:
        os.replace(stage / folder_name, target / folder_name)


def _seed_pinned_repository(
    source_cache: Path, staged_cache: Path, repository: str, revision: str
) -> None:
    """Build a temporary pinned repo cache from tree metadata and source blob links."""
    folder_name = f"models--{repository.replace('/', '--')}"
    source_repo = source_cache / folder_name
    source_snapshot = source_repo / "snapshots" / revision
    source_tree = source_repo / "trees" / f"{revision}.json"
    tree_bytes, entries = _read_pinned_cache_tree(
        source_snapshot, source_tree, repository, revision
    )
    staged_repo = staged_cache / folder_name
    staged_blobs = staged_repo / "blobs"
    staged_snapshot = staged_repo / "snapshots" / revision
    staged_blobs.mkdir(parents=True)
    staged_snapshot.mkdir(parents=True)
    blob_targets: dict[str, Path] = {}
    for file_name, metadata in entries.items():
        _seed_snapshot_entry(
            file_name,
            metadata,
            source_snapshot,
            source_repo / "blobs",
            staged_blobs,
            staged_snapshot,
            blob_targets,
            repository,
            revision,
        )
    staged_tree = staged_repo / "trees" / f"{revision}.json"
    staged_tree.parent.mkdir()
    staged_tree.write_bytes(tree_bytes)


def _seed_snapshot_entry(
    file_name: Any,
    metadata: Any,
    source_snapshot: Path,
    source_blobs: Path,
    staged_blobs: Path,
    staged_snapshot: Path,
    blob_targets: dict[str, Path],
    repository: str,
    revision: str,
) -> None:
    """Check Hub blob names and sizes, then add pointers to trusted source bytes."""
    relative = _pinned_snapshot_relative_path(file_name, repository, revision)
    expected_size, expected_blob_name = _pinned_snapshot_file_metadata(
        metadata, repository, revision
    )
    source_file = source_snapshot.joinpath(*relative.parts)
    source_blob = _source_blob_for_snapshot(source_file, source_blobs, repository, revision)
    if source_blob.stat().st_size != expected_size:
        raise RuntimeError(f"pinned model cache seed is incomplete for {repository}@{revision}")
    if source_blob.name != expected_blob_name:
        raise RuntimeError(
            f"pinned model cache seed has a blob identity mismatch for {repository}@{revision}"
        )
    staged_blob = staged_blobs / source_blob.name
    _link_seed_blob(staged_blob, source_blob, blob_targets, repository, revision)
    staged_file = staged_snapshot.joinpath(*relative.parts)
    staged_file.parent.mkdir(parents=True, exist_ok=True)
    staged_file.symlink_to(os.path.relpath(staged_blob, staged_file.parent))


def _read_pinned_cache_tree(
    snapshot: Path, tree_path: Path, repository: str, revision: str
) -> tuple[bytes, dict[str, Any]]:
    """Read pinned Hub tree metadata and reject missing or unbounded metadata."""
    if not snapshot.is_dir() or not tree_path.is_file():
        raise RuntimeError(f"pinned model cache seed is incomplete for {repository}@{revision}")
    tree_bytes, tree = _read_bounded_cache_tree(tree_path, repository, revision)
    if tree.get("format_version") != 1:
        raise RuntimeError(f"pinned model cache seed is incomplete for {repository}@{revision}")
    return tree_bytes, _pinned_cache_tree_entries(tree, repository, revision)


def _pinned_cache_tree_entries(
    tree: Mapping[str, Any], repository: str, revision: str
) -> dict[str, Any]:
    """Require a nonempty file mapping in the pinned Hub tree metadata."""
    entries = tree.get("files")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError(f"pinned model cache seed is incomplete for {repository}@{revision}")
    return entries


def _read_bounded_cache_tree(
    tree_path: Path, repository: str, revision: str
) -> tuple[bytes, dict[str, Any]]:
    """Read at most the bounded Hub tree metadata size before parsing JSON."""
    try:
        with tree_path.open("rb") as stream:
            data = stream.read(MAX_CACHE_TREE_METADATA_BYTES + 1)
        if len(data) > MAX_CACHE_TREE_METADATA_BYTES:
            raise RuntimeError(f"pinned model cache seed is incomplete for {repository}@{revision}")
        tree = json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise RuntimeError(
            f"pinned model cache seed is incomplete for {repository}@{revision}"
        ) from error
    if not isinstance(tree, dict):
        raise RuntimeError(f"pinned model cache seed is incomplete for {repository}@{revision}")
    return data, tree


def _pinned_snapshot_relative_path(file_name: Any, repository: str, revision: str) -> PurePosixPath:
    """Reject tree names that could escape their Hub snapshot directory."""
    message = f"pinned model cache seed has invalid tree metadata for {repository}@{revision}"
    if not isinstance(file_name, str):
        raise RuntimeError(message)
    relative = PurePosixPath(file_name)
    if not _valid_hub_snapshot_name(file_name, relative):
        raise RuntimeError(message)
    return relative


def _valid_hub_snapshot_name(file_name: str, path: PurePosixPath) -> bool:
    """Return whether a Hub filename is a safe relative POSIX path."""
    return bool(file_name) and "\\" not in file_name and _valid_hub_snapshot_parts(path)


def _valid_hub_snapshot_parts(path: PurePosixPath) -> bool:
    """Reject absolute paths, empty paths, and path traversal components."""
    if path.is_absolute() or not path.parts:
        return False
    return _valid_hub_snapshot_components(path.parts)


def _valid_hub_snapshot_components(parts: Sequence[str]) -> bool:
    """Reject empty or traversal path components."""
    return all(part != ".." for part in parts)


def _pinned_snapshot_file_metadata(
    metadata: Any, repository: str, revision: str
) -> tuple[int, str]:
    """Return expected size and Hub cache blob name from a tree entry."""
    message = f"pinned model cache seed has invalid tree metadata for {repository}@{revision}"
    if not _valid_hub_file_metadata(metadata):
        raise RuntimeError(message)
    size = metadata["size"]
    blob_id = metadata["blob_id"]
    lfs_sha256 = metadata.get("lfs_sha256")
    return size, lfs_sha256 if lfs_sha256 is not None else blob_id


def _valid_hub_file_metadata(metadata: Any) -> bool:
    """Validate Hub tree metadata without opening or hashing model weights."""
    if not isinstance(metadata, dict):
        return False
    size = metadata.get("size")
    blob_id = metadata.get("blob_id")
    lfs_sha256 = metadata.get("lfs_sha256")
    return (
        _valid_hub_blob_size(size)
        and _nonempty_string(blob_id)
        and (lfs_sha256 is None or _nonempty_string(lfs_sha256))
    )


def _valid_hub_blob_size(size: Any) -> bool:
    """Require nonnegative integer byte size metadata, excluding bool values."""
    return isinstance(size, int) and not isinstance(size, bool) and size >= 0


def _nonempty_string(value: Any) -> bool:
    """Return whether a metadata value is a nonempty string."""
    return isinstance(value, str) and bool(value)


def _source_blob_for_snapshot(
    source_file: Path, source_blobs: Path, repository: str, revision: str
) -> Path:
    """Resolve a snapshot link to its Hub blob entry, including shared storage links."""
    if not source_file.is_symlink():
        raise RuntimeError(f"pinned model cache seed is incomplete for {repository}@{revision}")
    try:
        source_blob = Path(os.path.abspath(source_file.parent / source_file.readlink()))
    except OSError as error:
        raise RuntimeError(
            f"pinned model cache seed is incomplete for {repository}@{revision}"
        ) from error
    if source_blob.parent != Path(os.path.abspath(source_blobs)) or not source_blob.is_file():
        raise RuntimeError(f"pinned model cache seed is incomplete for {repository}@{revision}")
    return source_blob


def _link_seed_blob(
    staged_blob: Path,
    source_blob: Path,
    blob_targets: dict[str, Path],
    repository: str,
    revision: str,
) -> None:
    """Link each source blob once and reject two paths with the same cache name."""
    previous = blob_targets.get(source_blob.name)
    if previous is not None and previous != source_blob:
        raise RuntimeError(
            f"pinned model cache seed has conflicting blob names for {repository}@{revision}"
        )
    if previous is None:
        staged_blob.symlink_to(source_blob)
        blob_targets[source_blob.name] = source_blob


def _validate_staged_repository(cache: Path, repository: str, revision: str) -> None:
    """Resolve the exact seeded commit with the pinned Hub client in offline mode."""
    from huggingface_hub import snapshot_download

    try:
        resolved = snapshot_download(
            repository,
            revision=revision,
            cache_dir=cache,
            local_files_only=True,
        )
    except Exception as error:
        raise RuntimeError(
            f"pinned model cache seed is incomplete for {repository}@{revision}"
        ) from error
    expected = cache / f"models--{repository.replace('/', '--')}" / "snapshots" / revision
    if Path(resolved) != expected:
        raise RuntimeError(
            f"pinned model cache seed resolved an unexpected snapshot for {repository}@{revision}"
        )


def _cache_seed_provenance(source_hub_cache: Path | None) -> dict[str, Any] | None:
    """Record the read-only cache and exact snapshots linked into the job cache."""
    if source_hub_cache is None:
        return None
    return {
        "source_hub_cache": str(source_hub_cache.expanduser().resolve()),
        "repositories": [
            {"repository": dspark.TARGET_MODEL, "revision": dspark.TARGET_REVISION},
            {"repository": dspark.DRAFT_MODEL, "revision": dspark.DRAFT_REVISION},
        ],
        "method": "copy tree metadata; link Hub-named source blobs (trusted cache bytes)",
    }


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sglang": version("sglang"),
        "flashinfer_python": version("flashinfer-python"),
        "transformers": version("transformers"),
        "pyarrow": version("pyarrow"),
    }


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)

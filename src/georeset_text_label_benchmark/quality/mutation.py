"""Fail the CI job when mutation results contain anything but killed mutants."""

from __future__ import annotations

import re
import subprocess

RESULT_PATTERN = re.compile(r"\s*(\S+): (killed|survived|no tests|timeout)")
EQUIVALENT_MUTANTS = {
    "georeset_text_label_benchmark.join.x__validate_text_hash__mutmut_18": (
        "In _validate_text_hash, the mutant changes False to None in an exception branch; "
        "both values are false under the only subsequent `if not valid_digest`, so every "
        "input and emitted exception are identical. Invalid hexadecimal digests are tested."
    ),
    "georeset_text_label_benchmark.join.x__verify_sentence_hash__mutmut_6": (
        "In _verify_sentence_hash, Python's codec registry resolves utf-8 and UTF-8 to the "
        "same codec and identical bytes; exact sentence SHA-256 validation is tested."
    ),
    "georeset_text_label_benchmark.pipeline.x__build_run__mutmut_21": (
        "In _build_run, PyArrow accepts zstd and ZSTD as the same compression codec; "
        "the produced Parquet metadata is verified to report ZSTD."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_27": (
        "_validate_predictions checks all three sequence lengths before this zip; replacing "
        "strict=True with strict=None cannot change iteration or validation."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_30": (
        "_validate_predictions checks all three sequence lengths before this zip; omitting "
        "strict therefore iterates the same aligned values."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x__validate_predictions__mutmut_31": (
        "_validate_predictions checks all three sequence lengths before this zip; strict=False "
        "therefore iterates the same aligned values."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_20": (
        "class_breakdown calls _validate_predictions, which enforces equal lengths before this "
        "zip; strict=None has identical results."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_23": (
        "class_breakdown calls _validate_predictions, which enforces equal lengths before this "
        "zip; omitting strict has identical results."
    ),
    "georeset_text_label_benchmark.pilot.metrics.x_class_breakdown__mutmut_24": (
        "class_breakdown calls _validate_predictions, which enforces equal lengths before this "
        "zip; strict=False has identical results."
    ),
    "georeset_text_label_benchmark.pilot.runner.x_sha256_file__mutmut_9": (
        "hashlib accepts sha256 and SHA256 as case-insensitive algorithm names; the digest is "
        "identical for the same file bytes."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__sha256_json__mutmut_3": (
        "json.dumps treats ensure_ascii=None as false, so it emits the same text as "
        "ensure_ascii=False for every JSON value."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__sha256_json__mutmut_17": (
        "Python resolves utf-8 and UTF-8 to the same codec, so encoding the canonical JSON "
        "payload produces identical bytes and hashes."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__write_json_exclusive__mutmut_14": (
        "json.dump treats ensure_ascii=None as false, so its UTF-8 JSON bytes match the "
        "ensure_ascii=False output."
    ),
    "georeset_text_label_benchmark.pilot.runner.x_freeze_sample__mutmut_32": (
        "Path.mkdir tests exist_ok by truth value; None and False both reject an existing "
        "directory. A race regression verifies that True is rejected."
    ),
    "georeset_text_label_benchmark.pilot.runner.x_freeze_sample__mutmut_34": (
        "Path.mkdir defaults exist_ok to False, so omitting the explicit False preserves the "
        "same exclusive directory creation behavior."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__read_frozen_sample__mutmut_7": (
        "Python resolves UTF-8 and utf-8 to the same codec; the reader still decodes the same "
        "bytes. Tests require an explicit UTF-8 codec and exact filename."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_10": (
        "importlib.metadata normalizes distribution names case-insensitively; version('TORCH') "
        "returns the same installed version as version('torch')."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_15": (
        "importlib.metadata normalizes distribution names case-insensitively; "
        "version('TRANSFORMERS') returns the same installed version as version('transformers')."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__runtime_metadata__mutmut_20": (
        "importlib.metadata normalizes distribution names case-insensitively; "
        "version('PYARROW') returns the same installed version as version('pyarrow')."
    ),
    "georeset_text_label_benchmark.pilot.runner.x__write_outputs__mutmut_14": (
        "PyArrow treats zstd and ZSTD as aliases for the same Parquet codec; a metadata "
        "assertion checks the resulting compression."
    ),
    "georeset_text_label_benchmark.pilot.runner.x_run_pilot__mutmut_80": (
        "rank_candidates defaults top_k to 5; omitting the explicit top_k=5 argument preserves "
        "the same ranking size."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_average_pool__mutmut_28": (
        "Tensor.unsqueeze accepts +1 and 1 as the same dimension index."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_46": (
        "For torch.nn.functional.normalize with dim=1, p=None and p=2 select the same vector "
        "norm; a non-unit [3, 4] vector is checked against [0.6, 0.8]."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_49": (
        "torch.nn.functional.normalize defaults p to 2, so omitting p preserves the explicit "
        "p=2 result."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_50": (
        "torch.nn.functional.normalize defaults dim to 1, so omitting dim preserves the explicit "
        "dim=1 result."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_encode_texts__mutmut_56": (
        "torch.cat defaults dim to 0, so omitting dim preserves concatenation along the batch axis."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_21": (
        "_validate_embedding_shapes requires the candidate row count to equal the code count, "
        "so score rows and codes always have equal lengths before this zip."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_24": (
        "_validate_embedding_shapes requires the candidate row count to equal the code count, "
        "so omitting strict cannot change this zip's aligned iteration."
    ),
    "georeset_text_label_benchmark.pilot.embeddings.x_rank_candidates__mutmut_25": (
        "_validate_embedding_shapes requires the candidate row count to equal the code count, "
        "so strict=False cannot change this zip's aligned iteration."
    ),
    "georeset_text_label_benchmark.pilot.cli.x_main__mutmut_36": (
        "json.dumps treats ensure_ascii=None as false, so it emits the same text as "
        "ensure_ascii=False for CLI results."
    ),
    "georeset_text_label_benchmark.pilot.sampling.x__validate_sentence_hash__mutmut_5": (
        "Python resolves utf-8 and UTF-8 to the same codec, so the sentence bytes and SHA-256 "
        "are unchanged."
    ),
    "georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_4": (
        "json.dumps treats ensure_ascii=None as false, so the stable identity JSON bytes match "
        "ensure_ascii=False."
    ),
    "georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_11": (
        "The sample identity payload is always a JSON list, which has no key/value separator; "
        "changing the unused colon separator cannot change its serialized bytes."
    ),
    "georeset_text_label_benchmark.pilot.sampling.x__sample_id__mutmut_15": (
        "Python resolves utf-8 and UTF-8 to the same codec, so stable sample identity hashes "
        "are unchanged."
    ),
    "georeset_text_label_benchmark.pilot.sampling.x__rank_rows__mutmut_5": (
        "Python resolves ascii and ASCII to the same codec; the deterministic ranking digest "
        "is unchanged."
    ),
}


def _result_line(line: str) -> tuple[str, str]:
    match = RESULT_PATTERN.fullmatch(line)
    if match is None:
        raise ValueError(f"unrecognized mutmut result line: {line!r}")
    return match.group(1), match.group(2)


def _parse_results(output: str) -> dict[str, str]:
    records = [_result_line(line) for line in output.splitlines() if line.strip()]
    results = dict(records)
    if not records:
        raise ValueError("mutmut returned no mutation results")
    if len(results) != len(records):
        raise ValueError("mutmut returned duplicate mutant identifiers")
    return results


def _failures(results: dict[str, str]) -> list[str]:
    return [
        f"{name}: {status}"
        for name, status in sorted(results.items())
        if status != "killed" and not (status == "survived" and name in EQUIVALENT_MUTANTS)
    ]


def _killed_count(results: dict[str, str]) -> int:
    return sum(status == "killed" for status in results.values())


def _equivalent_survivors(results: dict[str, str]) -> list[str]:
    return sorted(
        name
        for name, status in results.items()
        if status == "survived" and name in EQUIVALENT_MUTANTS
    )


def _read_results() -> dict[str, str] | None:
    completed = subprocess.run(
        ["mutmut", "results", "--all=true"], capture_output=True, text=True, check=False
    )
    if completed.returncode:
        print(completed.stderr or completed.stdout)
        return None
    try:
        results = _parse_results(completed.stdout)
    except ValueError as error:
        print(str(error))
        return None
    return results


def _report_results(results: dict[str, str]) -> int:
    failures = _failures(results)
    print(f"Mutation results: {_killed_count(results)}/{len(results)} killed")
    for name in _equivalent_survivors(results):
        print(f"Equivalent mutant documented: {name}: {EQUIVALENT_MUTANTS[name]}")
    if failures:
        print("Unresolved mutation results:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


def main() -> int:
    results = _read_results()
    if results is None:
        return 1
    return _report_results(results)

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

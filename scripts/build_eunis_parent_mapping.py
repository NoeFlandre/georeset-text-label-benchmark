"""Extract the candidate ancestor closure from the pinned official EEA workbooks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import posixpath
import re
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

HUB_URL = (
    "https://www.eea.europa.eu/en/datahub/datahubitem-view/638330ea-90e6-4e41-81ea-e70f25ae7117"
)
WORKBOOKS: dict[str, dict[str, Any]] = {
    "d7ed5483bfc38804ba9c6039b73e5c75801f7e7e905b8118f55999f1b516744a": {
        "classification_release": "EUNIS terrestrial habitat classification review v1 (2021; updated 2023)",
        "classification_source": "https://doi.org/10.2909/bfe4c237-e378-4a83-ab21-b3807f96c2e2",
        "workbook_name": "EUNIS terrestrial habitat classification 2021_1 including crosswalks.xlsx",
        "workbook_sha256": "1427d22db8d77d1053ab1683aa60713bdec3d810605dd6b7bc16683b4ca14665",
        "workbook_url": "https://sdi.eea.europa.eu/datashare/s/mqtf8t3acx9RECQ/download",
        "levels": [1, 2, 3],
    },
    "8d50548ad4e8dc5aa1bbae07282441d9137b78a373ee46c43cfbbd5fb0e9322a": {
        "classification_release": "EUNIS marine habitat classification review v1 (2022)",
        "classification_source": "https://doi.org/10.2909/8a5eccda-e6ea-4018-a373-5c76a8eeec78",
        "workbook_name": "EUNIS marine habitat classification 2022 including crosswalks.xlsx",
        "workbook_sha256": "09b47b35c1b6f41c3cff3365bc55e4bc42770945b9103e369b5cdeda9c1aa429",
        "workbook_url": "https://sdi.eea.europa.eu/datashare/s/Aaj3DLQoiFHZCW5/download",
        "levels": [1, 2, 3, 4],
    },
}
_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "package": "http://schemas.openxmlformats.org/package/2006/relationships",
}
_TERRESTRIAL_SOURCE = "d7ed5483bfc38804ba9c6039b73e5c75801f7e7e905b8118f55999f1b516744a"
_MARINE_SOURCE = "8d50548ad4e8dc5aa1bbae07282441d9137b78a373ee46c43cfbbd5fb0e9322a"


def main() -> int:
    args = _parser().parse_args()
    result = build_mapping(
        args.candidate_csv,
        args.frozen_sample,
        args.terrestrial_workbook,
        args.marine_workbook,
    )
    _write_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256_file(args.output),
                "candidate_count": result["verification"]["candidate_leaf_count"],
                "ancestor_counts_by_level": result["verification"]["ancestor_counts_by_level"],
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--frozen-sample", type=Path, required=True)
    parser.add_argument("--terrestrial-workbook", type=Path, required=True)
    parser.add_argument("--marine-workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_mapping(
    candidate_csv: Path,
    frozen_sample: Path,
    terrestrial_workbook: Path,
    marine_workbook: Path,
) -> dict[str, Any]:
    candidate_hash = _sha256_file(candidate_csv)
    candidate_rows = _read_candidate_csv(candidate_csv)
    workbooks = {
        _TERRESTRIAL_SOURCE: terrestrial_workbook,
        _MARINE_SOURCE: marine_workbook,
    }
    all_chains: dict[str, dict[str, Any]] = {}
    classification_records: list[dict[str, Any]] = []
    for source_hash, metadata in sorted(WORKBOOKS.items()):
        workbook = workbooks[source_hash]
        _validate_workbook_hash(workbook, metadata["workbook_sha256"])
        source_candidates = [row for row in candidate_rows if row["source_sha256"] == source_hash]
        source_nodes = _read_workbook_nodes(workbook)
        chains, used_nodes = _candidate_chains(source_candidates, source_nodes, source_hash)
        all_chains.update(chains)
        classification_records.append(
            {
                **{
                    key: metadata[key]
                    for key in (
                        "classification_release",
                        "classification_source",
                        "workbook_name",
                        "workbook_sha256",
                        "workbook_url",
                        "levels",
                    )
                },
                "source_sha256": source_hash,
                "nodes": sorted(
                    used_nodes.values(), key=lambda node: (node["level"], node["code"])
                ),
            }
        )
    _validate_all_candidates(candidate_rows, all_chains)
    verification = _verification(candidate_rows, all_chains)
    verification["sample_coverage"] = _sample_coverage(
        frozen_sample, candidate_hash, candidate_rows, all_chains
    )
    return {
        "format_version": 1,
        "source": {
            "publisher": "European Environment Agency",
            "datahub_url": HUB_URL,
            "mapping_method": (
                "Parent is the immediately preceding lower-level classification row in the same "
                "official workbook sheet; hierarchy codes are never sliced by character position."
            ),
            "license": "CC-BY-4.0",
        },
        "candidate_csv_sha256": candidate_hash,
        "classifications": classification_records,
        "candidate_chains": [all_chains[code] for code in sorted(all_chains)],
        "verification": verification,
    }


def _read_candidate_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "eunis_code",
            "eunis_name",
            "classification_release",
            "classification_source",
            "source_sha256",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("candidate CSV is missing the pinned EUNIS provenance fields")
        rows = [dict(row) for row in reader]
    codes = [row["eunis_code"] for row in rows]
    if len(codes) != len(set(codes)) or len(codes) != 158:
        raise ValueError("pinned candidate CSV must contain 158 unique codes")
    for row in rows:
        metadata = WORKBOOKS.get(row["source_sha256"])
        if metadata is None or any(
            row[field] != metadata[field]
            for field in ("classification_release", "classification_source")
        ):
            raise ValueError(f"candidate classification provenance mismatch: {row['eunis_code']}")
    return rows


def _validate_workbook_hash(path: Path, expected: str) -> None:
    if _sha256_file(path) != expected:
        raise ValueError(f"official EEA workbook SHA-256 mismatch: {path}")


def _read_workbook_nodes(path: Path) -> dict[str, dict[str, Any]]:
    with zipfile.ZipFile(path) as workbook:
        shared = _shared_strings(workbook)
        sheets = _workbook_sheets(workbook)
        result: dict[str, dict[str, Any]] = {}
        for name, target in sheets:
            rows = _worksheet_rows(workbook, target, shared)
            _append_sheet_nodes(name, rows, result)
    return result


def _shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(text.text or "" for text in item.findall(".//main:t", _NS)) for item in root]


def _workbook_sheets(workbook: zipfile.ZipFile) -> list[tuple[str, str]]:
    root = ET.fromstring(workbook.read("xl/workbook.xml"))
    relationships = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
    sheets = []
    for sheet in root.findall(".//main:sheets/main:sheet", _NS):
        relation_id = sheet.attrib[f"{{{_NS['rel']}}}id"]
        target = targets[relation_id].lstrip("/")
        if not target.startswith("xl/"):
            target = posixpath.normpath(posixpath.join("xl", target))
        sheets.append((sheet.attrib["name"], target))
    return sheets


def _worksheet_rows(
    workbook: zipfile.ZipFile, target: str, shared: Sequence[str]
) -> list[list[str | None]]:
    root = ET.fromstring(workbook.read(target))
    rows: list[list[str | None]] = []
    for row in root.findall(".//main:sheetData/main:row", _NS):
        cells: dict[int, str | None] = {}
        for cell in row.findall("main:c", _NS):
            index = _column_index(cell.attrib["r"])
            cells[index] = _cell_value(cell, shared)
        rows.append([cells.get(index) for index in range(max(cells, default=-1) + 1)])
    return rows


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"invalid spreadsheet cell reference: {reference}")
    index = 0
    for letter in letters.group(0):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _cell_value(cell: ET.Element, shared: Sequence[str]) -> str | None:
    value = cell.find("main:v", _NS)
    if value is not None:
        if cell.attrib.get("t") == "s":
            return shared[int(value.text or "0")]
        return value.text
    inline = cell.find("main:is", _NS)
    if inline is not None:
        return "".join(text.text or "" for text in inline.findall(".//main:t", _NS))
    return None


def _append_sheet_nodes(
    sheet_name: str,
    rows: Sequence[Sequence[str | None]],
    result: dict[str, dict[str, Any]],
) -> None:
    if not rows:
        return
    columns = _hierarchy_columns(rows[0])
    if columns is None:
        return
    stack: list[tuple[int, str]] = []
    for row in rows[1:]:
        record = _sheet_node(row, columns, sheet_name, stack)
        if record is None:
            continue
        _register_sheet_node(record, result)
        stack.append((record["level"], record["code"]))


def _hierarchy_columns(header: Sequence[str | None]) -> tuple[int, int, int] | None:
    headers = {str(value).strip().lower(): index for index, value in enumerate(header) if value}
    level = headers.get("level")
    code = next((index for key, index in headers.items() if key.startswith("code")), None)
    name = next((index for key, index in headers.items() if key.startswith("name")), None)
    if level is None or code is None or name is None:
        return None
    return level, code, name


def _sheet_node(
    row: Sequence[str | None],
    columns: tuple[int, int, int],
    sheet_name: str,
    stack: list[tuple[int, str]],
) -> dict[str, Any] | None:
    level_column, code_column, name_column = columns
    if max(columns) >= len(row):
        return None
    level = _integer_value(row[level_column])
    code_value = row[code_column]
    if level is None or not code_value:
        return None
    code = code_value.strip()
    while stack and stack[-1][0] >= level:
        stack.pop()
    parent = stack[-1][1] if stack else None
    if (level == 1) != (parent is None):
        raise ValueError(f"workbook hierarchy parent is missing at {sheet_name}:{code}")
    name_value = row[name_column]
    name = name_value.strip() if isinstance(name_value, str) else ""
    return {"code": code, "parent_code": parent, "level": level, "name": name}


def _register_sheet_node(record: dict[str, Any], result: dict[str, dict[str, Any]]) -> None:
    code = str(record["code"])
    existing = result.get(code)
    if existing is not None and existing != record:
        raise ValueError(f"official workbook repeats a code with different hierarchy: {code}")
    result[code] = record


def _integer_value(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _candidate_chains(
    candidates: Sequence[Mapping[str, str]],
    nodes: Mapping[str, Mapping[str, Any]],
    source_hash: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    chains: dict[str, dict[str, Any]] = {}
    used: dict[str, dict[str, Any]] = {}
    for row in candidates:
        code = row["eunis_code"]
        node = nodes.get(code)
        if node is None or node["name"] != row["eunis_name"].strip():
            raise ValueError(f"candidate code or name does not match official workbook: {code}")
        chain = _chain_for(code, nodes)
        for ancestor in chain:
            key = str(ancestor["code"])
            used[key] = dict(ancestor)
        chains[code] = {
            "code": code,
            "source_sha256": source_hash,
            "leaf_level": node["level"],
            "ancestor_chain": [ancestor["code"] for ancestor in chain],
        }
    return chains, used


def _chain_for(code: str, nodes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    current: str | None = code
    while current is not None:
        node = nodes.get(current)
        if node is None:
            raise ValueError(f"official workbook parent is missing in candidate chain: {current}")
        chain.append(dict(node))
        current = node["parent_code"] if isinstance(node["parent_code"], str) else None
    chain.reverse()
    return chain


def _validate_all_candidates(
    rows: Sequence[Mapping[str, str]], chains: Mapping[str, Mapping[str, Any]]
) -> None:
    if {row["eunis_code"] for row in rows} != set(chains):
        raise ValueError("not all pinned candidates have a verified official parent chain")
    by_source = Counter(
        (row["source_sha256"], chains[row["eunis_code"]]["leaf_level"]) for row in rows
    )
    expected = Counter({(_TERRESTRIAL_SOURCE, 3): 149, (_MARINE_SOURCE, 4): 9})
    if by_source != expected:
        raise ValueError(f"candidate source/rank counts differ from audit: {by_source}")


def _verification(
    rows: Sequence[Mapping[str, str]], chains: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    counts = _ancestor_counts(rows, chains)
    _validate_ancestor_counts(counts)
    examples = _verified_examples(chains)
    leaf_counts = _candidate_leaf_counts(rows, chains)
    return {
        "candidate_leaf_count": len(rows),
        "candidate_leaf_counts_by_source_and_level": leaf_counts,
        "ancestor_counts_by_level": counts,
        "verified_examples": examples,
        "excluded_from_level_4": {
            "candidate_count": 149,
            "reason": "terrestrial reference codes are official level 3, so no level 4 target exists",
        },
        "audit_count_note": (
            "The supplied terrestrial level-2 parent counts enumerate 27 unique codes; the EEA "
            "workbook confirms 27 terrestrial parents plus marine MA2, for 28 combined labels."
        ),
    }


def _ancestor_counts(
    rows: Sequence[Mapping[str, str]], chains: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    labels_by_level: dict[int, set[tuple[str, str]]] = {}
    for row in rows:
        chain = chains[row["eunis_code"]]["ancestor_chain"]
        if not isinstance(chain, list):
            raise ValueError("candidate chain must be a list")
        for level, code in enumerate(chain, start=1):
            labels_by_level.setdefault(level, set()).add((row["source_sha256"], code))
    return {str(level): len(codes) for level, codes in sorted(labels_by_level.items())}


def _validate_ancestor_counts(counts: Mapping[str, int]) -> None:
    expected = {"1": 7, "2": 28, "3": 152, "4": 9}
    if counts != expected:
        raise ValueError(f"official ancestor counts differ from audit: {counts}")


def _verified_examples(chains: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    examples = {
        "N11": ["N", "N1", "N11"],
        "MA221": ["M", "MA2", "MA22", "MA221"],
    }
    if any(chains[code]["ancestor_chain"] != chain for code, chain in examples.items()):
        raise ValueError("official workbook chains do not match the audited examples")
    return examples


def _candidate_leaf_counts(
    rows: Sequence[Mapping[str, str]], chains: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    counts = Counter(
        (row["source_sha256"], chains[row["eunis_code"]]["leaf_level"]) for row in rows
    )
    return {f"{source}:{level}": count for (source, level), count in sorted(counts.items())}


def _sample_coverage(
    path: Path,
    candidate_hash: str,
    candidates: Sequence[Mapping[str, str]],
    chains: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _sample, rows, id_digest = _validated_sample(path, candidate_hash, candidates)
    decisions = Counter(row["decision"] for row in rows)
    level_coverage = _sample_level_coverage(rows, chains)
    return {
        "frozen_sample_sha256": _sha256_file(path),
        "sample_ids_sha256": id_digest,
        "sample_size": len(rows),
        "decision_counts": dict(sorted(decisions.items())),
        "reference_eligibility_by_level": level_coverage,
    }


def _validated_sample(
    path: Path, candidate_hash: str, candidates: Sequence[Mapping[str, str]]
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    sample = json.loads(path.read_text(encoding="utf-8"))
    if sample.get("candidate_labels", {}).get("sha256") != candidate_hash:
        raise ValueError("frozen sample candidate table checksum mismatch")
    rows = sample.get("selected_rows")
    selection = sample.get("selection")
    if (
        not isinstance(rows, list)
        or not all(isinstance(row, dict) for row in rows)
        or not isinstance(selection, Mapping)
        or not rows
    ):
        raise ValueError("frozen sample is missing selected rows or selection metadata")
    codes = {row["eunis_code"] for row in candidates}
    if any(
        row.get("eunis_code") not in codes or row.get("decision") not in {"yes", "no"}
        for row in rows
    ):
        raise ValueError("frozen sample has an unknown candidate code or decision")
    ids = [row.get("sample_id") for row in rows]
    id_digest = hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if id_digest != selection.get("sample_ids_sha256"):
        raise ValueError("frozen sample ID digest mismatch")
    return sample, rows, id_digest


def _sample_level_coverage(
    rows: Sequence[Mapping[str, Any]], chains: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, int]]:
    return {str(level): _sample_level_row(level, rows, chains) for level in range(1, 5)}


def _sample_level_row(
    level: int,
    rows: Sequence[Mapping[str, Any]],
    chains: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    counts = Counter(
        (row["decision"], chains[row["eunis_code"]]["leaf_level"] >= level) for row in rows
    )
    eligible = sum(counts[decision, True] for decision in ("yes", "no"))
    excluded = sum(counts[decision, False] for decision in ("yes", "no"))
    return {
        "yes_eligible_n": counts["yes", True],
        "no_eligible_n": counts["no", True],
        "eligible_n": eligible,
        "excluded_shallower_reference_count": excluded,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

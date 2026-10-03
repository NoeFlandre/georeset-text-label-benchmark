"""Fail-closed CRAP scoring using conventional McCabe decision points."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

THRESHOLD = 6.0
CONTROL_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler)
DECISION_NODES = (*CONTROL_NODES, ast.IfExp)
NESTED_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
OWNED_SOURCE_DIRS = (Path("src/georeset_text_label_benchmark"),)


def _scope_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterable[ast.AST]:
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, NESTED_SCOPE_NODES):
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def _decision_points(node: ast.AST) -> int:
    if isinstance(node, ast.comprehension):
        return 1 + len(node.ifs)
    if isinstance(node, DECISION_NODES):
        return 1
    if isinstance(node, ast.BoolOp):
        return len(node.values) - 1
    return _match_decision_points(node)


def _match_decision_points(node: ast.AST) -> int:
    return max(0, len(node.cases) - 1) if isinstance(node, ast.Match) else 0


def _statement_lines(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    return {node.lineno for node in _scope_nodes(function) if isinstance(node, ast.stmt)}


def _complexity(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return 1 + sum(_decision_points(node) for node in _scope_nodes(function))


def _function_nodes(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    found: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return found


def _coverage_file(coverage: dict[str, Any], path: Path) -> dict[str, Any]:
    files = coverage.get("files")
    if not isinstance(files, dict):
        raise ValueError("coverage report is missing its files mapping")
    key = path.as_posix()
    if key not in files:
        raise ValueError(f"coverage report does not contain {key}")
    return files[key]


def _function_crap(function: ast.FunctionDef | ast.AsyncFunctionDef, executed: set[int]) -> float:
    lines = _statement_lines(function)
    coverage = len(lines & executed) / len(lines)
    complexity = _complexity(function)
    return complexity**2 * (1 - coverage) ** 3 + complexity


def _existing_directories(source_dirs: Sequence[Path]) -> list[Path]:
    directories: list[Path] = []
    missing: list[Path] = []
    for path in source_dirs:
        if path.is_dir():
            directories.append(path)
        else:
            missing.append(path)
    if missing:
        names = ", ".join(sorted(path.as_posix() for path in missing))
        raise ValueError("owned source directory missing: " + names)
    return directories


def _python_files(directories: Sequence[Path]) -> list[Path]:
    return sorted(path for directory in directories for path in directory.rglob("*.py"))


def _source_files(source_dirs: Sequence[Path]) -> list[Path]:
    files = _python_files(_existing_directories(source_dirs))
    if not files:
        raise ValueError("no owned Python source files found for CRAP scoring")
    return files


def _file_failures(path: Path, coverage: dict[str, Any]) -> list[str]:
    report = _coverage_file(coverage, path)
    executed = set(report["executed_lines"])
    tree = ast.parse(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for function in sorted(_function_nodes(tree), key=lambda node: node.lineno):
        score = _function_crap(function, executed)
        description = f"{path}:{function.lineno} {function.name}: CRAP={score:.3f}"
        print(description)
        if score >= THRESHOLD:
            failures.append(description)
    return failures


def main(report_path: str | Path, source_dirs: Sequence[Path] = OWNED_SOURCE_DIRS) -> int:
    coverage = json.loads(Path(report_path).read_text(encoding="utf-8"))
    failures = [
        failure for path in _source_files(source_dirs) for failure in _file_failures(path, coverage)
    ]
    if failures:
        print(f"{len(failures)} function(s) must have CRAP strictly below {THRESHOLD:g}")
        return 1
    return 0

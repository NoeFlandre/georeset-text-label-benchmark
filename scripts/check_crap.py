"""Fail when a production function has CRAP score greater than or equal to six."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

THRESHOLD = 6.0
CONTROL_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp)
NESTED_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _statement_lines(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    lines: set[int] = set()
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, NESTED_SCOPE_NODES):
            continue
        if isinstance(node, ast.stmt):
            lines.add(node.lineno)
        pending.extend(ast.iter_child_nodes(node))
    return lines


def _complexity(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    score = 1
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, NESTED_SCOPE_NODES):
            continue
        if isinstance(node, CONTROL_NODES):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
        elif isinstance(node, ast.Match):
            score += max(0, len(node.cases) - 1)
        pending.extend(ast.iter_child_nodes(node))
    return score


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
    key = path.as_posix()
    files = coverage.get("files", {})
    if key not in files:
        raise ValueError(f"coverage report does not contain {key}")
    return files[key]


def _function_crap(function: ast.FunctionDef | ast.AsyncFunctionDef, executed: set[int]) -> float:
    lines = _statement_lines(function)
    coverage = len(lines & executed) / len(lines) if lines else 1.0
    complexity = _complexity(function)
    return complexity**2 * (1 - coverage) ** 3 + complexity


def main(report_path: str) -> int:
    coverage = json.loads(Path(report_path).read_text(encoding="utf-8"))
    failures: list[str] = []
    for path in sorted(Path("src/georeset_text_label_benchmark").rglob("*.py")):
        if path.as_posix() not in coverage.get("files", {}):
            continue
        report = _coverage_file(coverage, path)
        executed = set(report["executed_lines"])
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in sorted(_function_nodes(tree), key=lambda node: node.lineno):
            score = _function_crap(function, executed)
            description = f"{path}:{function.lineno} {function.name}: CRAP={score:.3f}"
            print(description)
            if score >= THRESHOLD:
                failures.append(description)
    if failures:
        print(f"{len(failures)} function(s) must have CRAP strictly below {THRESHOLD:g}")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_crap.py coverage.json")
    raise SystemExit(main(sys.argv[1]))

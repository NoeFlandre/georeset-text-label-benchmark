"""Guard the removal of the E5 zero-shot ranking pilot."""

from __future__ import annotations

import argparse
import importlib.util

import pytest

from georeset_text_label_benchmark.pilot import cli, metrics, protocol, runner


def test_embeddings_module_is_removed() -> None:
    assert importlib.util.find_spec("georeset_text_label_benchmark.pilot.embeddings") is None


@pytest.mark.parametrize("name", ["classification_report", "class_breakdown", "group_breakdown"])
def test_top5_ranking_metrics_are_removed(name: str) -> None:
    assert not hasattr(metrics, name)


@pytest.mark.parametrize("name", ["MODEL_REPOSITORY", "MODEL_REVISION", "MAX_LENGTH", "BATCH_SIZE"])
def test_e5_protocol_constants_are_removed(name: str) -> None:
    assert not hasattr(protocol, name)


def test_run_pilot_is_removed() -> None:
    assert not hasattr(runner, "run_pilot")


def test_cli_has_no_e5_run_subcommand() -> None:
    parser = cli._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert "run" not in subparsers.choices

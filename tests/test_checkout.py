"""Real-Git tests for binding DSpark commit labels to the checkout that executes."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from georeset_text_label_benchmark.pilot import checkout

_GIT_IDENTITY = (
    "-c",
    "user.name=Pilot Test",
    "-c",
    "user.email=pilot@example.invalid",
    "-c",
    "commit.gpgsign=false",
)
_OTHER_COMMIT = "f" * 40


def _exact(message: str) -> str:
    return f"^{re.escape(message)}$"


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *_GIT_IDENTITY, *arguments],
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """A clean repository with one committed file, used as the executing checkout."""
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "pilot.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "pilot.py")
    _git(root, "commit", "-q", "-m", "pilot code")
    monkeypatch.setattr(checkout, "checkout_root", lambda: root)
    return root, _git(root, "rev-parse", "HEAD")


def test_clean_checkout_whose_head_is_both_labels_is_admitted(
    repository: tuple[Path, str],
) -> None:
    _, head = repository

    checkout.require_committed_checkout(head, head)


@pytest.mark.parametrize(
    ("computation", "validation"),
    [("other", "head"), ("head", "other"), ("other", "other")],
)
def test_labels_that_differ_from_head_are_rejected(
    repository: tuple[Path, str], computation: str, validation: str
) -> None:
    _, head = repository
    labels = {"head": head, "other": _OTHER_COMMIT}

    with pytest.raises(
        ValueError,
        match=_exact(f"Checkout commit {head} does not match the recorded commit labels."),
    ):
        checkout.require_committed_checkout(labels[computation], labels[validation])


def test_modified_tracked_file_is_rejected_as_an_unclean_checkout(
    repository: tuple[Path, str],
) -> None:
    root, head = repository
    (root / "pilot.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=_exact(
            "Use a clean, committed checkout so the run manifest identifies the executed code."
        ),
    ):
        checkout.require_committed_checkout(head, head)


def test_untracked_file_is_rejected_as_an_unclean_checkout(
    repository: tuple[Path, str],
) -> None:
    root, head = repository
    (root / "scratch.py").write_text("print('untracked')\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=_exact(
            "Use a clean, committed checkout so the run manifest identifies the executed code."
        ),
    ):
        checkout.require_committed_checkout(head, head)


def test_directory_that_is_not_a_git_checkout_is_not_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setattr(checkout, "checkout_root", lambda: plain)

    with pytest.raises(ValueError, match=_exact("Could not verify a clean, committed checkout.")):
        checkout.require_committed_checkout(_OTHER_COMMIT, _OTHER_COMMIT)


def test_missing_git_executable_is_reported_as_unverified(
    repository: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, head = repository
    monkeypatch.setenv("PATH", "")

    with pytest.raises(ValueError, match=_exact("Could not verify a clean, committed checkout.")):
        checkout.require_committed_checkout(head, head)


def test_checkout_root_is_the_repository_that_contains_this_package() -> None:
    root = checkout.checkout_root()

    assert (root / "pyproject.toml").is_file()
    assert (root / "src/georeset_text_label_benchmark/pilot/checkout.py").samefile(
        Path(checkout.__file__)
    )

"""Bind DSpark run labels to the clean checkout that is executing."""

from __future__ import annotations

import subprocess
from pathlib import Path


def require_committed_checkout(computation_commit: str, validation_commit: str) -> None:
    """Require a clean checkout whose HEAD is exactly the recorded commit labels.

    The manifest records these labels as the code that produced the outputs. Without this
    check, a run from one checkout could publish under another checkout's labels.
    """
    root = checkout_root()
    if _git_stdout(root, "status", "--porcelain", "--untracked-files=normal").strip():
        raise ValueError(
            "Use a clean, committed checkout so the run manifest identifies the executed code."
        )
    head = _git_stdout(root, "rev-parse", "HEAD").strip()
    if {computation_commit, validation_commit} != {head}:
        raise ValueError(f"Checkout commit {head} does not match the recorded commit labels.")


def checkout_root() -> Path:
    """Return the source checkout that contains the imported pilot package."""
    return Path(__file__).resolve().parents[3]


def _git_stdout(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, check=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Could not verify a clean, committed checkout.") from error
    return completed.stdout

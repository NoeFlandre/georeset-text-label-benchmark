"""Stage immutable pilot outputs before exclusive publication."""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Never


@contextmanager
def staged_directory(parent: Path, *, prefix: str) -> Iterator[Path]:
    """Create a private staging directory on the same filesystem as its destination."""
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    try:
        yield staging
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def publish_files(staging: Path, destination: Path, names: Sequence[str]) -> None:
    """Link staged files into a run directory without replacing existing files.

    Published links are intentionally never rolled back: unlinking by pathname
    cannot be made conditional on the inode, so a competing writer could replace
    a link between an ownership check and unlink. The manifest is published last
    and remains the commit marker for the complete output set.
    """
    for name in names:
        source = staging / name
        target = destination / name
        os.link(source, target)


def publish_directory(staging: Path, destination: Path) -> None:
    """Atomically move a staged directory into place, failing if the path exists."""
    if os.name == "nt":
        os.rename(staging, destination)
        return

    if sys.platform == "darwin":
        result = _rename_macos_exclusive(staging, destination)
    elif sys.platform.startswith("linux"):
        result = _rename_linux_exclusive(staging, destination)
    else:
        _raise_unsupported_rename()

    if result != 0:
        _raise_rename_error(destination)


def _rename_macos_exclusive(staging: Path, destination: Path) -> int:
    library = ctypes.CDLL(None, use_errno=True)
    rename_exclusive = getattr(library, "renamex_np", None)
    if rename_exclusive is None:
        _raise_unsupported_rename()
    return rename_exclusive(os.fsencode(staging), os.fsencode(destination), 0x00000004)


def _rename_linux_exclusive(staging: Path, destination: Path) -> int:
    library = ctypes.CDLL(None, use_errno=True)
    rename_exclusive = getattr(library, "renameat2", None)
    if rename_exclusive is None:
        _raise_unsupported_rename()
    return rename_exclusive(
        -100,
        os.fsencode(staging),
        -100,
        os.fsencode(destination),
        0x00000001,
    )


def _raise_unsupported_rename() -> Never:
    raise OSError(errno.ENOTSUP, "exclusive directory rename is unavailable")


def _raise_rename_error(destination: Path) -> None:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error), os.fspath(destination))

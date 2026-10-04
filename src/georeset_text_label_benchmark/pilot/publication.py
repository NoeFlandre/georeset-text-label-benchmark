"""Stage pilot outputs and publish them with no-clobber file operations."""

from __future__ import annotations

import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def staged_directory(
    parent: Path, *, prefix: str, preserve_on_error: bool = False
) -> Iterator[Path]:
    """Create a private stage beside its destination; optionally retain it on error."""
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    completed = False
    try:
        yield staging
        completed = True
    finally:
        if staging.exists() and (completed or not preserve_on_error):
            shutil.rmtree(staging)


def ensure_publication_supported(parent: Path) -> None:
    """Probe the destination mount's exclusive mkdir and hard-link semantics."""
    if not parent.is_dir():
        raise NotADirectoryError(os.fspath(parent))
    with tempfile.TemporaryDirectory(prefix=".georeset-publication-probe-", dir=parent) as raw:
        probe = Path(raw)
        source = probe / "source"
        link = probe / "link"
        source.write_bytes(b"publication capability probe")
        _probe_exclusive_link(source, link, parent)
        directory = probe / "directory"
        directory.mkdir()
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        else:
            raise OSError(
                errno.ENOTSUP,
                "output filesystem must support exclusive directory creation",
                os.fspath(parent),
            )


def find_staged_directory(parent: Path, *, prefix: str) -> Path | None:
    """Find one retained transaction stage, refusing ambiguous recovery candidates."""
    if not parent.is_dir():
        return None
    candidates = _staged_candidates(parent, prefix)
    if len(candidates) > 1:
        _raise_ambiguous_stages(candidates)
    return candidates[0] if candidates else None


def _staged_candidates(parent: Path, prefix: str) -> list[Path]:
    return sorted(
        path
        for path in parent.iterdir()
        if path.name.startswith(prefix) and path.is_dir() and not path.is_symlink()
    )


def _raise_ambiguous_stages(candidates: Sequence[Path]) -> None:
    paths = ", ".join(os.fspath(path) for path in candidates)
    raise RuntimeError(f"multiple staged publications require inspection: {paths}")


def _probe_exclusive_link(source: Path, link: Path, parent: Path) -> None:
    try:
        os.link(source, link)
    except OSError as error:
        raise OSError(
            error.errno,
            "output filesystem must support exclusive hard links",
            os.fspath(parent),
        ) from error
    source_stat = source.stat(follow_symlinks=False)
    link_stat = link.stat(follow_symlinks=False)
    if (source_stat.st_dev, source_stat.st_ino) != (link_stat.st_dev, link_stat.st_ino):
        raise OSError(
            errno.ENOTSUP,
            "output filesystem hard links do not preserve file identity",
            os.fspath(parent),
        )
    try:
        os.link(source, link)
    except FileExistsError:
        return
    except OSError as error:
        raise OSError(
            error.errno,
            "output filesystem must support exclusive hard links",
            os.fspath(parent),
        ) from error
    raise OSError(
        errno.ENOTSUP,
        "output filesystem hard links must fail when the destination exists",
        os.fspath(parent),
    )


def publish_files(staging: Path, destination: Path, names: Sequence[str]) -> None:
    """Link staged files without replacement, resuming only links to this stage."""
    for name in names:
        source = staging / name
        target = destination / name
        if _same_file(source, target):
            continue
        try:
            os.link(source, target)
        except FileExistsError:
            if not _same_file(source, target):
                raise


def publish_directory(staging: Path, destination: Path) -> None:
    """Create an output directory and publish staged files with an explicit commit marker.

    This avoids rename flags such as Linux ``RENAME_NOREPLACE``, which some
    network filesystems reject. Existing entries are resumable only when they
    are hard links to the corresponding staged files.
    """
    names = _staged_file_names(staging)
    if not os.path.lexists(destination):
        try:
            destination.mkdir()
        except FileExistsError:
            _require_owned_partial_directory(staging, destination, names)
    else:
        _require_owned_partial_directory(staging, destination, names)
    publish_files(staging, destination, names)


def _staged_file_names(staging: Path) -> list[str]:
    paths = list(staging.iterdir())
    _require_regular_staged_files(paths)
    return sorted(
        (path.name for path in paths),
        key=lambda name: (name in {"manifest.json", "dspark_manifest.json"}, name),
    )


def _require_regular_staged_files(paths: Sequence[Path]) -> None:
    if not paths:
        raise ValueError("staged publication must contain regular files only")
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("staged publication must contain regular files only")


def _require_owned_partial_directory(
    staging: Path, destination: Path, names: Sequence[str]
) -> None:
    _require_real_destination(destination)
    if not _owns_staged_entries(staging, destination, names):
        raise FileExistsError(
            errno.EEXIST, "output destination already exists", os.fspath(destination)
        )


def _require_real_destination(destination: Path) -> None:
    if not destination.is_dir() or destination.is_symlink():
        raise FileExistsError(
            errno.EEXIST, "output destination already exists", os.fspath(destination)
        )


def _owns_staged_entries(staging: Path, destination: Path, names: Sequence[str]) -> bool:
    entries = list(destination.iterdir())
    if not entries:
        return False
    for entry in entries:
        if entry.name not in names or not _same_file(staging / entry.name, entry):
            return False
    return True


def _same_file(source: Path, target: Path) -> bool:
    try:
        source_stat = source.lstat()
        target_stat = target.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(target_stat.st_mode) and (source_stat.st_dev, source_stat.st_ino) == (
        target_stat.st_dev,
        target_stat.st_ino,
    )

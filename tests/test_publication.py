"""Tests for staged, exclusive publication of pilot artifacts."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from georeset_text_label_benchmark.pilot import publication
from georeset_text_label_benchmark.pilot.publication import (
    publish_directory,
    publish_files,
    staged_directory,
)


def test_file_publication_leaves_no_commit_marker_when_a_path_already_exists(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    (destination / "metrics.json").write_bytes(b"competing writer")

    with staged_directory(destination, prefix=".stage-") as staging:
        (staging / "predictions.parquet").write_bytes(b"staged predictions")
        (staging / "metrics.json").write_bytes(b"staged metrics")

        with pytest.raises(FileExistsError):
            publish_files(
                staging,
                destination,
                ["predictions.parquet", "metrics.json", "manifest.json"],
            )

        assert (destination / "predictions.parquet").read_bytes() == b"staged predictions"
        assert (destination / "metrics.json").read_bytes() == b"competing writer"
        assert not (destination / "manifest.json").exists()
        assert (staging / "predictions.parquet").read_bytes() == b"staged predictions"


def test_directory_publication_never_replaces_an_empty_racing_directory(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    staging = tmp_path / ".run-staging"
    staging.mkdir()
    (staging / "manifest.json").write_bytes(b"complete staged manifest")

    with pytest.raises(FileExistsError):
        publish_directory(staging, destination)

    assert list(destination.iterdir()) == []
    assert (staging / "manifest.json").read_bytes() == b"complete staged manifest"


def test_windows_directory_publication_uses_exclusive_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, Path]] = []
    staging = tmp_path / "stage"
    destination = tmp_path / "run"

    with monkeypatch.context() as scoped:
        scoped.setattr(publication.os, "name", "nt")
        scoped.setattr(
            publication.os, "rename", lambda source, target: calls.append((source, target))
        )
        publication.publish_directory(staging, destination)

    assert calls == [(staging, destination)]


def test_macos_directory_publication_uses_exclusive_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[bytes, bytes, int]] = []
    library_loads: list[tuple[tuple[object, ...], dict[str, object]]] = []
    library = SimpleNamespace(
        renamex_np=lambda source, target, flags: calls.append((source, target, flags)) or 0
    )

    def load_library(*args: object, **kwargs: object) -> SimpleNamespace:
        library_loads.append((args, kwargs))
        return library

    with monkeypatch.context() as scoped:
        scoped.setattr(publication.sys, "platform", "darwin")
        scoped.setattr(publication.ctypes, "CDLL", load_library)
        publication.publish_directory(tmp_path / "stage", tmp_path / "run")

    assert library_loads == [((None,), {"use_errno": True})]
    assert len(calls) == 1
    source, target, flags = calls[0]
    assert source.endswith(b"stage")
    assert target.endswith(b"run")
    assert flags == 0x00000004


def test_linux_directory_publication_uses_renameat2_noreplace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, ...]] = []
    library_loads: list[tuple[tuple[object, ...], dict[str, object]]] = []
    library = SimpleNamespace(renameat2=lambda *args: calls.append(args) or 0)

    def load_library(*args: object, **kwargs: object) -> SimpleNamespace:
        library_loads.append((args, kwargs))
        return library

    staging = tmp_path / "stage"
    destination = tmp_path / "run"
    with monkeypatch.context() as scoped:
        scoped.setattr(publication.os, "name", "posix")
        scoped.setattr(publication.sys, "platform", "linux")
        scoped.setattr(publication.ctypes, "CDLL", load_library)
        publication.publish_directory(staging, destination)

    assert library_loads == [((None,), {"use_errno": True})]
    assert calls == [(-100, os.fsencode(staging), -100, os.fsencode(destination), 0x00000001)]


def test_directory_publication_preserves_rename_errno_message_and_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = SimpleNamespace(renameat2=lambda *_args: -1)
    destination = tmp_path / "run"
    expected_errno = errno.EEXIST

    with monkeypatch.context() as scoped:
        scoped.setattr(publication.os, "name", "posix")
        scoped.setattr(publication.sys, "platform", "linux")
        scoped.setattr(publication.ctypes, "CDLL", lambda *_args, **_kwargs: library)
        scoped.setattr(publication.ctypes, "get_errno", lambda: expected_errno)
        with pytest.raises(OSError, match=rf"^\[Errno {expected_errno}\] ") as error:
            publication.publish_directory(tmp_path / "stage", destination)

    assert error.value.errno == expected_errno
    assert error.value.strerror == os.strerror(expected_errno)
    assert error.value.filename == os.fspath(destination)


@pytest.mark.parametrize("platform", ["linux", "darwin", "freebsd"])
def test_directory_publication_fails_closed_without_exclusive_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(publication.sys, "platform", platform)
        scoped.setattr(publication.ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace())
        with pytest.raises(OSError, match="exclusive directory rename is unavailable") as error:
            publication.publish_directory(tmp_path / "stage", tmp_path / "run")

    assert error.value.errno == errno.ENOTSUP
    assert error.value.strerror == "exclusive directory rename is unavailable"


def test_file_publication_does_not_remove_a_replacement_created_after_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    original_link = publication.os.link

    def replace_first_then_fail(source: Path, target: Path) -> None:
        if target.name == "metrics.json":
            raise OSError(errno.EEXIST, "competing publisher")
        original_link(source, target)
        if target.name == "predictions.parquet":
            target.unlink()
            target.write_bytes(b"replacement")

    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "predictions.parquet").write_bytes(b"staged")
    (staging / "metrics.json").write_bytes(b"metrics")
    monkeypatch.setattr(publication.os, "link", replace_first_then_fail)

    with pytest.raises(OSError, match="competing publisher"):
        publication.publish_files(staging, destination, ["predictions.parquet", "metrics.json"])

    assert (destination / "predictions.parquet").read_bytes() == b"replacement"

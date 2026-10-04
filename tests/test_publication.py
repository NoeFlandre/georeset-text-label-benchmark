"""Tests for staged, exclusive publication of pilot artifacts."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

import pytest

from georeset_text_label_benchmark.pilot import publication


def test_file_publication_leaves_no_commit_marker_when_a_path_already_exists(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    (destination / "metrics.json").write_bytes(b"competing writer")

    with publication.staged_directory(destination, prefix=".stage-") as staging:
        (staging / "predictions.parquet").write_bytes(b"staged predictions")
        (staging / "metrics.json").write_bytes(b"staged metrics")

        with pytest.raises(FileExistsError):
            publication.publish_files(
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
        publication.publish_directory(staging, destination)

    assert list(destination.iterdir()) == []
    assert (staging / "manifest.json").read_bytes() == b"complete staged manifest"


def test_directory_publication_recovers_a_directory_created_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "stage"
    destination = tmp_path / "run"
    staging.mkdir()
    (staging / "metrics.json").write_bytes(b"metrics")
    (staging / "manifest.json").write_bytes(b"manifest")
    original_mkdir = Path.mkdir
    raced = False

    def create_partial_directory(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal raced
        if path == destination and not raced:
            raced = True
            original_mkdir(path, *args, **kwargs)
            os.link(staging / "metrics.json", destination / "metrics.json")
            raise FileExistsError(errno.EEXIST, "racing publisher", str(destination))
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", create_partial_directory)

    publication.publish_directory(staging, destination)

    assert {path.name for path in destination.iterdir()} == {"metrics.json", "manifest.json"}
    assert (destination / "metrics.json").stat().st_ino == (staging / "metrics.json").stat().st_ino
    assert (destination / "manifest.json").stat().st_ino == (
        staging / "manifest.json"
    ).stat().st_ino


def test_publication_preflight_accepts_atomic_local_filesystem_primitives(tmp_path: Path) -> None:
    publication.ensure_publication_supported(tmp_path)


def test_publication_preflight_probes_the_destination_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_temporary_directory = publication.tempfile.TemporaryDirectory
    observed_directories: list[Path | None] = []

    def create_probe(*args: Any, **kwargs: Any) -> Any:
        observed_directories.append(kwargs.get("dir"))
        return original_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(publication.tempfile, "TemporaryDirectory", create_probe)

    publication.ensure_publication_supported(tmp_path)

    assert observed_directories == [tmp_path]


def test_publication_preflight_requires_an_existing_directory(tmp_path: Path) -> None:
    parent = tmp_path / "file"
    parent.touch()

    with pytest.raises(NotADirectoryError) as error:
        publication.ensure_publication_supported(parent)

    assert error.value.args == (str(parent),)


def test_publication_preflight_checks_exclusive_directory_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_mkdir = Path.mkdir
    probe_calls = 0

    def allow_existing_probe_directory(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal probe_calls
        if path.name == "directory":
            probe_calls += 1
            if probe_calls == 2:
                return
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", allow_existing_probe_directory)

    with pytest.raises(
        OSError, match="output filesystem must support exclusive directory creation"
    ) as error:
        publication.ensure_publication_supported(tmp_path)

    assert (error.value.errno, error.value.strerror, error.value.filename) == (
        errno.ENOTSUP,
        "output filesystem must support exclusive directory creation",
        str(tmp_path),
    )


def test_publication_preflight_preserves_link_failure_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported_link(_source: Path, _target: Path) -> None:
        raise OSError(errno.EINVAL, "mount rejected the operation")

    monkeypatch.setattr(publication.os, "link", unsupported_link)

    with pytest.raises(
        OSError, match="output filesystem must support exclusive hard links"
    ) as error:
        publication.ensure_publication_supported(tmp_path)

    assert (error.value.errno, error.value.strerror, error.value.filename) == (
        errno.EINVAL,
        "output filesystem must support exclusive hard links",
        str(tmp_path),
    )


def test_publication_preflight_rejects_nonexclusive_existing_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_link = publication.os.link
    calls = 0

    def allow_existing_link(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_link(source, target)

    monkeypatch.setattr(publication.os, "link", allow_existing_link)

    with pytest.raises(
        OSError,
        match="output filesystem hard links must fail when the destination exists",
    ) as error:
        publication.ensure_publication_supported(tmp_path)

    assert (error.value.errno, error.value.strerror, error.value.filename) == (
        errno.ENOTSUP,
        "output filesystem hard links must fail when the destination exists",
        str(tmp_path),
    )


def test_publication_preflight_wraps_second_link_failure_with_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_link = publication.os.link
    calls = 0

    def fail_second_link(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EINVAL, "mount rejected the operation")
        original_link(source, target)

    monkeypatch.setattr(publication.os, "link", fail_second_link)

    with pytest.raises(
        OSError, match="output filesystem must support exclusive hard links"
    ) as error:
        publication.ensure_publication_supported(tmp_path)

    assert (error.value.errno, error.value.strerror, error.value.filename) == (
        errno.EINVAL,
        "output filesystem must support exclusive hard links",
        str(tmp_path),
    )


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


def test_file_publication_retry_skips_only_its_own_existing_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    staging = tmp_path / "stage"
    staging.mkdir()
    names = ("predictions.parquet", "metrics.json", "manifest.json")
    for name in names:
        (staging / name).write_bytes(name.encode("utf-8"))
    original_link = publication.os.link
    failed_once = False

    def fail_metrics_once(source: Path, target: Path) -> None:
        nonlocal failed_once
        if target.name == "metrics.json" and not failed_once:
            failed_once = True
            raise OSError(errno.EIO, "transient publication failure")
        original_link(source, target)

    monkeypatch.setattr(publication.os, "link", fail_metrics_once)

    with pytest.raises(OSError, match="transient publication failure"):
        publication.publish_files(staging, destination, names)

    assert (destination / "predictions.parquet").stat().st_ino == (
        staging / "predictions.parquet"
    ).stat().st_ino
    assert not (destination / "manifest.json").exists()

    publication.publish_files(staging, destination, names)

    assert all((destination / name).read_bytes() == name.encode("utf-8") for name in names)
    assert all(
        (destination / name).stat().st_ino == (staging / name).stat().st_ino for name in names
    )


def test_file_publication_accepts_a_link_that_succeeded_before_error_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "stage"
    destination = tmp_path / "run"
    staging.mkdir()
    destination.mkdir()
    names = ("metrics.json", "manifest.json")
    for name in names:
        (staging / name).write_bytes(name.encode("utf-8"))
    original_link = publication.os.link

    def link_then_raise(source: Path, target: Path) -> None:
        original_link(source, target)
        raise FileExistsError(errno.EEXIST, "link completed before error return", str(target))

    monkeypatch.setattr(publication.os, "link", link_then_raise)

    publication.publish_files(staging, destination, names)

    assert all(
        (destination / name).stat().st_ino == (staging / name).stat().st_ino for name in names
    )


def test_directory_publication_uses_exclusive_links_without_renameat2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    names = ("dspark_predictions.parquet", "dspark_metrics.json", "dspark_manifest.json")
    for name in names:
        (staging / name).write_bytes(name.encode("utf-8"))
    destination = tmp_path / "run"
    link_calls: list[tuple[Path, Path]] = []
    original_link = publication.os.link

    def record_link(source: Path, target: Path) -> None:
        link_calls.append((source, target))
        original_link(source, target)

    monkeypatch.setattr(publication.os, "link", record_link)
    monkeypatch.setattr(
        publication,
        "_rename_linux_exclusive",
        lambda *_args: pytest.fail("publication must not rely on renameat2 flags"),
        raising=False,
    )

    publication.publish_directory(staging, destination)

    assert {target.name for _, target in link_calls[:-1]} == set(names[:2])
    assert link_calls[-1][1].name == "dspark_manifest.json"
    assert all((destination / name).read_bytes() == name.encode("utf-8") for name in names)
    assert all(
        (destination / name).stat().st_ino == (staging / name).stat().st_ino for name in names
    )


def test_publication_preflight_rejects_filesystem_without_atomic_noclobber_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported_link(_source: Path, _target: Path) -> None:
        raise OSError(errno.EINVAL, "rename/link operation is unsupported on this mount")

    monkeypatch.setattr(publication.os, "link", unsupported_link)

    with pytest.raises(OSError, match="output filesystem must support exclusive hard links"):
        publication.ensure_publication_supported(tmp_path)


def test_staged_directory_discovery_rejects_missing_parent_and_ambiguous_stages(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "not-created"
    assert publication.find_staged_directory(parent, prefix=".run-staging-") is None

    parent.mkdir()
    (parent / ".run-staging-one").mkdir()
    (parent / ".run-staging-two").mkdir()
    with pytest.raises(RuntimeError) as error:
        publication.find_staged_directory(parent, prefix=".run-staging-")

    assert str(error.value) == (
        "multiple staged publications require inspection: "
        f"{parent / '.run-staging-one'}, {parent / '.run-staging-two'}"
    )


@pytest.mark.parametrize("entry_kind", ["empty", "symlink"])
def test_directory_publication_rejects_empty_or_nonregular_stages(
    tmp_path: Path, entry_kind: str
) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    if entry_kind == "symlink":
        target = tmp_path / "source"
        target.write_text("data", encoding="utf-8")
        (staging / "manifest.json").symlink_to(target)

    with pytest.raises(
        ValueError, match="staged publication must contain regular files only"
    ) as error:
        publication.publish_directory(staging, tmp_path / "run")

    assert str(error.value) == "staged publication must contain regular files only"


@pytest.mark.parametrize("destination_kind", ["file", "symlink"])
def test_directory_publication_rejects_file_or_symlink_destination(
    tmp_path: Path, destination_kind: str
) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "manifest.json").write_text("complete", encoding="utf-8")
    destination = tmp_path / "run"
    if destination_kind == "file":
        destination.write_text("competing writer", encoding="utf-8")
    else:
        target = tmp_path / "target"
        target.mkdir()
        destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(FileExistsError) as error:
        publication.publish_directory(staging, destination)

    assert (error.value.errno, error.value.strerror, error.value.filename) == (
        errno.EEXIST,
        "output destination already exists",
        str(destination),
    )


def test_publication_preflight_rejects_hard_links_that_do_not_preserve_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def copy_instead_of_link(source: Path, target: Path) -> None:
        target.write_bytes(source.read_bytes())

    monkeypatch.setattr(publication.os, "link", copy_instead_of_link)

    with pytest.raises(OSError, match="hard links do not preserve file identity") as error:
        publication.ensure_publication_supported(tmp_path)

    assert (error.value.errno, error.value.strerror, error.value.filename) == (
        errno.ENOTSUP,
        "output filesystem hard links do not preserve file identity",
        str(tmp_path),
    )


def test_directory_publication_resumes_only_a_matching_partial_directory(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    destination = tmp_path / "run"
    destination.mkdir()
    (staging / "metrics.json").write_bytes(b"metrics")
    (staging / "manifest.json").write_bytes(b"manifest")
    (staging / "predictions.parquet").write_bytes(b"predictions")
    os.link(staging / "metrics.json", destination / "metrics.json")

    publication.publish_directory(staging, destination)

    assert {path.name for path in destination.iterdir()} == {
        "metrics.json",
        "manifest.json",
        "predictions.parquet",
    }
    assert all(
        (destination / name).stat().st_ino == (staging / name).stat().st_ino
        for name in ("metrics.json", "manifest.json", "predictions.parquet")
    )


def test_staged_file_names_order_the_e5_manifest_last(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    for name in ("predictions.parquet", "metrics.json", "manifest.json"):
        (staging / name).touch()

    assert publication._staged_file_names(staging) == [
        "metrics.json",
        "predictions.parquet",
        "manifest.json",
    ]


def test_staged_directory_ownership_rejects_an_unexpected_hard_link(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    destination = tmp_path / "run"
    staging.mkdir()
    destination.mkdir()
    (staging / "manifest.json").write_bytes(b"manifest")
    (staging / "extra.bin").write_bytes(b"extra")
    os.link(staging / "manifest.json", destination / "manifest.json")
    os.link(staging / "extra.bin", destination / "extra.bin")

    assert not publication._owns_staged_entries(staging, destination, ["manifest.json"])


def test_staged_directory_ownership_rejects_an_empty_destination(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    destination = tmp_path / "run"
    staging.mkdir()
    destination.mkdir()
    (staging / "manifest.json").write_bytes(b"manifest")

    assert not publication._owns_staged_entries(staging, destination, ["manifest.json"])


def test_partial_directory_rejection_preserves_errno_message_and_path(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    destination = tmp_path / "run"
    staging.mkdir()
    destination.mkdir()
    (staging / "manifest.json").write_bytes(b"staged")
    (destination / "manifest.json").write_bytes(b"competitor")

    with pytest.raises(FileExistsError) as error:
        publication._require_owned_partial_directory(staging, destination, ["manifest.json"])

    assert (error.value.errno, error.value.strerror, error.value.filename) == (
        errno.EEXIST,
        "output destination already exists",
        str(destination),
    )


def test_same_file_rejects_a_symlink_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"shared")
    destination.symlink_to(source)

    assert not publication._same_file(source, destination)


def test_same_file_rejects_a_symlink_source(tmp_path: Path) -> None:
    target = tmp_path / "target"
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    target.write_bytes(b"shared")
    source.symlink_to(target)
    os.link(target, destination)

    assert not publication._same_file(source, destination)


def test_same_file_rejects_a_hard_link_to_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    target.write_bytes(b"shared")
    source.symlink_to(target)
    try:
        os.link(source, destination, follow_symlinks=False)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"filesystem cannot hard-link a symlink: {error}")

    assert not publication._same_file(source, destination)

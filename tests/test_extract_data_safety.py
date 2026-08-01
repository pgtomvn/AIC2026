"""Safety and selection tests for the data archive extractor."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import scripts.extract_data as extract_data_module
from scripts.extract_data import (
    ArchiveSpec,
    UnsafeArchiveError,
    build_parser,
    extract_archive,
    inspect_archive,
    main,
)


def _write_archive(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)


def test_cli_defaults_to_read_only_inspection() -> None:
    args = build_parser().parse_args([])

    assert args.inspect is False
    assert args.verify is False
    assert args.sample is None
    assert args.full is False
    assert args.force is False


def test_inspection_validates_and_inventories_without_output(tmp_path: Path) -> None:
    archive_path = tmp_path / "keyframes.zip"
    _write_archive(
        archive_path,
        {
            "keyframes/L21_V001/0001.jpg": b"first",
            "keyframes/L21_V002/0001.jpg": b"second",
        },
    )

    report = inspect_archive(
        archive_path,
        archive_root="keyframes",
        sample="L21_V001",
    )

    assert report.file_count == 2
    assert report.selected_file_count == 1
    assert report.uncompressed_bytes == 11
    assert report.selected_uncompressed_bytes == 5
    assert sorted(path.name for path in tmp_path.iterdir()) == ["keyframes.zip"]


def test_sample_extraction_normalizes_root_and_is_idempotent(tmp_path: Path) -> None:
    archive_path = tmp_path / "keyframes.zip"
    output_dir = tmp_path / "output"
    _write_archive(
        archive_path,
        {
            "keyframes/L21_V001/0001.jpg": b"selected",
            "keyframes/L21_V002/0001.jpg": b"not selected",
        },
    )

    first = extract_archive(
        archive_path,
        output_dir,
        archive_root="keyframes",
        sample="L21_V001",
    )
    second = extract_archive(
        archive_path,
        output_dir,
        archive_root="keyframes",
        sample="L21_V001",
    )

    extracted_path = output_dir / "L21_V001" / "0001.jpg"
    assert extracted_path.read_bytes() == b"selected"
    assert not (output_dir / "keyframes").exists()
    assert not (output_dir / "L21_V002").exists()
    assert first.extracted_file_count == 1
    assert first.skipped_file_count == 0
    assert second.extracted_file_count == 0
    assert second.skipped_file_count == 1


def test_video_sample_matches_filename_stem(tmp_path: Path) -> None:
    archive_path = tmp_path / "videos.zip"
    output_dir = tmp_path / "output"
    _write_archive(
        archive_path,
        {
            "video/L21_V001.mp4": b"selected",
            "video/L21_V002.mp4": b"not selected",
        },
    )

    report = extract_archive(
        archive_path,
        output_dir,
        archive_root="video",
        sample="L21_V001",
    )

    assert report.selected_file_count == 1
    assert (output_dir / "L21_V001.mp4").read_bytes() == b"selected"
    assert not (output_dir / "L21_V002.mp4").exists()


def test_existing_file_is_preserved_unless_force_is_explicit(tmp_path: Path) -> None:
    archive_path = tmp_path / "keyframes.zip"
    output_dir = tmp_path / "output"
    _write_archive(
        archive_path,
        {"keyframes/L21_V001/0001.jpg": b"archive value"},
    )
    target = output_dir / "L21_V001" / "0001.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"local value")

    skipped = extract_archive(
        archive_path,
        output_dir,
        archive_root="keyframes",
        sample="L21_V001",
    )
    assert target.read_bytes() == b"local value"
    assert skipped.skipped_file_count == 1

    overwritten = extract_archive(
        archive_path,
        output_dir,
        archive_root="keyframes",
        sample="L21_V001",
        force=True,
    )
    assert target.read_bytes() == b"archive value"
    assert overwritten.extracted_file_count == 1


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.txt",
        "/absolute.txt",
        "C:/absolute.txt",
        "keyframes/../escape.txt",
        "keyframes\\..\\escape.txt",
    ],
)
def test_unsafe_paths_are_rejected_before_output_is_created(
    tmp_path: Path,
    member_name: str,
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    output_dir = tmp_path / "output"
    _write_archive(archive_path, {member_name: b"unsafe"})

    with pytest.raises(UnsafeArchiveError):
        extract_archive(
            archive_path,
            output_dir,
            archive_root="keyframes",
        )

    assert not output_dir.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_unselected_unsafe_member_aborts_before_any_write(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    output_dir = tmp_path / "output"
    _write_archive(
        archive_path,
        {
            "keyframes/L21_V001/0001.jpg": b"safe",
            "keyframes/../../escape.txt": b"unsafe",
        },
    )

    with pytest.raises(UnsafeArchiveError):
        extract_archive(
            archive_path,
            output_dir,
            archive_root="keyframes",
            sample="L21_V001",
        )

    assert not output_dir.exists()


def test_missing_sample_does_not_create_destination(tmp_path: Path) -> None:
    archive_path = tmp_path / "keyframes.zip"
    output_dir = tmp_path / "output"
    _write_archive(
        archive_path,
        {"keyframes/L21_V002/0001.jpg": b"other"},
    )

    report = extract_archive(
        archive_path,
        output_dir,
        archive_root="keyframes",
        sample="L21_V001",
    )

    assert report.selected_file_count == 0
    assert report.extracted_file_count == 0
    assert not output_dir.exists()


def test_copy_failure_never_replaces_a_complete_target(
    tmp_path: Path, monkeypatch
) -> None:
    archive_path = tmp_path / "keyframes.zip"
    output_dir = tmp_path / "output"
    _write_archive(
        archive_path,
        {"keyframes/L21_V001/0001.jpg": b"replacement"},
    )
    target = output_dir / "L21_V001" / "0001.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"preserve me")

    def fail_after_partial_copy(source, destination) -> None:
        destination.write(source.read(3))
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(
        extract_data_module.shutil,
        "copyfileobj",
        fail_after_partial_copy,
    )

    with pytest.raises(OSError, match="simulated interrupted copy"):
        extract_archive(
            archive_path,
            output_dir,
            archive_root="keyframes",
            sample="L21_V001",
            force=True,
        )

    assert target.read_bytes() == b"preserve me"
    assert list(target.parent.glob(".*.tmp")) == []


def test_cli_sample_preflights_and_extracts_both_archives(
    tmp_path: Path, monkeypatch
) -> None:
    keyframes_archive = tmp_path / "keyframes.zip"
    videos_archive = tmp_path / "videos.zip"
    keyframes_output = tmp_path / "data" / "keyframes"
    videos_output = tmp_path / "data" / "videos"
    _write_archive(
        keyframes_archive,
        {"keyframes/L21_V001/0001.jpg": b"frame"},
    )
    _write_archive(
        videos_archive,
        {"video/L21_V001.mp4": b"video"},
    )
    specs = (
        ArchiveSpec(keyframes_archive, keyframes_output, "keyframes"),
        ArchiveSpec(videos_archive, videos_output, "video"),
    )
    monkeypatch.setattr(extract_data_module, "load_config", lambda: {})
    monkeypatch.setattr(extract_data_module, "setup_logger_from_config", lambda config: None)
    monkeypatch.setattr(extract_data_module, "_archive_specs", lambda config: specs)

    assert main(["--sample", "L21_V001"]) == 0
    assert (keyframes_output / "L21_V001" / "0001.jpg").read_bytes() == b"frame"
    assert (videos_output / "L21_V001.mp4").read_bytes() == b"video"


def test_cli_preflights_second_destination_before_first_write(
    tmp_path: Path, monkeypatch
) -> None:
    keyframes_archive = tmp_path / "keyframes.zip"
    videos_archive = tmp_path / "videos.zip"
    keyframes_output = tmp_path / "data" / "keyframes"
    videos_output = tmp_path / "data" / "videos"
    _write_archive(
        keyframes_archive,
        {"keyframes/L21_V001/0001.jpg": b"frame"},
    )
    _write_archive(
        videos_archive,
        {"video/L21_V001.mp4": b"video"},
    )
    videos_output.parent.mkdir(parents=True)
    videos_output.write_bytes(b"destination conflict")
    specs = (
        ArchiveSpec(keyframes_archive, keyframes_output, "keyframes"),
        ArchiveSpec(videos_archive, videos_output, "video"),
    )
    monkeypatch.setattr(extract_data_module, "load_config", lambda: {})
    monkeypatch.setattr(extract_data_module, "setup_logger_from_config", lambda config: None)
    monkeypatch.setattr(extract_data_module, "_archive_specs", lambda config: specs)

    assert main(["--sample", "L21_V001"]) == 3
    assert not keyframes_output.exists()
    assert videos_output.read_bytes() == b"destination conflict"


def test_foreign_archive_root_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "foreign.zip"
    output_dir = tmp_path / "output"
    _write_archive(archive_path, {"unexpected/file.jpg": b"unsafe"})

    with pytest.raises(UnsafeArchiveError, match="expected 'keyframes' root"):
        extract_archive(archive_path, output_dir, archive_root="keyframes")

    assert not output_dir.exists()


@pytest.mark.parametrize("configured_path", ["../outside", "C:/outside"])
def test_configured_output_must_stay_below_project_data(
    tmp_path: Path, monkeypatch, configured_path: str
) -> None:
    monkeypatch.setattr(extract_data_module, "PROJECT_ROOT", tmp_path)
    config = {
        "data": {
            "raw_keyframes": configured_path,
            "raw_videos": "data/raw/videos",
        }
    }

    with pytest.raises(ValueError, match="project-relative"):
        extract_data_module._archive_specs(config)

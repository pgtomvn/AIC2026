"""Offline tests for the shared Batch-2 ingestion command line."""

from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

import pytest

from src.ingest import cli, native_pipeline


def _config(profile: str = "laptop") -> dict:
    return {
        "profile": profile,
        "ingestion": {
            "worker_profiles": {
                "laptop": {"cpu_workers": 2, "queue_size": 4},
                "accelerated": {"cpu_workers": 8, "queue_size": 8},
            }
        },
    }


def test_dry_run_supports_input_dir_all_channels_and_profile_defaults(
    tmp_path, monkeypatch, capsys
) -> None:
    (tmp_path / "B2_V002.mp4").write_bytes(b"")
    (tmp_path / "B2_V001.mp4").write_bytes(b"")
    monkeypatch.setattr(cli, "load_config", lambda: _config())

    exit_code = cli.main(
        [
            "--batch-id",
            "AIC26-B2",
            "--input-dir",
            str(tmp_path),
            "--all-channels",
            "--max-videos",
            "1",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["cpu_workers"] == 2
    assert payload["queue_size"] == 4
    assert payload["channels"] == "all"
    assert payload["videos"] == ["B2_V001"]


def test_parallel_workers_is_cpu_alias_and_accelerated_defaults_are_bounded(
    tmp_path, monkeypatch, capsys
) -> None:
    (tmp_path / "B2_V001.mp4").write_bytes(b"")
    monkeypatch.setattr(cli, "load_config", lambda: _config("accelerated"))

    assert cli.main(
        [
            str(tmp_path),
            "--batch-id",
            "AIC26-B2",
            "--parallel-workers",
            "3",
            "--dry-run",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cpu_workers"] == 3
    assert payload["queue_size"] == 8


def test_dry_run_reports_index_queues_and_scoped_reprocess(
    tmp_path, monkeypatch, capsys
) -> None:
    (tmp_path / "B2_V001.mp4").write_bytes(b"")
    config = _config()
    config["ingestion"].update({"visual_queue_size": 7, "index_batch_size": 19})
    monkeypatch.setattr(cli, "load_config", lambda: config)

    assert cli.main(
        [
            str(tmp_path),
            "--batch-id",
            "AIC26-B2",
            "--organizer-compatible-visual",
            "--reprocess-stage",
            "visual",
            "--reprocess-video",
            "B2_V001",
            "--dry-run",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["visual_queue_size"] == 7
    assert payload["index_batch_size"] == 19
    assert payload["organizer_compatible_visual"] is True
    assert payload["reprocess_stages"] == ["visual"]


def test_reprocess_scope_requires_stage_and_video_together(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "B2_V001.mp4").write_bytes(b"")
    monkeypatch.setattr(cli, "load_config", lambda: _config())

    with pytest.raises(SystemExit):
        cli.main(
            [
                str(tmp_path),
                "--batch-id",
                "AIC26-B2",
                "--reprocess-stage",
                "visual",
                "--dry-run",
            ]
        )


def test_requested_enrichment_failure_returns_nonzero(
    tmp_path, monkeypatch, capsys
) -> None:
    source = tmp_path / "B2_V001.mp4"
    source.write_bytes(b"")

    class _Ledger:
        @staticmethod
        def status_summary(_batch_id):
            return SimpleNamespace(as_dict=lambda: {"stage_counts": {"ocr": {"failed": 1}}})

    class _Ingestor:
        ledger = _Ledger()

        @staticmethod
        def initialize(**_kwargs):
            return None

        @staticmethod
        def ingest_many(_sources, **_kwargs):
            return {"B2_V001": "active; enrichment failed: ocr"}

    monkeypatch.setattr(cli, "load_config", lambda: _config())
    monkeypatch.setattr(cli, "build_native_ingestor", lambda *_args, **_kwargs: _Ingestor())

    assert cli.main([str(source), "--batch-id", "AIC26-B2"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["core_failures"] == []
    assert payload["enrichment_failures"] == ["B2_V001"]


def test_native_pipeline_module_entrypoint_delegates_to_shared_cli(monkeypatch) -> None:
    calls: list[list[str] | None] = []
    monkeypatch.setattr(cli, "main", lambda argv=None: calls.append(argv) or 17)
    arguments = ["--batch-id", "AIC26-B2", "--input-dir", "batch2", "--dry-run"]

    assert native_pipeline._compatibility_main(arguments) == 17
    assert calls == [arguments]


def test_zip_ingestion_preserves_catalog_gate_before_pipeline_run(
    tmp_path, monkeypatch, capsys
) -> None:
    archive = tmp_path / "batch2.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("videos/B2_V001.mp4", b"fixture")

    events: list[tuple[str, object]] = []

    class _Ledger:
        @staticmethod
        def initialize():
            events.append(("ledger_initialize", None))

        @staticmethod
        def assert_archive_ready(path):
            events.append(("catalog", path))

        @staticmethod
        def status_summary(_batch_id):
            return SimpleNamespace(as_dict=lambda: {"stage_counts": {}})

    class _Ingestor:
        ledger = _Ledger()

        @staticmethod
        def initialize(**kwargs):
            events.append(("ingestor_initialize", kwargs))

        @staticmethod
        def ingest_many(sources, **kwargs):
            events.append(("ingest", (tuple(sources), kwargs)))
            return {"B2_V001": "active"}

    build_calls: list[dict] = []
    monkeypatch.setattr(cli, "load_config", lambda: _config())
    monkeypatch.setattr(
        cli,
        "build_native_ingestor",
        lambda *_args, **kwargs: build_calls.append(kwargs) or _Ingestor(),
    )

    assert cli.main([str(archive), "--batch-id", "AIC26-B2", "--all-channels"]) == 0
    capsys.readouterr()

    assert build_calls[0]["enrichment"] is True
    assert events == [
        ("ledger_initialize", None),
        ("catalog", archive.resolve()),
        (
            "ingestor_initialize",
            {"recreate_indexes": False, "recover_interrupted": True},
        ),
        (
            "ingest",
            (
                tuple(cli.discover_video_sources((archive,))),
                {"cpu_workers": 2, "queue_size": 4, "enrich": True},
            ),
        ),
    ]


def test_zip_catalog_failure_prevents_recreate_index_mutation(
    tmp_path, monkeypatch
) -> None:
    archive = tmp_path / "unmanaged-batch2.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("videos/B2_V001.mp4", b"fixture")

    events: list[tuple[str, object]] = []

    class _Ledger:
        @staticmethod
        def initialize():
            events.append(("ledger_initialize", None))

        @staticmethod
        def assert_archive_ready(path):
            events.append(("catalog", path))
            raise RuntimeError("archive is not verified")

    class _Ingestor:
        ledger = _Ledger()

        @staticmethod
        def initialize(**kwargs):
            events.append(("index_mutation", kwargs))

        @staticmethod
        def ingest_many(_sources, **_kwargs):
            events.append(("ingest", None))
            return {}

    monkeypatch.setattr(cli, "load_config", lambda: _config())
    monkeypatch.setattr(cli, "build_native_ingestor", lambda *_args, **_kwargs: _Ingestor())

    with pytest.raises(RuntimeError, match="not verified"):
        cli.main(
            [
                str(archive),
                "--batch-id",
                "AIC26-B2",
                "--all-channels",
                "--recreate-indexes",
            ]
        )

    assert events == [
        ("ledger_initialize", None),
        ("catalog", archive.resolve()),
    ]

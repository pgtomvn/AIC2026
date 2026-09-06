from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api import create_app
from src.contracts import QACompetitionSpec
from src.export.preflight import BundleEntry
from src.export.staging_store import (
    StagedEntry,
    StagingRevisionConflict,
    StagingStore,
    StagingStoreError,
)
from src.official_submission import (
    OfficialSubmissionValidationError,
    parse_submission_bytes,
)
from src.qa_submission import serialize_qa_submission


def _fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundle_entry(
    task: str,
    query_id: str,
    data: bytes,
    *,
    stage_count: int | None = None,
) -> BundleEntry:
    return BundleEntry(
        task=task,  # type: ignore[arg-type]
        query_id=query_id,
        csv_bytes=data,
        batch_id="batch",
        source_fingerprint=_fingerprint(data),
        stage_count=stage_count,
        source_ranks=(1,) if task == "kis" else None,
    )


def _clock() -> datetime:
    return datetime(2026, 8, 20, 9, 30, tzinfo=timezone.utc)


def test_qa_serializer_quotes_only_when_rfc4180_requires_it() -> None:
    data = serialize_qa_submission(
        (("L01_V028", 3450, '  “Năm người nói "xin chào" rồi đi”  '),),
        spec=QACompetitionSpec.conservative_general_text(),
        batch_id="batch",
        is_batch_active=lambda batch: batch == "batch",
        is_canonical_source_frame=lambda batch, video, frame: (
            batch == "batch" and (video, frame) == ("L01_V028", 3450)
        ),
    )

    assert data == 'L01_V028,3450,"năm người nói ""xin chào"" rồi đi"\n'.encode()
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data

    collapsed = serialize_qa_submission(
        (("L01_V028", 3450, '  "Năm\n người"  '),),
        spec=QACompetitionSpec.conservative_general_text(),
        batch_id="batch",
        is_batch_active=lambda _batch: True,
        is_canonical_source_frame=lambda *_identity: True,
    )
    assert collapsed == "L01_V028,3450,năm người\n".encode()


def test_official_parser_defaults_to_optional_quotes_with_strict_legacy_opt_in() -> None:
    unquoted = b"L01_V028,3450,5\n"
    parsed = parse_submission_bytes(
        unquoted,
        filename="query-p1-15-qa.csv",
        task="qa",
        spec_id="codabench_10187_2025",
    )
    assert parsed.rows == (("L01_V028", "3450", "5"),)

    with pytest.raises(OfficialSubmissionValidationError) as caught:
        parse_submission_bytes(
            unquoted,
            filename="query-p1-15-qa.csv",
            task="qa",
            spec_id="codabench_10187_2025",
            require_qa_answer_quotes=True,
        )
    assert caught.value.code == "submission_qa_answer_quotes_required"

    escaped = 'L01_V028,3450,"anh ấy nói ""xin chào"""\n'.encode()
    strict = parse_submission_bytes(
        escaped,
        filename="query-p1-15-qa.csv",
        task="qa",
        spec_id="codabench_10187_2025",
        require_qa_answer_quotes=True,
    )
    assert strict.rows[0][2] == 'anh ấy nói "xin chào"'

    with pytest.raises(OfficialSubmissionValidationError) as malformed:
        parse_submission_bytes(
            b'L01_V028,3450,"bad"quote"\n',
            filename="query-p1-15-qa.csv",
            task="qa",
            spec_id="codabench_10187_2025",
            require_qa_answer_quotes=True,
        )
    assert malformed.value.code in {
        "submission_csv_invalid",
        "submission_qa_answer_quotes_required",
    }


@pytest.mark.parametrize("length", (100, 101))
def test_strict_quoted_qa_enforces_unicode_codepoint_limit(length: int) -> None:
    data = f'L01_V028,3450,"{"đ" * length}"\n'.encode()
    if length == 100:
        parse_submission_bytes(
            data,
            filename="query-p1-15-qa.csv",
            task="qa",
            spec_id="codabench_10187_2025",
            require_qa_answer_quotes=True,
        )
        return
    with pytest.raises(OfficialSubmissionValidationError) as caught:
        parse_submission_bytes(
            data,
            filename="query-p1-15-qa.csv",
            task="qa",
            spec_id="codabench_10187_2025",
            require_qa_answer_quotes=True,
        )
    assert caught.value.code == "submission_answer_too_long"


def test_staging_store_survives_new_instance_and_preserves_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "outputs" / "submissions" / "active_staging.json"
    values = (
        _bundle_entry("kis", "p1-1", b"L21_V001,10\n"),
        _bundle_entry("qa", "p1-15", 'L21_V001,20,"năm người"\n'.encode()),
        _bundle_entry("trake", "p1-4", b"L21_V001,10,20\n", stage_count=2),
    )
    store = StagingStore(path, clock=_clock)
    revision = 0
    for entry in values:
        snapshot = store.upsert(
            StagedEntry.create(entry, row_count=1, clock=_clock),
            expected_revision=revision,
        )
        revision = snapshot.revision

    restored = StagingStore(path, clock=_clock).load()
    assert restored.revision == 3
    assert {item.filename: item.bundle_entry.csv_bytes for item in restored.entries} == {
        entry.filename: entry.csv_bytes for entry in values
    }
    assert all(
        item.sha256 == _fingerprint(item.bundle_entry.csv_bytes) for item in restored.entries
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["entries"]["query-p1-15-qa.csv"]["csv_content"] == ('L21_V001,20,"năm người"\n')

    empty = StagingStore(path, clock=_clock).clear(expected_revision=3)
    assert empty.revision == 4
    assert empty.entries == ()
    assert StagingStore(path).load().entries == ()
    assert not list(path.parent.glob("*.tmp"))


def test_staging_store_revision_conflict_and_corruption_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "active_staging.json"
    store = StagingStore(path, clock=_clock)
    entry = _bundle_entry("kis", "p1-1", b"L21_V001,10\n")
    store.upsert(
        StagedEntry.create(entry, row_count=1, clock=_clock),
        expected_revision=0,
    )
    with pytest.raises(StagingRevisionConflict):
        store.clear(expected_revision=0)

    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(StagingStoreError) as caught:
        StagingStore(path).load()
    assert caught.value.code == "staging_store_invalid"
    assert path.read_text(encoding="utf-8") == "{broken"


class _Ledger:
    @staticmethod
    def is_batch_active(batch_id: str) -> bool:
        return batch_id == "batch"


class _Repository:
    timestamps = {("L21_V001", 10): 100, ("L21_V001", 20): 200}

    def get_source_frame(self, batch_id: str, video_id: str, frame_idx: int):
        timestamp = self.timestamps.get((video_id, frame_idx)) if batch_id == "batch" else None
        return None if timestamp is None else SimpleNamespace(timestamp_ms=timestamp)

    get_frame = get_source_frame


def _context(tmp_path: Path):
    return SimpleNamespace(
        config={"competition": {"output_dir": str(tmp_path)}},
        ledger=_Ledger(),
        artifact_repository=_Repository(),
        pipeline=SimpleNamespace(),
    )


def _endpoint(app, path: str, method: str):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set())
    )


def test_staging_api_round_trip_revision_and_bundle_from_filenames(tmp_path: Path) -> None:
    app = create_app(_context(tmp_path))
    get_staging = _endpoint(app, "/v1/submissions/staging", "GET")
    upsert = _endpoint(app, "/v1/submissions/staging/entries", "POST")
    create_bundle = _endpoint(app, "/v1/submissions/bundles", "POST")
    initial = get_staging()
    assert initial["revision"] == 0
    data = "L21_V001,20,năm người\n".encode()
    snapshot = upsert(
        {
            "expected_revision": 0,
            "batch_id": "batch",
            "entry": {
                "task": "qa",
                "query_id": "p1-15",
                "csv_base64": base64.b64encode(data).decode("ascii"),
                "source_fingerprint": _fingerprint(data),
                "stage_count": None,
                "source_ranks": None,
            },
        }
    )
    assert snapshot["revision"] == 1
    assert base64.b64decode(snapshot["entries"]["query-p1-15-qa.csv"]["csv_base64"]) == data

    metadata = create_bundle(
        {
            "mode": "current",
            "batch_id": "batch",
            "entry_filenames": ["query-p1-15-qa.csv"],
            "staging_revision": 1,
        }
    )
    assert metadata["file_count"] == 1
    download = _endpoint(
        app,
        "/v1/submissions/bundles/{run_id}/download",
        "GET",
    )(metadata["run_id"])
    with zipfile.ZipFile(download.path) as archive:
        assert archive.namelist() == [
            "submission/",
            "submission/query-p1-15-qa.csv",
        ]
        assert archive.read("submission/query-p1-15-qa.csv") == data

    with pytest.raises(HTTPException) as caught:
        upsert(
            {
                "expected_revision": 0,
                "batch_id": "batch",
                "entry": {
                    "task": "qa",
                    "query_id": "p1-15",
                    "csv_base64": base64.b64encode(data).decode("ascii"),
                    "source_fingerprint": _fingerprint(data),
                },
            }
        )
    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == "staging_revision_conflict"


def test_staging_api_accepts_minimal_qa_and_supports_delete_and_clear(
    tmp_path: Path,
) -> None:
    app = create_app(_context(tmp_path))
    upsert = _endpoint(app, "/v1/submissions/staging/entries", "POST")
    delete = _endpoint(app, "/v1/submissions/staging/entries/{filename}", "DELETE")
    clear = _endpoint(app, "/v1/submissions/staging", "DELETE")
    unquoted = b"L21_V001,20,yes\n"
    qa_snapshot = upsert(
        {
            "expected_revision": 0,
            "batch_id": "batch",
            "entry": {
                "task": "qa",
                "query_id": "p1-15",
                "csv_base64": base64.b64encode(unquoted).decode("ascii"),
                "source_fingerprint": _fingerprint(unquoted),
            },
        }
    )
    assert base64.b64decode(
        qa_snapshot["entries"]["query-p1-15-qa.csv"]["csv_base64"]
    ) == unquoted
    after_qa_delete = delete(
        "query-p1-15-qa.csv", expected_revision=qa_snapshot["revision"]
    )

    kis = b"L21_V001,10\n"
    snapshot = upsert(
        {
            "expected_revision": after_qa_delete["revision"],
            "batch_id": "batch",
            "entry": {
                "task": "kis",
                "query_id": "p1-1",
                "csv_base64": base64.b64encode(kis).decode("ascii"),
                "source_fingerprint": _fingerprint(kis),
                "source_ranks": [1],
            },
        }
    )
    deleted = delete("query-p1-1-kis.csv", expected_revision=snapshot["revision"])
    assert deleted["entries"] == {}
    cleared = clear(expected_revision=deleted["revision"])
    assert cleared["entries"] == {}
    assert cleared["revision"] == deleted["revision"] + 1


def test_staging_row_api_appends_reorders_and_deletes_alternate_qa_answers(
    tmp_path: Path,
) -> None:
    app = create_app(_context(tmp_path))
    upsert = _endpoint(app, "/v1/submissions/staging/entries", "POST")
    mutate = _endpoint(
        app,
        "/v1/submissions/staging/entries/{filename}/rows",
        "POST",
    )
    create_bundle = _endpoint(app, "/v1/submissions/bundles", "POST")
    first = 'L21_V001,20,"xã cam bắc"\n'.encode()
    snapshot = upsert(
        {
            "expected_revision": 0,
            "batch_id": "batch",
            "entry": {
                "task": "qa",
                "query_id": "p1-15",
                "csv_base64": base64.b64encode(first).decode("ascii"),
                "source_fingerprint": _fingerprint(first),
            },
        }
    )
    second = 'L21_V001,20,"cam bắc"\n'.encode()
    snapshot = mutate(
        "query-p1-15-qa.csv",
        {
            "action": "append",
            "expected_revision": snapshot["revision"],
            "row_csv_base64": base64.b64encode(second).decode("ascii"),
        },
    )
    assert base64.b64decode(
        snapshot["entries"]["query-p1-15-qa.csv"]["csv_base64"]
    ) == first + second

    third = 'L21_V001,20,"cam bắc xã"\n'.encode()
    snapshot = mutate(
        "query-p1-15-qa.csv",
        {
            "action": "append",
            "expected_revision": snapshot["revision"],
            "row_csv_base64": base64.b64encode(third).decode("ascii"),
        },
    )
    assert base64.b64decode(
        snapshot["entries"]["query-p1-15-qa.csv"]["csv_base64"]
    ) == first + second + third
    snapshot = mutate(
        "query-p1-15-qa.csv",
        {
            "action": "delete",
            "expected_revision": snapshot["revision"],
            "rank": 2,
        },
    )
    expected = first + third
    assert snapshot["entries"]["query-p1-15-qa.csv"]["row_count"] == 2
    assert base64.b64decode(
        snapshot["entries"]["query-p1-15-qa.csv"]["csv_base64"]
    ) == expected

    metadata = create_bundle(
        {
            "mode": "current",
            "batch_id": "batch",
            "entry_filenames": ["query-p1-15-qa.csv"],
            "staging_revision": snapshot["revision"],
        }
    )
    download = _endpoint(
        app,
        "/v1/submissions/bundles/{run_id}/download",
        "GET",
    )(metadata["run_id"])
    with zipfile.ZipFile(download.path) as archive:
        assert archive.read("submission/query-p1-15-qa.csv") == expected

    with pytest.raises(HTTPException) as duplicate:
        mutate(
            "query-p1-15-qa.csv",
            {
                "action": "append",
                "expected_revision": snapshot["revision"],
                "row_csv_base64": base64.b64encode(third).decode("ascii"),
            },
        )
    assert duplicate.value.status_code == 422
    assert duplicate.value.detail["issues"][0]["code"] == "submission_duplicate_answer"

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.organizer_baseline import (
    BaselineHit,
    OrganizerBaselineError,
    compare_rankings,
    exact_archive_search,
    inspect_reference_archives,
    load_query_file,
)


def _archives(tmp_path: Path, *, invalid: str | None = None) -> tuple[Path, Path, Path]:
    map_path = tmp_path / "maps.zip"
    feature_path = tmp_path / "features.zip"
    object_path = tmp_path / "objects.zip"
    rows = {
        "V001": [(1, 0.0, 25.0, 0), (2, 1.0, 25.0, 25)],
        "V002": [(1, 0.0, 30.0, 0)],
    }
    with zipfile.ZipFile(map_path, "w") as archive:
        for video, values in rows.items():
            stream = io.StringIO(newline="")
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["n", "pts_time", "fps", "frame_idx"])
            writer.writerows(values)
            archive.writestr(f"map-keyframes/{video}.csv", stream.getvalue())
    features = {
        "V001": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
        "V002": np.asarray([[2**-0.5, 2**-0.5]], dtype=np.float16),
    }
    if invalid == "nan":
        features["V001"][0, 0] = np.nan
    if invalid == "rows":
        features["V002"] = np.vstack([features["V002"], features["V002"]])
    with zipfile.ZipFile(feature_path, "w") as archive:
        for video, array in features.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            archive.writestr(f"clip-features/{video}.npy", buffer.getvalue())
    with zipfile.ZipFile(object_path, "w") as archive:
        count = 2 if invalid == "objects" else 3
        for index in range(count):
            archive.writestr(f"objects/V001/{index + 1:03d}.json", "{}")
    return map_path, feature_path, object_path


def test_reference_archive_inspection_and_exact_search(tmp_path: Path) -> None:
    maps, features, objects = _archives(tmp_path)
    stats = inspect_reference_archives(
        maps,
        features,
        object_archive=objects,
        expected_dimension=2,
    )
    assert stats.videos == 2
    assert stats.map_rows == stats.feature_rows == stats.object_json_files == 3
    assert stats.feature_dtype == "float16"

    hits = exact_archive_search(np.asarray([1.0, 0.0]), maps, features, top_k=3)
    assert [(hit.video_id, hit.source_frame_idx) for hit in hits] == [
        ("V001", 0),
        ("V002", 0),
        ("V001", 25),
    ]


@pytest.mark.parametrize("invalid", ["nan", "rows", "objects"])
def test_reference_archive_inspection_rejects_corruption(
    tmp_path: Path,
    invalid: str,
) -> None:
    maps, features, objects = _archives(tmp_path, invalid=invalid)
    with pytest.raises(OrganizerBaselineError):
        inspect_reference_archives(
            maps,
            features,
            object_archive=objects,
            expected_dimension=2,
        )


def test_ranking_comparison_reports_identity_and_score_parity() -> None:
    baseline = [
        BaselineHit("V001", 10, 1, 0.9),
        BaselineHit("V002", 20, 1, 0.8),
    ]
    candidate = [
        {"score": 0.9, "payload": {"video_id": "V001", "source_frame_idx": 10}},
        {"score": 0.8, "payload": {"video_id": "V002", "source_frame_idx": 20}},
    ]
    result = compare_rankings(baseline, candidate)
    assert result["top_1_agreement"] is True
    assert result["identity_recall"] == 1.0
    assert result["max_common_score_delta"] == pytest.approx(0.0)


def test_query_loader_labels_plain_and_json_queries(tmp_path: Path) -> None:
    plain = tmp_path / "queries.txt"
    plain.write_text("first query\nsecond query\n", encoding="utf-8")
    assert [item["query_id"] for item in load_query_file(plain)] == ["q001", "q002"]

    structured = tmp_path / "queries.json"
    structured.write_text(
        json.dumps({"queries": [{"query_id": "known", "text": "a query"}]}),
        encoding="utf-8",
    )
    assert load_query_file(structured) == [{"query_id": "known", "text": "a query"}]

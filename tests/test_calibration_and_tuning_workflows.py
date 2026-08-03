"""Offline tests for bounded extraction calibration and cached-ranking tuning."""

from __future__ import annotations

import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.calibrate_extraction import _find_videos
from scripts.tune_retrieval import main as tune_main
from src.evaluation.textual_kis import GroundTruthInterval, GroundTruthQuery
from src.evaluation.tuning import (
    CachedChannelRanking,
    RetrievalTuningReport,
    load_cached_rankings,
    load_tuning_override,
    persist_tuning_report,
    tune_cached_rankings,
)
from src.extraction.calibration import (
    load_calibration_selection,
    persist_calibration_report,
    run_extraction_calibration,
)
from src.runtime import build_pipeline
from src.utils.io_helpers import load_config


def test_bounded_extraction_calibration_pools_per_video_coverage_and_persists(
    tmp_path: Path,
) -> None:
    videos = {"v1": tmp_path / "v1.mp4", "v2": tmp_path / "v2.mp4"}
    for path in videos.values():
        path.write_bytes(b"fixture")
    references = {"v1": (0, 1_000), "v2": (0, 8_000)}
    calls: list[tuple[str, float, float]] = []

    def runner(
        _path: Path, video_id: str, threshold: float, gap: float, _scratch: Path
    ) -> tuple[int, ...]:
        calls.append((video_id, threshold, gap))
        candidates = {
            (0.25, 2.0): {"v1": (0, 1_000), "v2": (0, 8_000)},
            (0.30, 2.0): {"v1": (0,), "v2": (0,)},
            (0.40, 2.0): {"v1": (0,), "v2": (0, 8_000)},
        }
        return candidates[(threshold, gap)][video_id]

    report = run_extraction_calibration(
        videos,
        references,
        scene_thresholds=(0.25, 0.30, 0.40),
        max_gaps_seconds=(2.0,),
        minimum_coverage=0.95,
        tolerance_ms=0,
        runner=runner,
    )

    assert report.selected.scene_threshold == 0.25
    assert report.selected.coverage == 1.0
    assert report.selected.frame_count == 4
    assert len(calls) == 6
    destination = persist_calibration_report(report, tmp_path / "selection.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["selected"]["scene_threshold"] == 0.25
    assert payload["video_ids"] == ["v1", "v2"]
    assert len(payload["configuration_fingerprint"]) == 64
    assert {value["video_id"] for value in payload["inputs"]} == {"v1", "v2"}
    assert load_calibration_selection(destination) == report.selected


def test_calibration_discovers_selected_zip_member_without_expanding_archive(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "videos.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("videos/L21_V001.mp4", b"selected-video")
        archive.writestr("videos/L21_V002.mp4", b"unselected-video")

    found = _find_videos(tmp_path, ["L21_V001"])

    assert found["L21_V001"].path == archive_path
    assert found["L21_V001"].member == "videos/L21_V001.mp4"
    assert not (tmp_path / "videos").exists()


def test_bounded_calibration_uses_configured_fallback_when_target_is_missed(
    tmp_path: Path,
) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fixture")

    report = run_extraction_calibration(
        {"v": video},
        {"v": (0, 1_000)},
        scene_thresholds=(0.30,),
        max_gaps_seconds=(2.0,),
        minimum_coverage=1.0,
        tolerance_ms=0,
        runner=lambda *_args: (0,),
    )

    assert report.selected.used_fallback is True
    assert report.selected.coverage == 0.5


def _labeled_fixture():
    ground_truth = (
        GroundTruthQuery("q1", (GroundTruthInterval("L21_V002", 10, 10),)),
    )
    cached = {
        "q1": {
            "visual": CachedChannelRanking(
                (
                    {"video_id": "L21_V001", "source_frame_idx": 1},
                    {"video_id": "L21_V002", "source_frame_idx": 10},
                ),
                5.0,
            ),
            "ocr": CachedChannelRanking(
                ({"video_id": "L21_V002", "source_frame_idx": 10},),
                2.0,
            ),
        }
    }
    return ground_truth, cached


def test_cached_tuning_optimizes_score_and_reports_cumulative_ablations(
    tmp_path: Path,
) -> None:
    ground_truth, cached = _labeled_fixture()

    report = tune_cached_rankings(
        cached,
        ground_truth,
        default_rrf_k=60,
        default_depths={"visual": 2, "ocr": 1},
        default_weights={"visual": 1.0, "ocr": 0.0},
        rrf_k_values=(20, 60),
        weight_values={"ocr": (0.0, 2.0)},
    )

    assert report.used_defaults is False
    assert report.selected_weights["ocr"] == 2.0
    assert report.best_metrics == {
        "mean_top_k_score": 1.0,
        "r_at_1": 1.0,
        "p95_latency_ms": 7.0,
    }
    assert [item.channels for item in report.ablations] == [
        ("visual",),
        ("visual", "ocr"),
    ]
    persisted = persist_tuning_report(report, tmp_path / "tuning.json")
    assert json.loads(persisted.read_text(encoding="utf-8"))["candidates_evaluated"] == 4
    override = load_tuning_override(persisted, expected_channels=("visual", "ocr"))
    assert override is not None
    assert override.rrf_k == report.selected_rrf_k


def test_tuning_preserves_defaults_without_labels_and_cli_is_runnable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tune_cached_rankings(
        None,
        None,
        default_rrf_k=60,
        default_depths={"visual": 300},
        default_weights={"visual": 1.0},
    )
    assert report.used_defaults is True
    assert report.candidates_evaluated == 0

    output = tmp_path / "defaults.json"
    assert tune_main(["--output", str(output)]) == 0
    displayed = json.loads(capsys.readouterr().out)
    assert displayed["used_defaults"] is True
    assert output.is_file()
    assert load_tuning_override(output, expected_channels=tuple(displayed["selected"]["channel_weights"])) is None


def test_runtime_consumes_only_valid_labeled_tuning_record(tmp_path: Path) -> None:
    config = deepcopy(load_config())
    depths = {str(key): int(value) for key, value in config["retrieval"]["candidate_depths"].items()}
    weights = {
        str(key): float(value) for key, value in config["retrieval"]["channel_weights"].items()
    }
    depths["visual"] = 111
    weights["visual"] = 1.75
    report = RetrievalTuningReport(
        selected_rrf_k=20,
        selected_depths=depths,
        selected_weights=weights,
        used_defaults=False,
        reason="fixture",
        candidates_evaluated=1,
        objective={"primary": "mean_top_k_score"},
        best_metrics={"mean_top_k_score": 1.0, "r_at_1": 1.0, "p95_latency_ms": 1.0},
        ablations=(),
    )
    record = persist_tuning_report(report, tmp_path / "active.json")
    config["retrieval"]["tuning_record"] = str(record)
    config["models"]["organizer_clip"]["enabled"] = False
    config["models"]["accelerated_visual"]["enabled"] = False

    pipeline = build_pipeline(config)

    visual = next(channel for channel in pipeline.dense_channels if channel.name == "visual")
    assert pipeline.rrf_k == 20
    assert visual.depth == 111
    assert visual.weight == 1.75


def test_cached_ranking_loader_validates_versioned_query_shape(tmp_path: Path) -> None:
    path = tmp_path / "rankings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queries": [
                    {
                        "query_id": "q1",
                        "channels": {
                            "visual": {
                                "latency_ms": 3.5,
                                "results": [{"video_id": "V", "source_frame_idx": 4}],
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_cached_rankings(path)

    assert loaded["q1"]["visual"].latency_ms == 3.5
    assert loaded["q1"]["visual"].results[0]["source_frame_idx"] == 4

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version 1"):
        load_cached_rankings(path)

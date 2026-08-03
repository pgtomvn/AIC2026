"""Calibration helpers and bounded workflow for adaptive keyframe extraction."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.extraction.keyframe_extractor import KeyframeExtractor


@dataclass(frozen=True)
class CalibrationResult:
    scene_threshold: float
    max_gap_seconds: float
    coverage: float
    frame_count: int
    used_fallback: bool = False


@dataclass(frozen=True, slots=True)
class CalibrationGridRun:
    """Aggregate result for one threshold/max-gap pair."""

    scene_threshold: float
    max_gap_seconds: float
    coverage: float
    frame_count: int
    video_count: int


@dataclass(frozen=True, slots=True)
class CalibrationInput:
    """Reproducible identity of one selected loose video or ZIP member."""

    video_id: str
    source_uri: str
    source_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ExtractionCalibrationReport:
    """Persistable calibration report for an explicitly bounded video selection."""

    selected: CalibrationResult
    minimum_coverage: float
    tolerance_ms: int
    video_ids: tuple[str, ...]
    runs: tuple[CalibrationGridRun, ...]
    inputs: tuple[CalibrationInput, ...]
    calibration_config: dict[str, Any]
    configuration_fingerprint: str
    report_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": 1,
            "selected": asdict(self.selected),
            "minimum_coverage": self.minimum_coverage,
            "tolerance_ms": self.tolerance_ms,
            "video_ids": list(self.video_ids),
            "runs": [asdict(run) for run in self.runs],
            "inputs": [asdict(value) for value in self.inputs],
            "calibration_config": self.calibration_config,
            "configuration_fingerprint": self.configuration_fingerprint,
            "report_fingerprint": self.report_fingerprint,
        }


ExtractionRunner = Callable[[Path, str, float, float, Path], Sequence[int]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(value: Mapping[str, Any]) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def temporal_coverage(
    reference_timestamps_ms: Iterable[int],
    candidate_timestamps_ms: Iterable[int],
    *,
    tolerance_ms: int = 1_000,
) -> float:
    """Fraction of reference timestamps having a candidate within tolerance."""

    references = sorted(int(value) for value in reference_timestamps_ms)
    candidates = sorted(int(value) for value in candidate_timestamps_ms)
    if not references:
        return 1.0
    if not candidates:
        return 0.0
    covered = 0
    position = 0
    for reference in references:
        while position + 1 < len(candidates) and candidates[position + 1] <= reference:
            position += 1
        distances = [abs(candidates[position] - reference)]
        if position + 1 < len(candidates):
            distances.append(abs(candidates[position + 1] - reference))
        if min(distances) <= tolerance_ms:
            covered += 1
    return covered / len(references)


def select_calibration(
    runs: Mapping[tuple[float, float], Iterable[int]],
    reference_timestamps_ms: Iterable[int],
    *,
    minimum_coverage: float = 0.95,
    tolerance_ms: int = 1_000,
    fallback: tuple[float, float] = (0.30, 2.0),
) -> CalibrationResult:
    """Choose the smallest qualifying extraction output from a parameter grid."""

    reference = tuple(reference_timestamps_ms)
    evaluated: list[CalibrationResult] = []
    for (threshold, gap), timestamps in runs.items():
        values = tuple(timestamps)
        evaluated.append(
            CalibrationResult(
                scene_threshold=float(threshold),
                max_gap_seconds=float(gap),
                coverage=temporal_coverage(reference, values, tolerance_ms=tolerance_ms),
                frame_count=len(values),
            )
        )
    qualifying = [result for result in evaluated if result.coverage >= minimum_coverage]
    if qualifying:
        return min(
            qualifying,
            key=lambda result: (
                result.frame_count,
                -result.coverage,
                result.max_gap_seconds,
                result.scene_threshold,
            ),
        )
    fallback_result = next(
        (
            result
            for result in evaluated
            if (result.scene_threshold, result.max_gap_seconds) == fallback
        ),
        None,
    )
    if fallback_result is None:
        return CalibrationResult(*fallback, coverage=0.0, frame_count=0, used_fallback=True)
    return CalibrationResult(
        fallback_result.scene_threshold,
        fallback_result.max_gap_seconds,
        fallback_result.coverage,
        fallback_result.frame_count,
        used_fallback=True,
    )


def _default_extraction_runner(
    video_path: Path,
    video_id: str,
    threshold: float,
    max_gap_seconds: float,
    output_dir: Path,
) -> Sequence[int]:
    extractor = KeyframeExtractor(
        output_dir=output_dir,
        batch_id="calibration",
        min_scene_change=threshold,
        max_gap_seconds=max_gap_seconds,
    )
    return tuple(
        int(record["timestamp_ms"])
        for record in extractor.extract(video_path, video_id=video_id)
    )


def run_extraction_calibration(
    videos: Mapping[str, str | Path],
    reference_timestamps_ms: Mapping[str, Iterable[int]],
    *,
    scene_thresholds: Sequence[float] = (0.25, 0.30, 0.35, 0.40),
    max_gaps_seconds: Sequence[float] = (2.0, 3.0, 4.0),
    minimum_coverage: float = 0.95,
    tolerance_ms: int = 1_000,
    fallback: tuple[float, float] = (0.30, 2.0),
    runner: ExtractionRunner | None = None,
    input_fingerprints: Mapping[str, Mapping[str, Any]] | None = None,
    extractor_settings: Mapping[str, Any] | None = None,
) -> ExtractionCalibrationReport:
    """Evaluate a bounded grid without persisting extracted frames.

    Coverage is computed independently per video before numerators and denominators
    are pooled.  This prevents timestamps near the beginning of one video from
    accidentally covering reference timestamps belonging to another video.
    """

    if not videos:
        raise ValueError("at least one explicitly selected raw video is required")
    if set(videos) != set(reference_timestamps_ms):
        raise ValueError("videos and reference timestamps must have identical video IDs")
    thresholds = tuple(float(value) for value in scene_thresholds)
    gaps = tuple(float(value) for value in max_gaps_seconds)
    if not thresholds or not gaps:
        raise ValueError("calibration grids cannot be empty")
    if not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be between zero and one")
    if tolerance_ms < 0:
        raise ValueError("tolerance_ms cannot be negative")
    selected_ids = tuple(sorted(videos))
    references = {
        video_id: tuple(int(value) for value in reference_timestamps_ms[video_id])
        for video_id in selected_ids
    }
    if not all(references.values()):
        raise ValueError("every selected video must have reference timestamps")
    active_runner = runner or _default_extraction_runner
    calibration_config = {
        "scene_threshold_grid": list(thresholds),
        "max_gap_grid_seconds": list(gaps),
        "minimum_coverage": float(minimum_coverage),
        "tolerance_ms": int(tolerance_ms),
        "fallback": [float(fallback[0]), float(fallback[1])],
        "extractor_settings": dict(extractor_settings or {}),
    }
    inputs: list[CalibrationInput] = []
    for video_id in selected_ids:
        supplied = None if input_fingerprints is None else input_fingerprints.get(video_id)
        path = Path(videos[video_id])
        if supplied is None:
            inputs.append(
                CalibrationInput(video_id, str(path.resolve()), _sha256(path), path.stat().st_size)
            )
            continue
        source_uri = supplied.get("source_uri")
        source_sha256 = supplied.get("source_sha256")
        size_bytes = supplied.get("size_bytes")
        if (
            not isinstance(source_uri, str)
            or not source_uri
            or not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError(f"invalid input fingerprint for {video_id}")
        inputs.append(CalibrationInput(video_id, source_uri, source_sha256, size_bytes))
    evaluated: list[CalibrationGridRun] = []
    with tempfile.TemporaryDirectory(prefix="aic-kis-calibration-") as temporary:
        scratch = Path(temporary)
        for threshold in thresholds:
            for gap in gaps:
                covered = 0
                reference_count = 0
                frame_count = 0
                for video_id in selected_ids:
                    path = Path(videos[video_id])
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    candidates = tuple(
                        int(value)
                        for value in active_runner(path, video_id, threshold, gap, scratch)
                    )
                    refs = references[video_id]
                    covered += round(
                        temporal_coverage(refs, candidates, tolerance_ms=tolerance_ms) * len(refs)
                    )
                    reference_count += len(refs)
                    frame_count += len(candidates)
                evaluated.append(
                    CalibrationGridRun(
                        scene_threshold=threshold,
                        max_gap_seconds=gap,
                        coverage=covered / reference_count,
                        frame_count=frame_count,
                        video_count=len(selected_ids),
                    )
                )
    qualifying = [run for run in evaluated if run.coverage >= minimum_coverage]
    if qualifying:
        winner = min(
            qualifying,
            key=lambda run: (
                run.frame_count,
                -run.coverage,
                run.max_gap_seconds,
                run.scene_threshold,
            ),
        )
        selected = CalibrationResult(
            winner.scene_threshold,
            winner.max_gap_seconds,
            winner.coverage,
            winner.frame_count,
        )
    else:
        fallback_run = next(
            (
                run
                for run in evaluated
                if (run.scene_threshold, run.max_gap_seconds) == fallback
            ),
            None,
        )
        selected = CalibrationResult(
            fallback[0],
            fallback[1],
            0.0 if fallback_run is None else fallback_run.coverage,
            0 if fallback_run is None else fallback_run.frame_count,
            used_fallback=True,
        )
    report_content = {
        "selected": asdict(selected),
        "inputs": [asdict(value) for value in inputs],
        "runs": [asdict(value) for value in evaluated],
    }
    return ExtractionCalibrationReport(
        selected=selected,
        minimum_coverage=minimum_coverage,
        tolerance_ms=tolerance_ms,
        video_ids=selected_ids,
        runs=tuple(evaluated),
        inputs=tuple(inputs),
        calibration_config=calibration_config,
        configuration_fingerprint=_config_fingerprint(calibration_config),
        report_fingerprint=_config_fingerprint(report_content),
    )


def load_calibration_selection(path: str | Path) -> CalibrationResult:
    """Validate and load an operational extraction selection record."""

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("calibration record requires schema_version 1")
    config = payload.get("calibration_config")
    if not isinstance(config, Mapping):
        raise ValueError("calibration record is missing calibration_config")
    if payload.get("configuration_fingerprint") != _config_fingerprint(config):
        raise ValueError("calibration record configuration fingerprint mismatch")
    raw = payload.get("selected")
    if not isinstance(raw, Mapping):
        raise ValueError("calibration record is missing selected parameters")
    result = CalibrationResult(
        scene_threshold=float(raw["scene_threshold"]),
        max_gap_seconds=float(raw["max_gap_seconds"]),
        coverage=float(raw["coverage"]),
        frame_count=int(raw["frame_count"]),
        used_fallback=bool(raw.get("used_fallback", False)),
    )
    report_content = {
        "selected": dict(raw),
        "inputs": payload.get("inputs"),
        "runs": payload.get("runs"),
    }
    if payload.get("report_fingerprint") != _config_fingerprint(report_content):
        raise ValueError("calibration record report fingerprint mismatch")
    if not 0 <= result.scene_threshold <= 1 or result.max_gap_seconds <= 0:
        raise ValueError("calibration record contains invalid extraction parameters")
    if not 0 <= result.coverage <= 1 or result.frame_count < 0:
        raise ValueError("calibration record contains invalid metrics")
    return result


def persist_calibration_report(
    report: ExtractionCalibrationReport, destination: str | Path
) -> Path:
    """Atomically persist a calibration report without modifying project defaults."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return path


__all__ = [
    "CalibrationGridRun",
    "CalibrationInput",
    "CalibrationResult",
    "ExtractionCalibrationReport",
    "persist_calibration_report",
    "load_calibration_selection",
    "run_extraction_calibration",
    "select_calibration",
    "temporal_coverage",
]

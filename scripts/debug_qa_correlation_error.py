"""Diagnose a sanitized QA correlation event and reproduce stale contract skew."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import (  # noqa: E402
    QACandidateRequest,
    QACompetitionSpec,
    QAPrediction,
)
from src.qa_submission import serialize_qa_submission  # noqa: E402
from src.query.entity_extractor import EntityExtractor  # noqa: E402
from src.query.query_decomposer import decompose_query  # noqa: E402
from src.vlm.qa_reasoner import QAImage  # noqa: E402

_CORRELATION = re.compile(r"^ERR-QA-[0-9A-F]{8}$")
_TRACE_FRAME = re.compile(r'File "([^"]+)", line (\d+), in ([^\r\n]+)')
_WINDOWS_PATH = re.compile(r"[A-Za-z]:\\[^\s\"']+")


@dataclass(frozen=True, slots=True)
class _LegacyQACandidateRequest:
    query_id: str
    event_description: str
    batch_filters: tuple[str, ...] = ()
    top_k: int = 5
    query_language: str = "auto"
    event_description_en: str | None = None


def _safe_message(value: object) -> str:
    text = str(value or "")[:512]
    return _WINDOWS_PATH.sub("<path>", text)


def _project_frames(text: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for raw_path, raw_line, function in _TRACE_FRAME.findall(text):
        try:
            relative = Path(raw_path).resolve().relative_to(PROJECT_ROOT.resolve())
        except (OSError, ValueError):
            continue
        if not relative.parts or relative.parts[0] not in {"app", "scripts", "src"}:
            continue
        frames.append(
            {
                "file": relative.as_posix(),
                "line": int(raw_line),
                "function": function.strip()[:128],
            }
        )
    return frames


def load_correlation_event(log_path: Path, correlation_id: str) -> dict[str, Any] | None:
    """Return one allowlisted log event without exposing request content or absolute paths."""

    with log_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            record = payload.get("record") if isinstance(payload, dict) else None
            extra = record.get("extra") if isinstance(record, dict) else None
            if not isinstance(extra, dict) or extra.get("event_id") != correlation_id:
                continue
            exception = record.get("exception")
            exception = exception if isinstance(exception, dict) else {}
            traceback_text = str(payload.get("text") or "")
            return {
                "correlation_id": correlation_id,
                "workflow": extra.get("workflow"),
                "stage": extra.get("stage"),
                "batch_id": extra.get("batch_id"),
                "exception_type": str(exception.get("type") or "unknown")[:128],
                "exception_message": _safe_message(exception.get("value")),
                "project_frames": _project_frames(traceback_text),
                "payload_available": False,
            }
    return None


def _contract_probe() -> dict[str, Any]:
    payload = {
        "query_id": "synthetic-correlation-reproducer",
        "event_description": "A television news presenter is speaking.",
        "batch_filters": ("aic25-b1-reference",),
        "top_k": 5,
        "query_language": "en",
        "event_description_en": None,
        "entity_constraints": (),
    }
    stale_message = ""
    try:
        _LegacyQACandidateRequest(**payload)  # type: ignore[arg-type]
    except TypeError as error:
        stale_message = _safe_message(error)
    fresh = QACandidateRequest(**payload)
    return {
        "stale_failure_reproduced": "entity_constraints" in stale_message,
        "stale_exception": stale_message,
        "fresh_contract_accepted": fresh.query_id == payload["query_id"],
    }


def _component_probes() -> dict[str, Any]:
    query = "Một chương trình từ thiện của câu lạc bộ FANA tại tỉnh Khánh Hòa."
    entities = EntityExtractor().extract(query)
    decomposition = decompose_query(
        query,
        tuple((entity, "optional") for entity in entities),
        english_text="a charity club distributes gifts in a rural village",
    )

    image_bytes = b"\x89PNG\r\n\x1a\nsynthetic-offline-probe"
    image = QAImage(
        data=image_bytes,
        mime_type="image/png",
        index=1,
        source_frame_idx=10,
        timestamp_ms=1_000,
        sha256=hashlib.sha256(image_bytes).hexdigest(),
    )

    spec = QACompetitionSpec.conservative_general_text()
    normalized = spec.normalize_answer('  "Năm\nngười"  ')
    prediction = QAPrediction(
        rank=1,
        batch_id="synthetic",
        video_id="L01_V001",
        source_frame_idx=10,
        timestamp_ms=1_000,
        answer=normalized,
        confidence_score=1.0,
        answer_source="operator",
        window_id="synthetic:L01_V001:10",
        evidence={"candidate_id": "synthetic:L01_V001:10"},
    )
    csv_bytes = serialize_qa_submission(
        (prediction,),
        spec=spec,
        batch_id="synthetic",
        is_batch_active=lambda value: value == "synthetic",
        is_canonical_source_frame=lambda batch, video, frame: (
            batch == "synthetic" and video == "L01_V001" and frame == 10
        ),
    )
    return {
        "entity_decomposition": {
            "passed": {entity.text for entity in entities} == {"FANA", "Khánh Hòa"},
            "entity_count": len(entities),
            "visual_description_available": bool(decomposition.visual_description_en),
        },
        "single_image_packaging": {
            "passed": image.index == 1 and image.source_frame_idx == 10,
            "image_count": 1,
        },
        "qa_formatting": {
            "passed": csv_bytes == "L01_V001,10,năm người\n".encode(),
            "normalized_answer": normalized,
            "csv_preview": csv_bytes.decode("utf-8").strip(),
        },
        "provider_called": False,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correlation-id", default="ERR-QA-47022AFB")
    parser.add_argument("--log", type=Path, default=PROJECT_ROOT / "outputs/logs/app_error.log")
    args = parser.parse_args()
    correlation_id = str(args.correlation_id).strip().upper()
    if not _CORRELATION.fullmatch(correlation_id):
        print(json.dumps({"code": "correlation_id_invalid", "passed": False}))
        return 2
    log_path = args.log if args.log.is_absolute() else PROJECT_ROOT / args.log
    if not log_path.is_file():
        print(json.dumps({"code": "correlation_log_missing", "passed": False}))
        return 2
    event = load_correlation_event(log_path, correlation_id)
    if event is None:
        print(json.dumps({"code": "correlation_event_missing", "passed": False}))
        return 2

    contract = _contract_probe()
    probes = _component_probes()
    root_cause = (
        "streamlit_hot_reload_contract_skew"
        if event["exception_type"] == "TypeError"
        and "entity_constraints" in event["exception_message"]
        else "unclassified"
    )
    passed = bool(
        root_cause == "streamlit_hot_reload_contract_skew"
        and contract["stale_failure_reproduced"]
        and contract["fresh_contract_accepted"]
        and all(
            bool(probes[name]["passed"])
            for name in ("entity_decomposition", "single_image_packaging", "qa_formatting")
        )
    )
    report = {
        "event": event,
        "root_cause": root_cause,
        "execution_context": {
            "request_contract_construction": "failed",
            "entity_decomposition": "not_reached_in_original_event",
            "candidate_resolution": "not_reached_in_original_event",
            "vlm_routing": "not_reached_in_original_event",
            "response_formatting": "not_reached_in_original_event",
        },
        "contract_probe": contract,
        "offline_component_probes": probes,
        "passed": passed,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

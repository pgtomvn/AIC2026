#!/usr/bin/env python3
"""Build a deterministic OCR quality sidecar from an immutable JSONL artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.ocr_quality import (  # noqa: E402
    OCRQualityFeature,
    build_ocr_quality_features,
    write_ocr_quality_sidecar_streaming,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> Iterator[Mapping[str, Any]]:
    opener = gzip.open if path.suffix.casefold() == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"OCR record {line_number} must be an object")
            yield value


def _image_sizes(path: Path | None) -> dict[str, tuple[int, int]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError("image-size manifest must be an object")
    sizes: dict[str, tuple[int, int]] = {}
    for video_id, dimensions in value.items():
        if (
            not isinstance(video_id, str)
            or not isinstance(dimensions, (list, tuple))
            or len(dimensions) != 2
        ):
            raise ValueError("image-size manifest entry is invalid")
        sizes[video_id] = (int(dimensions[0]), int(dimensions[1]))
    return sizes


def _feature_groups(
    records: Iterable[Mapping[str, Any]],
    *,
    image_sizes: Mapping[str, tuple[int, int]],
) -> Iterator[Sequence[OCRQualityFeature]]:
    """Yield one bounded video group and reject non-contiguous input ordering."""

    previous_video_id: str | None = None
    for video_id, group in groupby(records, key=lambda value: str(value.get("video_id", ""))):
        if not video_id:
            raise ValueError("OCR quality input has a missing video_id")
        if previous_video_id is not None and video_id <= previous_video_id:
            raise ValueError("OCR artifact must be grouped in ascending video_id order")
        yield build_ocr_quality_features(group, image_sizes=image_sizes)
        previous_video_id = video_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256")
    parser.add_argument("--image-sizes", type=Path)
    args = parser.parse_args(argv)
    actual_sha256 = _sha256(args.ocr_jsonl)
    if args.source_sha256 and actual_sha256 != args.source_sha256.casefold():
        raise ValueError("OCR artifact SHA-256 does not match")
    sizes = _image_sizes(args.image_sizes)
    fingerprint, feature_count = write_ocr_quality_sidecar_streaming(
        args.output,
        _feature_groups(_records(args.ocr_jsonl), image_sizes=sizes),
        source_fingerprint=actual_sha256,
    )
    report = {
        "output": str(args.output),
        "source_sha256": actual_sha256,
        "feature_count": feature_count,
        "roi_available": bool(sizes),
        "fingerprint": fingerprint,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

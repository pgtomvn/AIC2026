"""Offline contracts for ordinal OCR staging; no models, services or downloads."""

import gzip
import hashlib
import io
import json
import zipfile

import pytest
from PIL import Image

from scripts.build_b02_ocr_notebooks import build
from scripts.colab_b02_ocr import pilot_selection, run_worker, validate_checkpoint
from src.extraction.vietocr_engine import VIETOCR_REVISION, VietOCREngine
from src.ingest.cloud_artifacts import _atomic_json, _stable_json_hash


class FakeEngine:
    identity = {"test": "fake-not-production"}

    def extract_text(self, image):
        return [{"text_raw": "Việt Nam", "confidence": 0.9}]


def fixture_plan(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = io.BytesIO()
    Image.new("RGB", (12, 12), "white").save(payload, format="PNG")
    videos = []
    with zipfile.ZipFile(raw / "frames.zip", "w") as archive:
        for video in ("M08_V001", "N001_001", "S01_V001"):
            frames = []
            for ordinal in (1, 2):
                member = f"keyframes/{video}/{ordinal:03}.png"
                archive.writestr(member, payload.getvalue())
                info = archive.getinfo(member)
                frames.append({"member": member, "archive": "frames.zip",
                               "crc32": info.CRC, "size_bytes": info.file_size,
                               "keyframe_ordinal": ordinal, "timestamp_ms": 0,
                               "source_frame_idx": 0})
            videos.append({"video_id": video, "worker_id": 0, "frames": frames})
    plan = {"batch_id": "aic26-b2", "worker_count": 2, "videos": videos}
    plan["fingerprint"] = _stable_json_hash(plan)
    return raw, plan


def test_pilot_distinct_from_full_and_preserves_equal_pts_and_frame_zero(tmp_path):
    raw, plan = fixture_plan(tmp_path)
    out = tmp_path / "out"
    result = run_worker(raw, out, plan, 0, FakeEngine(), "a" * 64)
    assert result["scanned_frames"] == 6
    assert result["documents"] == 6
    assert result["state"] == "worker_complete"
    assert not (out / "full").exists()
    receipt = json.loads((out / "pilot/receipts/M08_V001.json").read_text())
    assert receipt["coverage_complete"] is False
    with gzip.open(out / "pilot/shards/M08_V001.jsonl.gz", "rt", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    assert len({d["segment_id"] for d in docs}) == 2
    assert all(d["start_ms"] == 0 and d["source_frame_idx"] == 0 for d in docs)
    assert run_worker(raw, out, plan, 0, FakeEngine(), "a" * 64)["documents"] == 6
    with pytest.raises(RuntimeError, match="Pilot review"):
        run_worker(raw, out, plan, 0, FakeEngine(), "a" * 64, full=True)
    with pytest.raises(ValueError, match="Frozen plan"):
        run_worker(raw, out, plan, 0, FakeEngine(), "b" * 64)


def test_no_text_is_success_but_inference_error_is_not(tmp_path):
    raw, plan = fixture_plan(tmp_path)

    class Empty(FakeEngine):
        def extract_text(self, image):
            return []

    class Broken(FakeEngine):
        def extract_text(self, image):
            raise RuntimeError("inference broken")

    out = tmp_path / "zero"
    run_worker(raw, out, plan, 0, Empty(), "a" * 64)
    receipt = json.loads((out / "pilot/receipts/M08_V001.json").read_text())
    assert receipt["zero_text"] is True and receipt["scanned_frames"] == 2
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="inference broken"):
        run_worker(raw, failed, plan, 0, Broken(), "a" * 64)
    assert not (failed / "pilot/receipts/M08_V001.json").exists()
    assert json.loads((failed / "pilot/progress/worker-0.json").read_text())["state"] == "failed"


def test_checkpoint_corruption_and_orphan_are_not_overwritten(tmp_path):
    raw, plan = fixture_plan(tmp_path)
    out = tmp_path / "out"
    run_worker(raw, out, plan, 0, FakeEngine(), "a" * 64)
    path = out / "pilot/shards/M08_V001.jsonl.gz"
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        run_worker(raw, out, plan, 0, FakeEngine(), "a" * 64)
    assert path.read_bytes() == b"broken"
    with pytest.raises(ValueError, match="Orphan"):
        validate_checkpoint(tmp_path / "missing.json", path, {})


def test_pilot_sampling_is_deterministic_and_balanced(tmp_path):
    _, plan = fixture_plan(tmp_path)
    assert pilot_selection(plan) == pilot_selection(plan)
    assert len(pilot_selection(plan)) == 3
    assert sum(map(len, pilot_selection(plan).values())) == 6


def test_frozen_assets_are_verified_without_loading_models(tmp_path):
    files = {}
    for name in ("detector.onnx", "vgg_seq2seq.pth", "config.json"):
        (tmp_path / name).write_bytes(b"fixture")
        files[name] = hashlib.sha256(b"fixture").hexdigest()
    _atomic_json(tmp_path / "assets.json", {"revision": VIETOCR_REVISION, "files": files})
    engine = VietOCREngine(tmp_path)
    assert engine._recognizer is None
    assert engine.identity["files"] == files
    assert engine._recognizer is None
    (tmp_path / "detector.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        VietOCREngine(tmp_path).identity


def test_notebooks_have_three_code_cells_frozen_bundle_and_no_secrets(tmp_path):
    report = build(tmp_path)
    import ast
    import base64

    for path in report["notebooks"]:
        from pathlib import Path

        notebook = json.loads(Path(path).read_text(encoding="utf-8"))
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        assert len(cells) == 3
        for cell in cells:
            compile(cell, path, "exec")
        nodes = ast.parse(cells[1]).body
        payload = base64.b64decode(ast.literal_eval(nodes[1].value))
        assert hashlib.sha256(payload).hexdigest() == report["source_sha256"]
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            assert "b02-p0-manifest.json" in z.namelist()
            assert all(not name.startswith(("outputs/", "models/", "data/"))
                       and ".env" not in name for name in z.namelist())
        assert "--full" not in cells[2]

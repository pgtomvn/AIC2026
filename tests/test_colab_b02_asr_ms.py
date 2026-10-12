"""M/S-only ASR scope does not read traffic, keyframe or object payloads."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_b02_asr_ms_notebook import build
from scripts.colab_video_asr import inventory_b02_ms
from src.ingest.cloud_artifacts import _stable_json_hash

P0_DIR = Path(__file__).resolve().parents[1] / "outputs" / "checkpoints" / "b02-p0"
P0_MANIFEST = P0_DIR / "b02-p0-manifest-v3.json"
requires_p0_manifest = pytest.mark.skipif(
    not P0_MANIFEST.is_file(), reason="needs the local B02 P0 manifest (generated data, not in Git)"
)


def sources(tmp_path):
    videos = {f"M01_V{i:03}": {} for i in range(1, 305)}
    videos.update({f"S01_V{i:03}": {} for i in range(1, 13)})
    videos.update({f"N001_{i:03}": {} for i in range(1, 299)})
    manifest = {"batch_id": "aic26-b2", "videos": videos}
    manifest["manifest_sha256"] = _stable_json_hash(manifest)
    with zipfile.ZipFile(tmp_path / "Videos_M01.zip", "w") as z:
        for video in videos:
            if video.startswith("M"):
                z.writestr(f"M01/{video}.mp4", b"fixture")
    with zipfile.ZipFile(tmp_path / "Video_S01.zip", "w") as z:
        for video in videos:
            if video.startswith("S"):
                z.writestr(f"{video}.mp4", b"fixture")
    (tmp_path / "Video_N001-N010.zip").write_bytes(b"must not read")
    (tmp_path / "Keyframes_M01.zip").write_bytes(b"must not read")
    return manifest


def test_ms_only_excludes_all_n_without_no_audio_receipts(tmp_path):
    manifest = sources(tmp_path)
    plan = inventory_b02_ms(tmp_path, manifest, 1)
    assert plan["expected_videos"] == 316
    assert len(plan["excluded_video_ids"]) == 298
    assert all(v["video_id"].startswith(("M", "S")) for v in plan["videos"])
    assert plan == inventory_b02_ms(tmp_path, manifest, 1)
    assert "no_audio_videos" not in plan


def test_missing_ms_source_and_unexpected_video_fail_closed(tmp_path):
    manifest = sources(tmp_path)
    manifest["videos"]["M10_V999"] = manifest["videos"].pop("M01_V001")
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = _stable_json_hash(manifest)
    with pytest.raises(ValueError, match="coverage mismatch"):
        inventory_b02_ms(tmp_path, manifest, 1)


@requires_p0_manifest
def test_three_cells_cpu_one_video_new_namespace(tmp_path):
    report = build(tmp_path)
    assert hashlib.sha256(Path(report["bundle"]).read_bytes()).hexdigest() == report["source_sha256"]
    notebook = json.loads(Path(report["notebook"]).read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 3 and "accelerator" not in notebook["metadata"]
    for cell in cells:
        compile(cell, "notebook", "exec")
    assert "--max-videos', '1'" in cells[2]
    assert "asr-ms-cpu-seekable-20260925-v2" in cells[1]
    assert "--b02-ms-manifest" in cells[1]
    assert "'--only-group', 'asr-colab'" in cells[1]
    assert "capture_output=True" in cells[2]
    assert "worker-0-pilot.log" in cells[2]

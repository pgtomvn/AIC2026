"""Explicit Colab-only asset download; subsequent starts verify frozen bytes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction.vietocr_engine import VIETOCR_REVISION, VietOCREngine, digest_file
from src.ingest.cloud_artifacts import _atomic_json


def prepare(destination: Path) -> None:
    """Freeze official upstream assets once. No production data is touched."""
    if (destination / "assets.json").exists():
        print(json.dumps(VietOCREngine(destination).identity))
        return
    import yaml

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError("Incomplete asset directory needs audit; refusing replacement")
    base = f"https://raw.githubusercontent.com/pbcquoc/vietocr/{VIETOCR_REVISION}/config"
    config = {}
    for name in ("base.yml", "vgg-seq2seq.yml"):
        with urllib.request.urlopen(f"{base}/{name}", timeout=60) as response:
            config.update(yaml.safe_load(response.read()))
    weights_url = config["weights"]
    if weights_url != "https://vocr.vn/data/vietocr/vgg_seq2seq.pth":
        raise ValueError("Unexpected official weights URL")
    detector_url = (
        "https://huggingface.co/DjB314/RapidOCR/resolve/"
        "04e88d0483f0bfcdd0f29429594244c10bfc2867/"
        "v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx"
    )
    for name, url in (("detector.onnx", detector_url), ("vgg_seq2seq.pth", weights_url)):
        temporary = destination / f"{name}.part"
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if not temporary.stat().st_size:
            raise ValueError("Empty model download")
        if name == "detector.onnx":
            expected = next(line.split()[0] for line in
                            (ROOT / "checksums/rapidocr-reference.sha256").read_text().splitlines()
                            if line.endswith("/PP-OCRv6_det_small.onnx"))
            if digest_file(temporary) != expected:
                raise ValueError("Detector digest mismatch")
        os.replace(temporary, destination / name)
    config["cnn"]["pretrained"] = False
    config["predictor"]["beamsearch"] = False
    _atomic_json(destination / "config.json", config)
    _atomic_json(destination / "assets.json", {
        "revision": VIETOCR_REVISION,
        "recognizer_origin": weights_url,
        "recognizer_verification": "first_download_sha256; quality_pilot_required",
        "files": {name: digest_file(destination / name)
                  for name in ("detector.onnx", "vgg_seq2seq.pth", "config.json")},
    })
    print(json.dumps(VietOCREngine(destination).identity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    prepare(parser.parse_args().assets)

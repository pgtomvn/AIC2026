"""Build two self-contained, three-code-cell B02 OCR Colab notebooks."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''import base64, hashlib, io, os, shutil, subprocess, sys, zipfile
from pathlib import Path
raw = Path('/content/drive/MyDrive/AIC2026/B02/raw')
out = raw.parent / 'out' / 'ocr-vietocr-v1'
assets = raw.parent / 'source' / 'ocr-vietocr-v1-assets'
repo = Path('/content/aic2026-b02-ocr')
assert raw.is_dir(), 'Mount the Drive containing AIC2026/B02/raw'
payload = base64.b64decode(SOURCE_BASE64)
assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
repo.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as source:
    for member in source.infolist():
        target = (repo / member.filename).resolve()
        assert repo.resolve() in target.parents, 'Unsafe source member'
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open(member) as src, target.open('wb') as dst:
            shutil.copyfileobj(src, dst)
os.chdir(repo)
os.environ['PYTHONUTF8'] = '1'
os.environ['UV_CACHE_DIR'] = '/content/uv-b02-ocr-cache'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.11.16'], check=True)
subprocess.run(['uv', 'python', 'install', '3.11'], check=True)
subprocess.run(['uv', 'sync', '--python', '3.11', '--only-group', 'ocr-vietocr', '--frozen'], check=True)
# --only-group excludes the project dependencies, so use the environment directly.
python = str(repo / '.venv/bin/python')
subprocess.run([python, '-c', 'import torch; assert torch.cuda.is_available(), "Select T4 GPU first"; print(torch.cuda.get_device_name(0))'], check=True)
command = [python, 'scripts/colab_b02_ocr.py', '--raw', str(raw), '--output', str(out),
    '--manifest', 'b02-p0-manifest.json', '--assets', str(assets),
    '--worker-id', str(WORKER_ID), '--source-sha256', SOURCE_SHA256, '--resume']
subprocess.run(command + ['--inventory-only'], check=True)
subprocess.run([python, 'scripts/prepare_b02_vietocr.py', '--assets', str(assets)], check=True)
print('Ready for PILOT. 300 frames total across M/N/S and both workers. No index mutation.')
'''


def build(destination: Path) -> dict:
    """Allowlist source only; never include notebooks, credentials, models or data trees."""
    files = {p.relative_to(ROOT).as_posix(): p for p in (ROOT / "src").rglob("*.py")}
    for name in ("pyproject.toml", "uv.lock", "checksums/rapidocr-reference.sha256",
                 "scripts/colab_b02_ocr.py", "scripts/prepare_b02_vietocr.py",
                 "scripts/colab_video_asr.py"):
        files[name] = ROOT / name
    files["b02-p0-manifest.json"] = ROOT / "outputs/checkpoints/b02-p0/b02-p0-manifest-v3.json"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            if path.is_symlink() or ROOT not in path.resolve().parents:
                raise ValueError("Unsafe source path")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for worker in (0, 1):
        sources = [
            f"WORKER_ID = {worker}\nfrom google.colab import drive\ndrive.mount('/content/drive')\n",
            f"SOURCE_SHA256 = {sha!r}\nSOURCE_BASE64 = {encoded!r}\n" + SETUP,
            "# Pilot first. Full corpus requires a reviewed pilot-approval.json.\n"
            "subprocess.run(command, check=True)\n",
        ]
        notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
            "accelerator": "GPU", "colab": {"name": f"B02 VietOCR worker {worker}",
                                             "gpuType": "T4", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        }, "cells": [{"cell_type": "markdown", "id": "instructions", "metadata": {},
                       "source": [f"# B02 OCR · worker {worker}/2\n",
                                  "1. Mount Drive. 2. Setup + validate sources. 3. Run pilot.\n",
                                  "PP-OCRv6 detector (CPU) + VietOCR vgg_seq2seq (CUDA).\n",
                                  "Do not launch duplicate worker IDs. Start setup sequentially.\n",
                                  "Checkpoint: AIC2026/B02/out/ocr-vietocr-v1.\n",
                                  "Pilot results are NOT full-video receipts or production data.\n"]}]}
        for index, source in enumerate(sources):
            notebook["cells"].append({"cell_type": "code", "id": f"cell-{index + 1}",
                                      "metadata": {}, "source": source.splitlines(keepends=True),
                                      "execution_count": None, "outputs": []})
        path = destination / f"aic2026_b02_vietocr_worker_{worker}.ipynb"
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        paths.append(str(path))
    report = {"source_sha256": sha, "source_bytes": len(payload), "notebooks": paths,
              "state": "prepared_not_executed", "full_run_requires_pilot_review": True}
    (destination / "build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "outputs/colab/b02-ocr-vietocr-v1"), indent=2))

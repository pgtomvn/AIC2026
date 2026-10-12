"""Build self-contained Colab workers from a credential-free frozen source ZIP."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from build_colab_source_bundle import build_colab_source_bundle

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/colab/b02-asr-v1"


def cell(source: str, name: str) -> dict:
    return {"cell_type": "code", "id": name, "metadata": {},
            "execution_count": None, "outputs": [], "source": source.splitlines(True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    cpu = args.device == "cpu"
    destination = ROOT / "outputs/colab/b02-asr-cpu-v1" if cpu else DEST
    report = build_colab_source_bundle(ROOT, destination / "b02-asr-source.zip")
    payload = base64.b64encode(Path(report["bundle"]).read_bytes()).decode("ascii")
    bootstrap = '''import base64, hashlib, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/AIC2026/B02')
RAW_ROOT = DRIVE_ROOT / 'raw'
OUTPUT_ROOT = DRIVE_ROOT / 'out' / 'asr-video-only-v1'
REPO_ROOT = Path('/content/aic2026-b02-asr')
assert RAW_ROOT.is_dir(), 'Mount the Drive containing AIC2026/B02/raw first'
payload = base64.b64decode(SOURCE_BASE64)
assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
REPO_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as source:
    for member in source.infolist():
        target = (REPO_ROOT / member.filename).resolve()
        assert REPO_ROOT.resolve() in target.parents, 'Unsafe source member'
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as inp, target.open('wb') as out:
                shutil.copyfileobj(inp, out)
os.chdir(REPO_ROOT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.11.16'], check=True)
subprocess.run(['uv', 'python', 'install', '3.11'], check=True)
subprocess.run(['uv', 'sync', '--python', '3.11', '--group', 'competition', '--frozen'], check=True)
command = ['uv', 'run', '--frozen', 'python', 'scripts/colab_video_asr.py',
    '--raw-dir', str(RAW_ROOT), '--output-dir', str(OUTPUT_ROOT),
    '--batch-id', 'aic26-b2', '--worker-count', '2', '--worker-id', str(WORKER_ID),
    '--ignore-empty', 'Videos_M010.zip', '--source-sha256', SOURCE_SHA256, '--resume']
subprocess.run(command + ['--inventory-only'], check=True)
gpu = subprocess.check_output(['uv', 'run', '--frozen', 'python', '-c',
    'import torch; assert torch.cuda.is_available(), "GPU required"; print(torch.cuda.get_device_name(0))'], text=True)
print('GPU:', gpu.strip(), 'worker:', WORKER_ID, 'source:', SOURCE_SHA256, flush=True)
subprocess.run(['uv', 'run', '--frozen', 'python', 'scripts/prepare_models.py', '--only', 'asr'], check=True)
print('Ready. Run the final cell. No B01/index mutation.', flush=True)
'''
    if cpu:
        bootstrap = bootstrap.replace("'asr-video-only-v1'", "'asr-video-only-cpu-v1'")
        bootstrap = bootstrap.replace("'--worker-count', '2'", "'--worker-count', '1'")
        bootstrap = bootstrap.replace("SOURCE_SHA256, '--resume']",
                                      "SOURCE_SHA256, '--resume', '--device', 'cpu']")
        start = bootstrap.index("gpu = subprocess.check_output")
        end = bootstrap.index("subprocess.run(['uv', 'run', '--frozen', 'python', "
                              "'scripts/prepare_models.py'", start)
        bootstrap = (bootstrap[:start]
                     + "print('CPU int8 explicitly selected; same large-v3-turbo model', flush=True)\n"
                     + bootstrap[end:])
    for worker in ((0,) if cpu else (0, 1)):
        notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
            "accelerator": "GPU", "colab": {"name": f"AIC2026 B02 ASR worker {worker}",
                                             "gpuType": "T4", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        }, "cells": [
            {"cell_type": "markdown", "id": "readme", "metadata": {}, "source": [
                f"# B02 ASR · Worker {worker}/2\n",
                "large-v3-turbo / CUDA int8_float16 / Vietnamese. Checkpoint per video.\n",
                "1. Select T4 GPU and mount Drive. 2. Run setup. 3. Run worker.\n",
                "Do not run a second copy with the same worker ID. No keyframe maps needed.\n",
                "Outputs: `AIC2026/B02/out/asr-video-only-v1`. This is staging, not imported.\n",
                "Existing checkpoints are verified; mismatches stop without overwriting.\n",
            ]},
            cell(f"WORKER_ID = {worker}\nfrom google.colab import drive\ndrive.mount('/content/drive')\n", "mount"),
            cell(f"SOURCE_SHA256 = {report['sha256']!r}\nSOURCE_BASE64 = {payload!r}\n" + bootstrap, "setup"),
            cell("subprocess.run(command, check=True)\n", "run"),
        ]}
        suffix = "_cpu" if cpu else ""
        if cpu:
            notebook["metadata"].pop("accelerator")
            notebook["metadata"]["colab"].pop("gpuType")
            notebook["metadata"]["colab"]["name"] = "AIC2026 B02 ASR CPU temporary"
            notebook["cells"][0]["source"] = [
                "# B02 ASR · CPU temporary worker\n",
                "Same large-v3-turbo, CPU int8, Vietnamese. Separate CPU staging receipts.\n",
                "Mount Drive, run setup, then run one-video measurement.\n",
                "Outputs: `AIC2026/B02/out/asr-video-only-cpu-v1`. Not imported.\n",
                "Do not start the GPU worker concurrently; audit before changing execution profile.\n",
            ]
            notebook["cells"][-1] = cell(
                "subprocess.run(command + ['--max-videos', '1'], check=True)\n",
                "sample",
            )
            notebook["cells"].append(cell("subprocess.run(command, check=True)\n", "continue"))
        path = destination / f"aic2026_b02_asr_worker_{worker}{suffix}.ipynb"
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

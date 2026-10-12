"""Three-cell B02 CPU ASR pilot for M/S only; no old receipt reuse."""

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
out = raw.parent / 'out' / 'asr-ms-cpu-seekable-20260925-v2'
repo = Path('/content/aic2026-b02-asr-ms')
assert raw.is_dir(), 'Mount Drive containing AIC2026/B02/raw'
payload = base64.b64decode(SOURCE_BASE64)
assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
repo.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as source:
    for member in source.infolist():
        target = (repo / member.filename).resolve()
        assert repo.resolve() in target.parents
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open(member) as src, target.open('wb') as dst:
            shutil.copyfileobj(src, dst)
os.chdir(repo)
os.environ['PYTHONUTF8'] = '1'
os.environ['UV_CACHE_DIR'] = '/content/uv-b02-asr-cache'
assert shutil.which('ffmpeg'), 'ffmpeg is required'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.11.16'], check=True)
subprocess.run(['uv', 'python', 'install', '3.11'], check=True)
subprocess.run(['uv', 'sync', '--python', '3.11', '--only-group', 'asr-colab', '--frozen'], check=True)
python = str(repo / '.venv/bin/python')
command = [python, 'scripts/colab_video_asr.py', '--raw-dir', str(raw),
    '--output-dir', str(out), '--batch-id', 'aic26-b2', '--worker-count', '1',
    '--worker-id', '0', '--b02-ms-manifest', 'b02-p0-manifest.json',
    '--source-sha256', SOURCE_SHA256, '--resume', '--device', 'cpu']
subprocess.run(command + ['--inventory-only'], check=True)
subprocess.run([python, 'scripts/prepare_models.py', '--only', 'asr'], check=True)
print('Ready: 316 M/S videos, 298 N videos EXCLUDED. large-v3-turbo CPU int8.')
print('Cell 3 measures ONE full video. Review audio and realtime factor before extending.')
'''


def build(destination: Path) -> dict:
    files = {p.relative_to(ROOT).as_posix(): p for p in (ROOT / "src").rglob("*.py")}
    for name in ("pyproject.toml", "uv.lock", "scripts/colab_video_asr.py",
                 "scripts/prepare_models.py", "config/settings.yaml",
                 "checksums/whisper-large-v3-turbo.sha256"):
        files[name] = ROOT / name
    files["b02-p0-manifest.json"] = ROOT / "outputs/checkpoints/b02-p0/b02-p0-manifest-v3.json"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            if path.is_symlink() or ROOT not in path.resolve().parents:
                raise ValueError("Unsafe bundle path")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    sources = ["from google.colab import drive\ndrive.mount('/content/drive')\n",
               f"SOURCE_SHA256 = {sha!r}\nSOURCE_BASE64 = {base64.b64encode(payload).decode()!r}\n" + SETUP,
               "# One complete video first. Preserve child output for diagnosis.\n"
               "pilot = subprocess.run(command + ['--max-videos', '1'],\n"
               "    capture_output=True, text=True)\n"
               "diagnostics = out / 'diagnostics'\n"
               "diagnostics.mkdir(parents=True, exist_ok=True)\n"
               "log = diagnostics / 'worker-0-pilot.log'\n"
               "log.write_text('STDOUT\\n' + pilot.stdout + '\\nSTDERR\\n' + pilot.stderr, encoding='utf-8')\n"
               "print(pilot.stdout[-20000:], flush=True)\n"
               "print(pilot.stderr[-20000:], flush=True)\n"
               "print('Diagnostic log:', log, flush=True)\n"
               "if pilot.returncode:\n"
               "    raise subprocess.CalledProcessError(pilot.returncode, pilot.args,\n"
               "                                       output=pilot.stdout, stderr=pilot.stderr)\n"]
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": "B02 ASR M-S CPU pilot", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
    }, "cells": [{"cell_type": "markdown", "id": "readme", "metadata": {}, "source": [
        "# B02 ASR · M/S only · CPU pilot\n",
        "Three cells: Mount Drive → Setup → One-video timing/audio check.\n",
        "large-v3-turbo, int8 CPU. 298 traffic N videos excluded by user request.\n",
        "New output: `B02/out/asr-ms-cpu-seekable-20260925-v2`. No old receipt reuse.\n",
        "No model downloads on your PC; no production index changes.\n",
    ]}]}
    for index, source in enumerate(sources):
        notebook["cells"].append({"cell_type": "code", "id": f"asr-{index + 1}",
                                  "metadata": {}, "source": source.splitlines(keepends=True),
                                  "execution_count": None, "outputs": []})
    destination.mkdir(parents=True, exist_ok=True)
    bundle = destination / f"b02-asr-ms-{sha[:12]}.zip"
    bundle.write_bytes(payload)
    path = destination / "aic2026_b02_asr_ms_cpu_diagnostic.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    report = {"source_sha256": sha, "source_bytes": len(payload), "notebook": str(path),
              "bundle": str(bundle),
              "scope": "M/S only", "state": "prepared_not_executed",
              "captures_child_output": True,
              "output_namespace": "asr-ms-cpu-seekable-20260925-v2"}
    (destination / "build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "outputs/colab/b02-asr-ms-cpu-20260925-v2"), indent=2))

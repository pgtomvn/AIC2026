"""Install a user-supplied FFmpeg ZIP into the project-local tool cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tooling.ffmpeg import install_ffmpeg_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checksum-verify and atomically install only ffmpeg/ffprobe."
    )
    parser.add_argument("--archive", type=Path, required=True, help="User-supplied FFmpeg ZIP")
    parser.add_argument("--sha256", required=True, help="Required full archive SHA-256")
    parser.add_argument(
        "--install-root",
        type=Path,
        default=ROOT / ".tools" / "ffmpeg",
        help="Destination below the project .tools directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tools_root = (ROOT / ".tools").resolve()
    install_root = args.install_root.resolve()
    if not install_root.is_relative_to(tools_root):
        raise SystemExit(f"--install-root must remain below {tools_root}")
    installation = install_ffmpeg_archive(
        args.archive,
        args.sha256,
        install_root=install_root,
    )
    print(
        json.dumps(
            {
                "version": installation.version,
                "archive_sha256": installation.archive_sha256,
                "ffmpeg": str(installation.ffmpeg),
                "ffprobe": str(installation.ffprobe),
                "manifest": str(installation.active_manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

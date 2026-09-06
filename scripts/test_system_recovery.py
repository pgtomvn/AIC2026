"""Verify malformed KIS input does not degrade the next request in the same API process."""

from __future__ import annotations

import argparse
import json
import time

import requests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--batch-id", default="aic25-b1-reference")
    args = parser.parse_args(argv)
    api_url = str(args.api_url).rstrip("/")
    session = requests.Session()
    cases = (
        ("malformed", "\x00\u202e\x01", 422),
        ("recovery", "một con mèo", 200),
    )
    outcomes = []
    for name, text, expected in cases:
        started = time.perf_counter()
        try:
            response = session.post(
                f"{api_url}/v1/kis/search",
                json={
                    "query_id": f"system-recovery-{name}",
                    "original_text": text,
                    "batch_filters": [args.batch_id],
                    "limit": 5,
                    "refine_raw": False,
                },
                timeout=(3, 30),
            )
            payload = response.json()
            hits = payload.get("hits", []) if isinstance(payload, dict) else []
            ok = response.status_code == expected and (name == "malformed" or bool(hits))
            outcomes.append(
                {
                    "case": name,
                    "ok": ok,
                    "status_code": response.status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
                    "hit_count": len(hits) if isinstance(hits, list) else 0,
                }
            )
        except requests.RequestException:
            outcomes.append({"case": name, "ok": False, "code": "api_unavailable"})
            break
    report = {"ok": len(outcomes) == 2 and all(item["ok"] for item in outcomes), "outcomes": outcomes}
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

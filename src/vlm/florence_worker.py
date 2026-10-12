"""One-shot Florence inference, isolated from the long-lived retrieval API."""

from __future__ import annotations

import base64
import contextlib
import json
import sys

from src.contracts import QACompetitionSpec, QAQueryRequest
from src.vlm.florence2 import Florence2TaskReasoner
from src.vlm.qa_reasoner import QAImage, QAReasonerError


def main() -> None:
    """Consume one local request and emit only a result or sanitized failure."""

    try:
        payload = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            reasoner = Florence2TaskReasoner(**payload["config"])
            image = payload["image"]
            result = reasoner.reason_task(
                QAQueryRequest.from_dict(payload["request"]),
                (QAImage(
                    data=base64.b64decode(image["data"], validate=True),
                    mime_type=image["mime_type"], index=image["index"],
                    source_frame_idx=image["source_frame_idx"],
                    timestamp_ms=image["timestamp_ms"], sha256=image["sha256"],
                ),),
                QACompetitionSpec.from_dict(payload["spec"]),
                task=payload["task"],
                target_label=payload.get("target_label"),
            )
        response = {"result": result.to_dict(), "metrics": reasoner.last_metrics}
    except QAReasonerError as error:
        response = {"error": error.safe_metadata()}
    except Exception:
        response = {"error": {"category": "provider_request_failed"}}
    print(json.dumps(response, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

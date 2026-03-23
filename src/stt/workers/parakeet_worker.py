#!/usr/bin/env python3
"""
Parakeet MLX worker process for STT.

Runs Parakeet transcription in a separate process so the main app stays responsive
and can recover by restarting the worker if Parakeet hangs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any


def _write_json(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="STT Parakeet MLX worker")
    parser.add_argument("--model", required=True, help="HF repo for Parakeet model")
    args = parser.parse_args()

    model_name = args.model

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from parakeet_mlx import from_pretrained

        _log(f"[stt:parakeet-worker] Loading model: {model_name}")
        model = from_pretrained(model_name)
        _write_json({"type": "ready"})
        _log("[stt:parakeet-worker] Ready")
    except Exception as e:
        _write_json({"type": "error", "error": f"Failed to initialize Parakeet: {e}"})
        _log(traceback.format_exc())
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except Exception:
            _log(f"[stt:parakeet-worker] Non-JSON input ignored: {line!r}")
            continue

        msg_type = message.get("type")
        if msg_type == "shutdown":
            _write_json({"type": "shutdown_ack"})
            return 0

        if msg_type != "transcribe":
            _log(f"[stt:parakeet-worker] Unknown message type: {msg_type!r}")
            continue

        req_id = message.get("id")
        audio_file_path = message.get("audio_file_path")

        try:
            result = model.transcribe(audio_file_path)
            text = (result.text or "").strip()
            _write_json({"type": "result", "id": req_id, "text": text, "error": None})
        except Exception as e:
            _log(traceback.format_exc())
            _write_json({"type": "result", "id": req_id, "text": "", "error": str(e)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

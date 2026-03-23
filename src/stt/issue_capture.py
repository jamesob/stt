from __future__ import annotations

import json
import os
import shutil
import time


def maybe_capture_mlx_issue(
    *,
    provider,
    wav_path: str,
    language: str,
    prompt: str,
) -> bool:
    """Persist MLX provider errors + audio to .bugs/ for later debugging.

    Returns True if the wav was moved/copied into the issue directory.
    """
    last_error = getattr(provider, "_last_error", None)
    last_trace = getattr(provider, "_last_error_trace", None)
    if not last_error and not last_trace:
        return False

    base_dir = os.path.join(os.path.dirname(__file__), ".bugs")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    issue_dir = os.path.join(base_dir, timestamp)
    os.makedirs(issue_dir, exist_ok=True)

    moved_path = os.path.join(issue_dir, os.path.basename(wav_path))
    moved = False
    try:
        shutil.move(wav_path, moved_path)
        moved = True
    except Exception:
        try:
            shutil.copy2(wav_path, moved_path)
            moved = True
        except Exception:
            moved_path = wav_path

    issue_log_path = os.path.join(issue_dir, "issue.log")
    metadata = {
        "timestamp": timestamp,
        "provider": getattr(provider, "name", None) or type(provider).__name__,
        "model": getattr(provider, "model", None),
        "language": language,
        "prompt": prompt,
        "audio_file": moved_path,
        "error": last_error,
        "traceback": last_trace,
    }
    try:
        with open(issue_log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata, indent=2, ensure_ascii=True))
    except Exception:
        pass

    print(f"⚠️  Saved MLX issue to {issue_dir}")
    return moved


from __future__ import annotations

import os
import shutil
import time


DEFAULT_RECORDINGS_DIR = os.path.expanduser("~/.stt-recordings")
DEFAULT_RECORDINGS_MAX_BYTES = 1024 * 1024 * 1024  # 1GB


def enforce_recordings_limit(*, recordings_dir: str, recordings_max_bytes: int) -> None:
    """Delete oldest recordings if total size exceeds recordings_max_bytes."""
    if not os.path.isdir(recordings_dir):
        return

    files: list[tuple[str, float, int]] = []
    total_size = 0
    for name in os.listdir(recordings_dir):
        if not name.endswith(".wav"):
            continue
        path = os.path.join(recordings_dir, name)
        try:
            stat = os.stat(path)
            files.append((path, stat.st_mtime, stat.st_size))
            total_size += stat.st_size
        except OSError:
            continue

    if total_size <= recordings_max_bytes:
        return

    files.sort(key=lambda x: x[1])  # oldest first
    for path, _, size in files:
        if total_size <= recordings_max_bytes:
            break
        try:
            os.unlink(path)
            total_size -= size
            txt_path = path.rsplit(".", 1)[0] + ".txt"
            try:
                os.unlink(txt_path)
            except OSError:
                pass
        except OSError:
            pass


def archive_recording(
    wav_path: str,
    *,
    keep_recordings: bool,
    recordings_dir: str,
    recordings_max_bytes: int,
    text: str | None = None,
) -> str | None:
    """Move recording to archive dir; returns new path or None on failure."""
    if not keep_recordings:
        return None

    os.makedirs(recordings_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(recordings_dir, f"{timestamp}.wav")

    try:
        shutil.move(wav_path, dest)
    except Exception:
        try:
            shutil.copy2(wav_path, dest)
        except Exception:
            return None

    if text:
        txt_path = os.path.join(recordings_dir, f"{timestamp}.txt")
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    enforce_recordings_limit(recordings_dir=recordings_dir, recordings_max_bytes=recordings_max_bytes)
    return dest


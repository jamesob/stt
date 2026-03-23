#!/usr/bin/env python3
"""
Bruteforce audio recording start/stop/cancel to surface stuck states.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time

os.environ.setdefault("STT_HEADLESS", "1")

from stt.app import STTApp


class _NullOverlay:
    def show(self):
        pass

    def hide(self):
        pass

    def update_waveform(self, values, above_threshold=False):
        pass

    def set_transcribing(self, transcribing: bool):
        pass

    def set_shift_held(self, held: bool):
        pass


class _DummyProvider:
    name = "dummy"

    def is_available(self):
        return True

    def warmup(self):
        return None

    def transcribe(self, audio_file_path: str, language: str, prompt: str | None = None):
        return ""


def _safe_unlink(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress test STT recording engine")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--min-hold", type=float, default=0.02)
    parser.add_argument("--max-hold", type=float, default=0.5)
    parser.add_argument("--min-gap", type=float, default=0.01)
    parser.add_argument("--max-gap", type=float, default=0.2)
    parser.add_argument("--mode", choices=["stop", "cancel", "mixed", "full"], default="mixed")
    parser.add_argument("--cancel-rate", type=float, default=0.2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--async-start", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    app = STTApp(
        device_name=args.device,
        provider=_DummyProvider(),
        overlay=_NullOverlay(),
        sound_player=lambda _: None,
        text_injector=lambda *_: None,
        keep_recordings=False,
    )

    if args.mode == "full":
        app.type_text = lambda *_, **__: None
        app.print_ready_prompt = lambda: None

    stats = {
        "iterations": 0,
        "stops": 0,
        "cancels": 0,
        "errors": 0,
        "empty_wav": 0,
    }

    try:
        for i in range(args.iterations):
            stats["iterations"] += 1

            def _start():
                app.start_recording()

            if args.async_start:
                threading.Thread(target=_start, daemon=True).start()
            else:
                _start()

            time.sleep(random.uniform(args.min_hold, args.max_hold))

            action = args.mode
            if args.mode == "mixed":
                action = "cancel" if random.random() < args.cancel_rate else "stop"

            try:
                if action == "cancel":
                    app.cancel_recording()
                    stats["cancels"] += 1
                elif action == "full":
                    app.process_recording(send_enter=False)
                    stats["stops"] += 1
                else:
                    wav_path, frames, peak = app.stop_recording()
                    if not wav_path:
                        stats["empty_wav"] += 1
                    _safe_unlink(wav_path)
                    stats["stops"] += 1
            except Exception:
                stats["errors"] += 1

            time.sleep(random.uniform(args.min_gap, args.max_gap))
    finally:
        try:
            app._audio_worker.stop(force=True)
        except Exception:
            pass

    print("stress_done", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

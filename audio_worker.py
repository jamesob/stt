#!/usr/bin/env python3
"""
Audio recording worker process for STT.

Runs PortAudio/sounddevice recording in a separate process so any rare driver
deadlocks during stop/close can't freeze the main UI.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from typing import Any


_STDOUT_LOCK = threading.Lock()


def _write_json(message: dict[str, Any]) -> None:
    data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with _STDOUT_LOCK:
            os.write(sys.stdout.fileno(), data)
    except BrokenPipeError:
        os._exit(0)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


WAVEFORM_BARS = 20
WAVEFORM_INTERVAL_S = 0.033  # ~30fps
PEAK_WINDOW_SIZE = 90  # ~3 seconds rolling window for peak normalization
WAVEFORM_MAX_CHUNKS = 64  # cap to avoid unbounded growth if reader stalls
START_TIMEOUT_S = 2.0


class Recorder:
    def __init__(self):
        self._recording = False
        self._stream = None
        self._chunks = []
        self._sample_rate = None
        self._channels = None
        self._waveform_buffer = []
        self._peak_level = 0.01  # Auto-normalizing peak (starts low)
        self._peak_history = []  # Rolling window for percentile-based normalization
        self._waveform_lock = threading.Lock()
        self._waveform_thread = None
        self._waveform_stop = None
        self._force_exit = False
        self._close_thread = None

    def start(self, *, device_name: str | None, sample_rate: int, channels: int) -> None:
        if self._recording:
            raise RuntimeError("Already recording")

        # Wait for any pending stream close from a previous recording.
        self._await_pending_close()
        if self._force_exit:
            raise RuntimeError("Audio driver stuck; worker restart required")

        import numpy as np
        import sounddevice as sd

        # Resolve device name to index at recording time (handles plug/unplug)
        device_index = None
        if device_name:
            device_index = self._resolve_device(device_name, sd)
            if device_index is None:
                raise RuntimeError(f"Audio device '{device_name}' not found")

        self._chunks = []
        self._sample_rate = sample_rate
        self._channels = channels
        self._recording = True
        self._waveform_buffer = []
        self._peak_level = 0.01  # Reset peak for new recording
        self._peak_history = []  # Reset rolling window

        def callback(indata, frames, time_info, status):
            if status:
                _log(f"[stt:audio-worker] Status: {status}")
            if self._recording:
                self._chunks.append(indata.copy())
                with self._waveform_lock:
                    self._waveform_buffer.append(indata.copy())
                    if len(self._waveform_buffer) > WAVEFORM_MAX_CHUNKS:
                        self._waveform_buffer = self._waveform_buffer[-WAVEFORM_MAX_CHUNKS // 2:]

        start_state = {"stream": None, "error": None}

        def _do_start():
            try:
                stream = sd.InputStream(
                    device=device_index,
                    samplerate=sample_rate,
                    channels=channels,
                    dtype=np.float32,
                    callback=callback,
                )
                stream.start()
                start_state["stream"] = stream
            except Exception as e:
                start_state["error"] = e

        thread = threading.Thread(target=_do_start, daemon=True)
        thread.start()
        thread.join(timeout=START_TIMEOUT_S)
        if thread.is_alive():
            self._recording = False
            self._force_exit = True
            raise TimeoutError("Timed out starting audio stream")

        if start_state["error"] is not None:
            self._recording = False
            raise start_state["error"]

        stream = start_state["stream"]
        if stream is None:
            self._recording = False
            raise RuntimeError("Audio stream failed to start")

        self._stream = stream
        self._start_waveform_thread()

    def _resolve_device(self, name: str, sd) -> int | None:
        """Resolve device name to current index."""
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0 and dev['name'] == name:
                return i
        return None

    def _start_waveform_thread(self) -> None:
        if self._waveform_thread and self._waveform_thread.is_alive():
            return
        self._waveform_stop = threading.Event()
        self._waveform_thread = threading.Thread(target=self._waveform_loop, daemon=True)
        self._waveform_thread.start()

    def _stop_waveform_thread(self) -> None:
        if self._waveform_stop:
            self._waveform_stop.set()
        if self._waveform_thread:
            self._waveform_thread.join(timeout=0.5)
            if self._waveform_thread.is_alive():
                return
        self._waveform_thread = None
        self._waveform_stop = None

    def _waveform_loop(self) -> None:
        import time
        import numpy as np

        next_tick = time.time()
        while self._waveform_stop and not self._waveform_stop.is_set():
            now = time.time()
            delay = next_tick - now
            if delay > 0:
                self._waveform_stop.wait(timeout=delay)
                continue
            next_tick = now + WAVEFORM_INTERVAL_S

            with self._waveform_lock:
                if not self._waveform_buffer:
                    continue
                buffers = self._waveform_buffer
                self._waveform_buffer = []

            try:
                self._send_waveform(np, buffers)
            except Exception:
                _log(traceback.format_exc())

    def _send_waveform(self, np, buffers) -> None:
        """Calculate and send waveform data with auto-normalization"""
        if not buffers:
            return

        # Concatenate recent audio
        audio = np.concatenate(buffers, axis=0)

        # Take absolute values and flatten to mono
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = np.abs(audio)

        # Downsample to WAVEFORM_BARS values
        samples_per_bar = len(audio) // WAVEFORM_BARS
        if samples_per_bar < 1:
            samples_per_bar = 1

        raw_values = []
        for i in range(WAVEFORM_BARS):
            start = i * samples_per_bar
            end = start + samples_per_bar
            if end > len(audio):
                end = len(audio)
            if start < len(audio):
                # Use RMS for smoother visualization
                chunk = audio[start:end]
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                raw_values.append(rms)
            else:
                raw_values.append(0.0)

        # Auto-normalize using rolling window percentile (ignores transient spikes)
        current_max = max(raw_values) if raw_values else 0
        self._peak_history.append(current_max)
        if len(self._peak_history) > PEAK_WINDOW_SIZE:
            self._peak_history.pop(0)

        # Use 85th percentile with EMA smoothing for stable normalization
        target_peak = float(np.percentile(self._peak_history, 85))
        target_peak = max(target_peak, 0.005)  # floor to prevent /0
        self._peak_level = self._peak_level * 0.8 + target_peak * 0.2  # smooth transitions

        # Normalize values to 0-1 range based on peak
        values = [min(1.0, v / self._peak_level * 0.85) for v in raw_values]

        _write_json({"type": "waveform", "values": values, "raw_peak": current_max})

    def stop(self, *, wav_path: str) -> tuple[int, float]:
        if not self._recording:
            return 0, 0.0

        self._recording = False
        self._stop_waveform_thread()
        with self._waveform_lock:
            self._waveform_buffer = []
        stream = self._stream
        self._stream = None
        chunks = self._chunks
        self._chunks = []

        if not chunks:
            if stream is not None:
                self._close_stream(stream)
            return 0, 0.0

        import numpy as np
        from scipy.io import wavfile

        audio = np.concatenate(chunks, axis=0)
        frames = int(audio.shape[0])
        peak = float(np.max(np.abs(audio)))

        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(wav_path, int(self._sample_rate or 16000), audio_int16)

        # Close stream *after* saving wav so the data capture isn't delayed.
        if stream is not None:
            self._close_stream(stream)

        return frames, peak

    def cancel(self) -> None:
        self._recording = False
        self._chunks = []
        self._stop_waveform_thread()
        with self._waveform_lock:
            self._waveform_buffer = []

        stream = self._stream
        self._stream = None
        if stream is not None:
            self._close_stream(stream)

    def shutdown(self) -> None:
        self.cancel()
        self._await_pending_close()

    def _close_stream(self, stream) -> None:
        """Close a stream, waiting up to 2 s inline.

        If the close is still running after 2 s the thread reference is
        saved so that ``_await_pending_close`` (called from ``start()``
        and ``shutdown()``) can give it a little more time before
        declaring the driver stuck.
        """
        # Finish any earlier close first.
        self._await_pending_close()

        def _close():
            try:
                stream.abort(ignore_errors=True)
                stream.close(ignore_errors=True)
            except Exception:
                _log(traceback.format_exc())

        thread = threading.Thread(target=_close, daemon=True)
        thread.start()
        thread.join(timeout=2.0)
        if thread.is_alive():
            # Not done yet – park it for start()/shutdown() to check.
            self._close_thread = thread

    def _await_pending_close(self) -> None:
        """Wait for a pending stream close to finish.  Sets *_force_exit* on timeout."""
        thread = self._close_thread
        if thread is None:
            return
        self._close_thread = None
        if not thread.is_alive():
            return
        # Already waited 2 s in _close_stream; give 1 more second.
        thread.join(timeout=1.0)
        if thread.is_alive():
            _log("[stt:audio-worker] Stream close stuck; forcing worker restart")
            self._force_exit = True

    def should_exit(self) -> bool:
        return self._force_exit


def main() -> int:
    recorder = Recorder()
    _write_json({"type": "ready"})

    parent_pid = os.environ.get("STT_PARENT_PID")
    if parent_pid:
        try:
            parent_pid_int = int(parent_pid)
        except ValueError:
            parent_pid_int = None
        if parent_pid_int:
            def _watch_parent() -> None:
                while True:
                    time.sleep(2)
                    try:
                        os.kill(parent_pid_int, 0)
                    except Exception:
                        _log("[stt:audio-worker] Parent gone; exiting")
                        try:
                            recorder.shutdown()
                        except Exception:
                            pass
                        os._exit(0)
            threading.Thread(target=_watch_parent, daemon=True).start()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except Exception:
            _log(f"[stt:audio-worker] Non-JSON input ignored: {line!r}")
            continue

        msg_type = message.get("type")
        req_id = message.get("id")

        try:
            if msg_type == "shutdown":
                recorder.shutdown()
                _write_json({"type": "shutdown_ack"})
                return 0

            if msg_type == "start":
                recorder.start(
                    device_name=message.get("device_name"),
                    sample_rate=int(message.get("sample_rate") or 16000),
                    channels=int(message.get("channels") or 1),
                )
                _write_json({"type": "started", "id": req_id})
                if recorder.should_exit():
                    return 0
                continue

            if msg_type == "stop":
                wav_path = message.get("wav_path")
                if not wav_path:
                    raise ValueError("Missing wav_path")
                frames, peak = recorder.stop(wav_path=str(wav_path))
                _write_json({"type": "stopped", "id": req_id, "wav_path": wav_path, "frames": frames, "peak": peak})
                if recorder.should_exit():
                    return 0
                continue

            if msg_type == "cancel":
                recorder.cancel()
                _write_json({"type": "canceled", "id": req_id})
                if recorder.should_exit():
                    return 0
                continue

            _log(f"[stt:audio-worker] Unknown message type: {msg_type!r}")
        except Exception as e:
            _log(traceback.format_exc())
            _write_json({"type": "error", "id": req_id, "error": str(e)})
            if recorder.should_exit():
                return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

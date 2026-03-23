#!/usr/bin/env python3
"""Test harness for overlay design iteration"""

import sys
import math
import time
from AppKit import NSApplication, NSApp, NSApplicationActivationPolicyAccessory
from Foundation import NSTimer, NSRunLoop, NSDefaultRunLoopMode

# Initialize app first
app = NSApplication.sharedApplication()
app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

from stt.ui.overlay import get_overlay, BAR_COUNT
from stt.ui.prompt_overlay import PromptOverlay

class OverlayHarness:
    def __init__(self):
        self.recording_overlay = get_overlay()
        self.prompt_overlay = PromptOverlay(on_select=self._on_prompt_select)
        self.phase = 0.0
        self.mode = "recording"  # recording, transcribing, prompt
        self.timer = None

    def _on_prompt_select(self, text, enter):
        print(f"Selected: {text} (enter={enter})")

    def start(self):
        print("Overlay Test Harness")
        print("=" * 40)
        print("Commands (type and press Enter):")
        print("  r - Show recording overlay (waveform)")
        print("  t - Switch to transcribing animation")
        print("  s - Toggle shift-held indicator")
        print("  p - Show prompt overlay")
        print("  h - Hide all overlays")
        print("  q - Quit")
        print("=" * 40)

        # Start with recording overlay
        self._show_recording()

        # Start animation timer
        self.timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            0.05, True, lambda _: self._animate()
        )

        # Run with stdin checking
        import select
        import threading

        def input_thread():
            while True:
                try:
                    line = sys.stdin.readline().strip().lower()
                    if line == 'q':
                        NSApp.terminate_(None)
                        break
                    elif line == 'r':
                        self._show_recording()
                    elif line == 't':
                        self._show_transcribing()
                    elif line == 's':
                        self._toggle_shift()
                    elif line == 'p':
                        self._show_prompt()
                    elif line == 'h':
                        self._hide_all()
                except:
                    break

        t = threading.Thread(target=input_thread, daemon=True)
        t.start()

        NSApp.run()

    def _show_recording(self):
        self.prompt_overlay.hide()
        self.recording_overlay.show()
        self.recording_overlay.set_transcribing(False)
        self.mode = "recording"
        print("→ Recording mode (waveform)")

    def _show_transcribing(self):
        self.prompt_overlay.hide()
        self.recording_overlay.show()
        self.recording_overlay.set_transcribing(True)
        self.mode = "transcribing"
        print("→ Transcribing mode (animated)")

    def _toggle_shift(self):
        if hasattr(self, '_shift_held'):
            self._shift_held = not self._shift_held
        else:
            self._shift_held = True
        self.recording_overlay.set_shift_held(self._shift_held)
        print(f"→ Shift held: {self._shift_held}")

    def _show_prompt(self):
        self.recording_overlay.hide()
        self.prompt_overlay.show()
        self.mode = "prompt"
        print("→ Prompt overlay")

    def _hide_all(self):
        self.recording_overlay.hide()
        self.prompt_overlay.hide()
        self.mode = "hidden"
        print("→ All hidden")

    def _animate(self):
        if self.mode == "recording":
            # Simulate waveform data
            self.phase += 0.15
            values = []
            for i in range(BAR_COUNT):
                # Mix of waves for organic look
                v = (math.sin(self.phase + i * 0.3) * 0.3 +
                     math.sin(self.phase * 1.7 + i * 0.5) * 0.2 +
                     math.sin(self.phase * 0.5 + i * 0.2) * 0.2 + 0.3)
                values.append(max(0, min(1, v)))
            self.recording_overlay.update_waveform(values, above_threshold=True)


if __name__ == "__main__":
    harness = OverlayHarness()
    harness.start()

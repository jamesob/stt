from __future__ import annotations

import subprocess
import time
from typing import Literal


PasteMethod = Literal["osascript", "cgevent"]


def paste_text(text: str, *, send_enter: bool = False, method: PasteMethod = "osascript") -> None:
    """Paste text into the active app using the clipboard, then optionally send Enter."""
    if not text:
        return

    old_clipboard = subprocess.run(["pbpaste"], capture_output=True, timeout=2).stdout
    try:
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True, timeout=2)

        if method == "osascript":
            paste_script = """
            tell application "System Events"
                keystroke "v" using command down
            end tell
            """
            subprocess.run(["osascript", "-e", paste_script], check=True, timeout=5)
            if send_enter:
                enter_script = """
                tell application "System Events"
                    key code 36
                end tell
                """
                subprocess.run(["osascript", "-e", enter_script], check=True, timeout=5)
        elif method == "cgevent":
            from Quartz import (
                CGEventCreateKeyboardEvent,
                CGEventPost,
                CGEventSetFlags,
                kCGHIDEventTap,
                kCGEventFlagMaskCommand,
            )

            # Key code 9 = 'v' on US keyboard.
            v_keycode = 9

            event_down = CGEventCreateKeyboardEvent(None, v_keycode, True)
            CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
            CGEventPost(kCGHIDEventTap, event_down)

            event_up = CGEventCreateKeyboardEvent(None, v_keycode, False)
            CGEventSetFlags(event_up, kCGEventFlagMaskCommand)
            CGEventPost(kCGHIDEventTap, event_up)

            if send_enter:
                time.sleep(0.05)
                enter_down = CGEventCreateKeyboardEvent(None, 36, True)
                CGEventSetFlags(enter_down, 0)
                CGEventPost(kCGHIDEventTap, enter_down)
                enter_up = CGEventCreateKeyboardEvent(None, 36, False)
                CGEventSetFlags(enter_up, 0)
                CGEventPost(kCGHIDEventTap, enter_up)
        else:
            raise ValueError(f"Unknown paste method: {method}")

        time.sleep(0.05)
    finally:
        subprocess.run(["pbcopy"], input=old_clipboard, check=True, timeout=2)


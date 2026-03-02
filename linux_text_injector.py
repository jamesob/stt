"""Text injection for Wayland Linux using wl-copy + wtype."""

from __future__ import annotations

import subprocess


def paste_text(
    text: str, *, send_enter: bool = False
) -> None:
    """Paste text into the active Wayland app via clipboard + wtype."""
    if not text:
        return

    # Save current clipboard
    old_clipboard: bytes | None = None
    try:
        result = subprocess.run(
            ["wl-paste", "--no-newline"],
            capture_output=True,
            timeout=2,
        )
        if result.returncode == 0:
            old_clipboard = result.stdout
    except Exception:
        pass

    try:
        # Copy text to clipboard
        subprocess.run(
            ["wl-copy", "--", text],
            check=True,
            timeout=2,
        )

        # Simulate Ctrl+V
        subprocess.run(
            ["wtype", "-M", "ctrl", "-P", "v", "-p", "v", "-m", "ctrl"],
            check=True,
            timeout=2,
        )

        if send_enter:
            subprocess.run(
                ["wtype", "-P", "Return", "-p", "Return"],
                check=True,
                timeout=2,
            )
    finally:
        # Restore old clipboard
        if old_clipboard is not None:
            try:
                subprocess.run(
                    ["wl-copy", "--"],
                    input=old_clipboard,
                    check=True,
                    timeout=2,
                )
            except Exception:
                pass

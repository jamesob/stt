"""Text injection for Wayland Linux using wtype."""

from __future__ import annotations

import subprocess


def paste_text(
    text: str, *, send_enter: bool = False
) -> None:
    """Type text into the active Wayland app via wtype."""
    if not text:
        return

    # wtype directly types text into the focused window.
    # This works in both GUI apps and terminals (unlike Ctrl+V
    # clipboard paste, which terminals bind to Ctrl+Shift+V).
    cmd = ["wtype", "--", text]
    subprocess.run(cmd, check=True, timeout=5)

    if send_enter:
        subprocess.run(
            ["wtype", "-P", "Return", "-p", "Return"],
            check=True,
            timeout=2,
        )

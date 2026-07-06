"""Modifier key constants for Linux input handling.

On macOS, pynput.keyboard.Key is used directly. On Linux/Wayland,
pynput requires X11 at import time, so we define equivalent
constants here. The evdev listener maps to these, keeping the
controller code platform-agnostic.
"""

from enum import Enum


class Key(Enum):
    cmd = "cmd"
    cmd_l = "cmd_l"
    cmd_r = "cmd_r"
    alt_l = "alt_l"
    alt_r = "alt_r"
    ctrl_l = "ctrl_l"
    ctrl_r = "ctrl_r"
    shift_l = "shift_l"
    shift_r = "shift_r"
    esc = "esc"

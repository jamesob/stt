from __future__ import annotations

import sys

IS_LINUX = sys.platform == "linux"
IS_MACOS = sys.platform == "darwin"

HOTKEY_DISPLAY_NAMES: dict[str, str] = {
    "cmd_r": "Right ⌘",
    "cmd_l": "Left ⌘",
    "alt_r": "Right ⌥",
    "alt_l": "Left ⌥",
    "ctrl_r": "Right ⌃",
    "ctrl_l": "Left ⌃",
    "shift_r": "Right ⇧",
    "shift_l": "Left ⇧",
}

HOTKEY_DISPLAY_NAMES_LINUX: dict[str, str] = {
    "cmd_r": "Right Super",
    "cmd_l": "Left Super",
    "alt_r": "Right Alt",
    "alt_l": "Left Alt",
    "ctrl_r": "Right Ctrl",
    "ctrl_l": "Left Ctrl",
    "shift_r": "Right Shift",
    "shift_l": "Left Shift",
}


def get_hotkey_display_name(hotkey_id: str) -> str:
    if IS_LINUX:
        return HOTKEY_DISPLAY_NAMES_LINUX.get(hotkey_id, hotkey_id)
    return HOTKEY_DISPLAY_NAMES.get(hotkey_id, hotkey_id)


class NullOverlay:
    def show(self):
        pass

    def hide(self):
        pass

    def update_waveform(self, values, above_threshold: bool = False):
        pass

    def set_transcribing(self, transcribing: bool):
        pass

    def set_shift_held(self, held: bool):
        pass


def noop_text_injector(_text: str, _send_enter: bool = False) -> None:
    return


def noop_sound(_path: str) -> None:
    return


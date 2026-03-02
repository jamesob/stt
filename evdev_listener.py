"""evdev-based keyboard listener for Wayland Linux.

Monitors /dev/input/event* devices for keyboard events and translates
them to pynput Key objects so the existing InputController handlers
work unchanged.
"""

from __future__ import annotations

import os
import select
import threading
import time
from typing import Callable, Optional

import evdev
from evdev import InputDevice, categorize, ecodes
from pynput.keyboard import Key


# evdev keycode -> pynput Key mapping (modifiers only)
_EVDEV_TO_PYNPUT: dict[int, Key] = {
    ecodes.KEY_LEFTSHIFT: Key.shift_l,
    ecodes.KEY_RIGHTSHIFT: Key.shift_r,
    ecodes.KEY_LEFTCTRL: Key.ctrl_l,
    ecodes.KEY_RIGHTCTRL: Key.ctrl_r,
    ecodes.KEY_LEFTALT: Key.alt_l,
    ecodes.KEY_RIGHTALT: Key.alt_r,
    ecodes.KEY_LEFTMETA: Key.cmd_l,
    ecodes.KEY_RIGHTMETA: Key.cmd_r,
    ecodes.KEY_ESC: Key.esc,
}

# Capabilities a device must have to count as a keyboard
_REQUIRED_KEYS = {ecodes.KEY_A, ecodes.KEY_Z, ecodes.KEY_SPACE, ecodes.KEY_ENTER}

_RESCAN_INTERVAL_S = 3.0


def _is_keyboard(device: InputDevice) -> bool:
    caps = device.capabilities().get(ecodes.EV_KEY, [])
    return _REQUIRED_KEYS.issubset(caps)


def _find_keyboards() -> list[InputDevice]:
    devices: list[InputDevice] = []
    for fn in sorted(evdev.list_devices()):
        try:
            dev = InputDevice(fn)
            if _is_keyboard(dev):
                devices.append(dev)
        except (PermissionError, OSError):
            continue
    return devices


class EvdevKeyboardListener:
    """Keyboard listener using evdev (for Wayland Linux).

    Provides the same start()/stop() interface as pynput.keyboard.Listener
    and delivers pynput Key objects to the on_press/on_release callbacks.
    """

    def __init__(
        self,
        on_press: Optional[Callable] = None,
        on_release: Optional[Callable] = None,
    ):
        self._on_press = on_press
        self._on_release = on_release
        self._devices: dict[str, InputDevice] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Wake pipe: writing a byte unblocks select()
        self._wake_r, self._wake_w = os.pipe()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="evdev-listener"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Wake select() so it can exit
        try:
            os.write(self._wake_w, b"\x00")
        except OSError:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._close_devices()
        for fd in (self._wake_r, self._wake_w):
            try:
                os.close(fd)
            except OSError:
                pass

    def _close_devices(self) -> None:
        for dev in self._devices.values():
            try:
                dev.close()
            except Exception:
                pass
        self._devices.clear()

    def _scan_devices(self) -> None:
        """Discover keyboards, adding any new ones."""
        for dev in _find_keyboards():
            if dev.path not in self._devices:
                self._devices[dev.path] = dev
                print(f"evdev: monitoring {dev.path} ({dev.name})")

    def _run(self) -> None:
        self._scan_devices()
        if not self._devices:
            print(
                "evdev: no keyboards found. "
                "Check that your user is in the 'input' group."
            )

        last_scan = time.monotonic()

        while not self._stop_event.is_set():
            # Periodic rescan for hotplugged devices
            now = time.monotonic()
            if now - last_scan >= _RESCAN_INTERVAL_S:
                self._scan_devices()
                last_scan = now

            fds = {
                dev.fd: dev for dev in self._devices.values()
            }
            if not fds:
                # No devices yet -- sleep briefly and retry
                time.sleep(0.5)
                last_scan = 0  # force rescan
                continue

            try:
                readable, _, _ = select.select(
                    list(fds.keys()) + [self._wake_r],
                    [], [],
                    _RESCAN_INTERVAL_S,
                )
            except (ValueError, OSError):
                break

            for fd in readable:
                if fd == self._wake_r:
                    # Drain wake pipe
                    try:
                        os.read(self._wake_r, 64)
                    except OSError:
                        pass
                    continue

                dev = fds.get(fd)
                if dev is None:
                    continue

                try:
                    for event in dev.read():
                        if event.type != ecodes.EV_KEY:
                            continue
                        key = _EVDEV_TO_PYNPUT.get(event.code)
                        if key is None:
                            continue
                        if event.value == 1:  # key down
                            if self._on_press:
                                self._on_press(key)
                        elif event.value == 0:  # key up
                            if self._on_release:
                                self._on_release(key)
                        # value == 2 is autorepeat -- ignore
                except OSError:
                    # Device disconnected
                    print(
                        f"evdev: lost device {dev.path}"
                    )
                    try:
                        dev.close()
                    except Exception:
                        pass
                    self._devices.pop(dev.path, None)

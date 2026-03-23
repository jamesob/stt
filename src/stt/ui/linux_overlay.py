"""Recording overlay for Wayland Linux using GTK4 + gtk4-layer-shell.

Same interface as overlay.RecordingOverlay: show(), hide(),
update_waveform(), set_transcribing(), set_shift_held().
Falls back to NullOverlay if GTK4 or layer-shell is unavailable.
"""

from __future__ import annotations

import math
import threading
from typing import Optional

from stt.defaults import NullOverlay

# Overlay dimensions (match macOS)
PILL_WIDTH = 280
PILL_HEIGHT = 60
BAR_COUNT = 20
BAR_WIDTH = 5
BAR_GAP = 4
BAR_MAX_HEIGHT = 40
BAR_MIN_HEIGHT = 4
CORNER_RADIUS = PILL_HEIGHT / 2
MIC_AREA_WIDTH = 52

# Preload gtk4-layer-shell before GTK/wayland to fix link order.
# See https://github.com/wmww/gtk4-layer-shell/blob/main/linking.md
_HAS_LAYER_SHELL = False
try:
    import ctypes
    ctypes.CDLL("libgtk4-layer-shell.so", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

try:
    import gi
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk, GLib, Gdk
    _HAS_GTK = True
except (ImportError, ValueError):
    _HAS_GTK = False

try:
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gtk4LayerShell
    _HAS_LAYER_SHELL = True
except Exception:
    pass


class LinuxOverlay:
    """GTK4 + layer-shell recording overlay for Wayland."""

    def __init__(self):
        self._visible = False
        self._lock = threading.Lock()
        self._waveform = [0.0] * BAR_COUNT
        self._smoothed = [0.0] * BAR_COUNT
        self._transcribing = False
        self._animation_phase = 0.0
        self._shift_held = False
        self._threshold_crossed = False
        self._animation_source: int = 0
        self._window: Optional[Gtk.Window] = None
        self._drawing_area: Optional[Gtk.DrawingArea] = None

        self._init_window()

    def _init_window(self) -> None:
        app = Gtk.Application.get_default()
        self._window = Gtk.Window()
        self._window.set_default_size(PILL_WIDTH, PILL_HEIGHT)
        self._window.set_decorated(False)
        self._window.set_resizable(False)

        if _HAS_LAYER_SHELL:
            Gtk4LayerShell.init_for_window(self._window)
            Gtk4LayerShell.set_layer(
                self._window,
                Gtk4LayerShell.Layer.OVERLAY,
            )
            Gtk4LayerShell.set_anchor(
                self._window,
                Gtk4LayerShell.Edge.BOTTOM, True,
            )
            Gtk4LayerShell.set_margin(
                self._window,
                Gtk4LayerShell.Edge.BOTTOM,
                int(Gdk.Display.get_default()
                    .get_monitors().get_item(0)
                    .get_geometry().height / 3),
            )
            Gtk4LayerShell.set_keyboard_mode(
                self._window,
                Gtk4LayerShell.KeyboardMode.NONE,
            )

        # Transparent background via CSS
        css = Gtk.CssProvider()
        css.load_from_string(
            "window { background-color: transparent; }"
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        self._drawing_area = Gtk.DrawingArea()
        self._drawing_area.set_content_width(PILL_WIDTH)
        self._drawing_area.set_content_height(PILL_HEIGHT)
        self._drawing_area.set_draw_func(self._draw)
        self._window.set_child(self._drawing_area)

    def _draw(self, area, cr, width, height) -> None:
        """Cairo draw callback for the pill + waveform."""
        # Pill background
        _rounded_rect(cr, 0, 0, width, height, CORNER_RADIUS)
        cr.set_source_rgba(0.05, 0.05, 0.05, 0.85)
        cr.fill()

        # Subtle border
        _rounded_rect(cr, 0.25, 0.25, width - 0.5, height - 0.5, CORNER_RADIUS - 0.25)
        cr.set_source_rgba(1, 1, 1, 0.08)
        cr.set_line_width(0.5)
        cr.stroke()

        # Mic icon (simple circle + rectangle representation)
        cx = MIC_AREA_WIDTH / 2 + 6
        cy = height / 2
        cr.set_source_rgba(1, 1, 1, 1)
        # Mic body
        _rounded_rect(cr, cx - 5, cy - 10, 10, 16, 5)
        cr.fill()
        # Mic stand arc
        cr.arc(cx, cy + 3, 9, math.pi, 0)
        cr.set_line_width(2)
        cr.stroke()
        # Mic stem
        cr.move_to(cx, cy + 12)
        cr.line_to(cx, cy + 16)
        cr.stroke()
        cr.move_to(cx - 5, cy + 16)
        cr.line_to(cx + 5, cy + 16)
        cr.stroke()

        if self._shift_held:
            # Small enter arrow indicator
            ex = cx + 16
            ey = cy
            cr.set_source_rgba(1, 1, 1, 0.8)
            cr.set_line_width(1.5)
            cr.move_to(ex + 4, ey - 5)
            cr.line_to(ex + 4, ey + 2)
            cr.line_to(ex - 4, ey + 2)
            cr.stroke()
            # Arrowhead
            cr.move_to(ex - 4, ey + 2)
            cr.line_to(ex - 1, ey - 1)
            cr.move_to(ex - 4, ey + 2)
            cr.line_to(ex - 1, ey + 5)
            cr.stroke()

        # Waveform bars
        total_bars_width = (
            BAR_COUNT * BAR_WIDTH + (BAR_COUNT - 1) * BAR_GAP
        )
        right_padding = 16
        available = width - MIC_AREA_WIDTH - right_padding
        start_x = MIC_AREA_WIDTH + (available - total_bars_width) / 2
        center_y = height / 2

        for i in range(BAR_COUNT):
            if self._transcribing:
                wave = (
                    math.sin(self._animation_phase + i * 0.3)
                    * 0.5 + 0.5
                )
                value = 0.3 + wave * 0.5
                alpha = 0.15 + wave * 0.15
                cr.set_source_rgba(1, 1, 1, alpha)
            else:
                value = self._smoothed[i]
                if self._threshold_crossed:
                    cr.set_source_rgba(1, 1, 1, 1)
                else:
                    cr.set_source_rgba(1, 1, 1, 0.35)

            bar_h = BAR_MIN_HEIGHT + value * (
                BAR_MAX_HEIGHT - BAR_MIN_HEIGHT
            )
            x = start_x + i * (BAR_WIDTH + BAR_GAP)
            y = center_y - bar_h / 2
            _rounded_rect(
                cr, x, y, BAR_WIDTH, bar_h, BAR_WIDTH / 2
            )
            cr.fill()

    def _animate_tick(self) -> bool:
        """GLib timeout callback for transcribing animation."""
        self._animation_phase += 0.15
        if self._animation_phase > math.pi * 2:
            self._animation_phase -= math.pi * 2
        if self._drawing_area:
            self._drawing_area.queue_draw()
        return self._transcribing  # Keep running while transcribing

    def show(self) -> None:
        with self._lock:
            if self._visible:
                return
            self._visible = True

        def _do():
            self._smoothed = [0.0] * BAR_COUNT
            self._shift_held = False
            self._threshold_crossed = False
            self._transcribing = False
            if self._window:
                self._window.set_visible(True)

        GLib.idle_add(_do)

    def hide(self) -> None:
        with self._lock:
            if not self._visible:
                return
            self._visible = False

        def _do():
            if self._window:
                self._window.set_visible(False)
            self._transcribing = False

        GLib.idle_add(_do)

    def update_waveform(
        self, values: list[float], above_threshold: bool = False
    ) -> None:
        with self._lock:
            if not self._visible:
                return

        def _do():
            if len(values) >= BAR_COUNT:
                self._waveform = values[:BAR_COUNT]
            else:
                self._waveform = values + [0.0] * (
                    BAR_COUNT - len(values)
                )
            for i in range(BAR_COUNT):
                target = self._waveform[i]
                self._smoothed[i] += (
                    (target - self._smoothed[i]) * 0.4
                )
            if above_threshold:
                self._threshold_crossed = True
            if self._drawing_area:
                self._drawing_area.queue_draw()

        GLib.idle_add(_do)

    def set_transcribing(self, transcribing: bool) -> None:
        def _do():
            was = self._transcribing
            self._transcribing = transcribing
            if transcribing and not was:
                self._animation_source = GLib.timeout_add(
                    50, self._animate_tick
                )
            if self._drawing_area:
                self._drawing_area.queue_draw()

        GLib.idle_add(_do)

    def set_shift_held(self, held: bool) -> None:
        def _do():
            self._shift_held = held
            if self._drawing_area:
                self._drawing_area.queue_draw()

        GLib.idle_add(_do)


def _rounded_rect(cr, x, y, w, h, r) -> None:
    """Draw a rounded rectangle path."""
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


def get_overlay():
    """Get overlay instance, falling back to NullOverlay."""
    if not _HAS_GTK:
        print(
            "Warning: GTK4 not available, overlay disabled. "
            "Install PyGObject and GTK4."
        )
        return NullOverlay()
    if not _HAS_LAYER_SHELL:
        print(
            "Warning: gtk4-layer-shell not available, "
            "overlay may not float correctly."
        )
    try:
        return LinuxOverlay()
    except Exception as e:
        print(f"Warning: Failed to create overlay: {e}")
        return NullOverlay()

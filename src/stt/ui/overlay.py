"""Recording overlay with waveform visualization"""

import objc
import threading
from typing import Optional

from AppKit import (
    NSWindow,
    NSWindowStyleMaskBorderless,
    NSBackingStoreBuffered,
    NSFloatingWindowLevel,
    NSScreen,
    NSView,
    NSColor,
    NSBezierPath,
    NSGraphicsContext,
    NSCompositingOperationSourceOver,
    NSImage,
    NSImageView,
    NSImageScaleProportionallyUpOrDown,
    NSEvent,
)
from Foundation import NSRect, NSPoint, NSSize, NSMakeRect, NSPointInRect


# Overlay dimensions
PILL_WIDTH = 280
PILL_HEIGHT = 60
BAR_COUNT = 20
BAR_WIDTH = 5
BAR_GAP = 4
BAR_MAX_HEIGHT = 40
BAR_MIN_HEIGHT = 4
CORNER_RADIUS = PILL_HEIGHT / 2
MIC_AREA_WIDTH = 52  # Space for mic icon on left (with padding)


class WaveformView(NSView):
    """Custom view that draws pill background and waveform bars"""

    def initWithFrame_(self, frame):
        self = objc.super(WaveformView, self).initWithFrame_(frame)
        if self:
            self._waveform = [0.0] * BAR_COUNT
            self._smoothed = [0.0] * BAR_COUNT
            self._transcribing = False
            self._animation_phase = 0.0
            self._shift_held = False
            self._threshold_crossed = False  # Latches true once above silence threshold
        return self

    def setShiftHeld_(self, held):
        """Update shift key state"""
        if self._shift_held != held:
            self._shift_held = held
            self.setNeedsDisplay_(True)

    def setTranscribing_(self, transcribing):
        """Switch to transcribing animation mode"""
        self._transcribing = transcribing
        if transcribing:
            self._start_animation()
        else:
            self._stop_animation()

    def _start_animation(self):
        """Start the transcribing animation timer"""
        from Foundation import NSTimer
        self._animation_timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            0.05, True, lambda _: self._animate_step()
        )

    def _stop_animation(self):
        """Stop the transcribing animation timer"""
        if hasattr(self, '_animation_timer') and self._animation_timer:
            self._animation_timer.invalidate()
            self._animation_timer = None

    def _animate_step(self):
        """Advance animation phase"""
        import math
        self._animation_phase += 0.15
        if self._animation_phase > math.pi * 2:
            self._animation_phase -= math.pi * 2
        self.setNeedsDisplay_(True)

    def setWaveform_aboveThreshold_(self, values, above_threshold):
        """Update waveform values (list of 0.0-1.0) and threshold state"""
        if len(values) >= BAR_COUNT:
            self._waveform = values[:BAR_COUNT]
        else:
            # Pad with zeros if not enough values
            self._waveform = values + [0.0] * (BAR_COUNT - len(values))

        # Smooth interpolation
        for i in range(BAR_COUNT):
            target = self._waveform[i]
            self._smoothed[i] += (target - self._smoothed[i]) * 0.4

        if above_threshold:
            self._threshold_crossed = True
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        """Draw the pill background, mic icon, and waveform bars"""
        bounds = self.bounds()

        # Inset slightly for crisp stroke
        inset = 0.5
        pill_rect = NSMakeRect(
            bounds.origin.x + inset,
            bounds.origin.y + inset,
            bounds.size.width - 2 * inset,
            bounds.size.height - 2 * inset
        )

        # Draw pill background
        bg_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.05, 0.05, 0.05, 0.85)
        bg_color.setFill()

        pill_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            pill_rect, CORNER_RADIUS - inset, CORNER_RADIUS - inset
        )
        pill_path.fill()

        # Subtle border for crisp edge
        stroke_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.08)
        stroke_color.setStroke()
        pill_path.setLineWidth_(0.5)
        pill_path.stroke()

        # Draw microphone icon
        bar_color = NSColor.whiteColor()
        bar_color.setFill()
        self._draw_mic_icon(bounds)

        # Draw waveform bars (shifted right for mic icon)
        import math
        total_bars_width = BAR_COUNT * BAR_WIDTH + (BAR_COUNT - 1) * BAR_GAP
        # Center bars between mic area and right edge (with padding for corner)
        right_padding = 16
        available_width = bounds.size.width - MIC_AREA_WIDTH - right_padding
        start_x = MIC_AREA_WIDTH + (available_width - total_bars_width) / 2
        center_y = bounds.size.height / 2

        # Set bar color based on state
        if self._transcribing:
            pass  # Color set per-bar in loop
        elif self._threshold_crossed:
            # Bright white once threshold crossed
            bar_color = NSColor.whiteColor()
            bar_color.setFill()
        else:
            # Dim until threshold crossed
            bar_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.35)
            bar_color.setFill()

        for i in range(BAR_COUNT):
            if self._transcribing:
                # Animated wave pattern during transcription
                wave = math.sin(self._animation_phase + i * 0.3) * 0.5 + 0.5
                value = 0.3 + wave * 0.5
                # Low opacity bars for transcribing
                alpha = 0.15 + wave * 0.15
                bar_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, alpha)
                bar_color.setFill()
            else:
                value = self._smoothed[i]

            bar_height = BAR_MIN_HEIGHT + value * (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT)

            x = start_x + i * (BAR_WIDTH + BAR_GAP)
            y = center_y - bar_height / 2

            bar_rect = NSMakeRect(x, y, BAR_WIDTH, bar_height)
            bar_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bar_rect, BAR_WIDTH / 2, BAR_WIDTH / 2
            )
            bar_path.fill()

    def _draw_mic_icon(self, bounds):
        """Draw microphone icon (and enter icon if shift held) using SF Symbols"""
        from AppKit import NSImageSymbolConfiguration, NSFontWeightRegular

        # NSFontWeightRegular = 0.0, 28pt
        config = NSImageSymbolConfiguration.configurationWithPointSize_weight_(28, NSFontWeightRegular)
        color_config = NSImageSymbolConfiguration.configurationWithHierarchicalColor_(
            NSColor.whiteColor()
        )
        combined = config.configurationByApplyingConfiguration_(color_config)

        mic_image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "mic", "Recording"
        )
        if not mic_image:
            return

        mic_image = mic_image.imageWithSymbolConfiguration_(combined)
        if not mic_image:
            return

        center_y = bounds.size.height / 2

        if self._shift_held:
            # Draw both mic and enter icons side by side
            enter_image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                "return", "Enter"
            )
            if enter_image:
                enter_image = enter_image.imageWithSymbolConfiguration_(combined)

            # Mic icon
            mic_size = mic_image.size()
            mic_scale = 22 / max(mic_size.width, mic_size.height)
            mic_w = mic_size.width * mic_scale
            mic_h = mic_size.height * mic_scale

            # Enter icon
            if enter_image:
                enter_size = enter_image.size()
                enter_scale = 18 / max(enter_size.width, enter_size.height)
                enter_w = enter_size.width * enter_scale
                enter_h = enter_size.height * enter_scale
            else:
                enter_w = 0

            # Position both centered in mic area with left margin
            gap = 4
            total_w = mic_w + gap + enter_w
            start_x = (MIC_AREA_WIDTH - total_w) / 2 + 12

            mic_image.drawInRect_(NSMakeRect(
                start_x, center_y - mic_h / 2, mic_w, mic_h
            ))

            if enter_image:
                enter_image.drawInRect_(NSMakeRect(
                    start_x + mic_w + gap, center_y - enter_h / 2, enter_w, enter_h
                ))
        else:
            # Just mic icon centered
            img_size = mic_image.size()
            scale = 28 / max(img_size.width, img_size.height)
            draw_width = img_size.width * scale
            draw_height = img_size.height * scale

            # Center icon in mic area with left padding for pill corner
            center_x = 14 + (MIC_AREA_WIDTH - 14) / 2
            icon_rect = NSMakeRect(
                center_x - draw_width / 2,
                center_y - draw_height / 2,
                draw_width,
                draw_height
            )
            mic_image.drawInRect_(icon_rect)


class RecordingOverlay:
    """Manages the recording overlay window"""

    def __init__(self):
        self._window: Optional[NSWindow] = None
        self._view: Optional[WaveformView] = None
        self._visible = False
        self._lock = threading.Lock()

    def _ensure_window(self):
        """Create window if not exists (must be called on main thread)"""
        if self._window is not None:
            return

        # Find screen containing mouse cursor
        mouse_loc = NSEvent.mouseLocation()
        screen = None
        for s in NSScreen.screens():
            if NSPointInRect(mouse_loc, s.frame()):
                screen = s
                break
        if screen is None:
            screen = NSScreen.mainScreen()
        screen_frame = screen.frame()

        # Position: horizontally centered, one third from bottom (relative to screen)
        x = screen_frame.origin.x + (screen_frame.size.width - PILL_WIDTH) / 2
        y = screen_frame.origin.y + screen_frame.size.height / 3 - PILL_HEIGHT / 2

        frame = NSMakeRect(x, y, PILL_WIDTH, PILL_HEIGHT)

        # Create borderless, transparent window
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )

        window.setLevel_(NSFloatingWindowLevel)
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setIgnoresMouseEvents_(True)
        window.setHasShadow_(False)

        # Create waveform view
        view_frame = NSMakeRect(0, 0, PILL_WIDTH, PILL_HEIGHT)
        view = WaveformView.alloc().initWithFrame_(view_frame)
        window.setContentView_(view)

        self._window = window
        self._view = view

    def _position_on_mouse_screen(self):
        """Reposition window to screen containing mouse cursor"""
        if self._window is None:
            return

        # Find screen containing mouse cursor
        mouse_loc = NSEvent.mouseLocation()
        screen = None
        for s in NSScreen.screens():
            if NSPointInRect(mouse_loc, s.frame()):
                screen = s
                break
        if screen is None:
            screen = NSScreen.mainScreen()
        screen_frame = screen.frame()

        # Position: horizontally centered, one third from bottom (relative to screen)
        x = screen_frame.origin.x + (screen_frame.size.width - PILL_WIDTH) / 2
        y = screen_frame.origin.y + screen_frame.size.height / 3 - PILL_HEIGHT / 2

        self._window.setFrameOrigin_(NSPoint(x, y))

    def show(self):
        """Show the overlay"""
        with self._lock:
            if self._visible:
                return
            self._visible = True

        def _show():
            self._ensure_window()
            self._position_on_mouse_screen()
            if self._view:
                self._view._smoothed = [0.0] * BAR_COUNT
                self._view._shift_held = False
                self._view._threshold_crossed = False
                self._view.setTranscribing_(False)
            self._window.orderFront_(None)

        _run_on_main_thread(_show)

    def hide(self):
        """Hide the overlay"""
        with self._lock:
            if not self._visible:
                return
            self._visible = False

        def _hide():
            if self._window:
                self._window.orderOut_(None)

        _run_on_main_thread(_hide)

    def update_waveform(self, values: list[float], above_threshold: bool = False):
        """Update waveform display"""
        with self._lock:
            if not self._visible:
                return

        def _update():
            if self._view:
                self._view.setWaveform_aboveThreshold_(values, above_threshold)

        _run_on_main_thread(_update)

    def set_transcribing(self, transcribing: bool):
        """Switch to/from transcribing animation mode"""
        def _set():
            if self._view:
                self._view.setTranscribing_(transcribing)

        _run_on_main_thread(_set)

    def set_shift_held(self, held: bool):
        """Update shift key indicator"""
        def _set():
            if self._view:
                self._view.setShiftHeld_(held)

        _run_on_main_thread(_set)


def _run_on_main_thread(func):
    """Run function on main thread"""
    from Foundation import NSThread, NSRunLoop, NSRunLoopCommonModes

    if NSThread.isMainThread():
        func()
        return

    # Use CFRunLoopPerformBlock equivalent via NSTimer
    from AppKit import NSApplication
    app = NSApplication.sharedApplication()
    if app:
        # Schedule on next run loop iteration
        from Foundation import NSTimer

        def fire_(_):
            func()

        timer = NSTimer.timerWithTimeInterval_repeats_block_(0, False, fire_)
        NSRunLoop.mainRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)


# Global overlay instance
_overlay: Optional[RecordingOverlay] = None


def get_overlay() -> RecordingOverlay:
    """Get or create the global overlay instance"""
    global _overlay
    if _overlay is None:
        _overlay = RecordingOverlay()
    return _overlay

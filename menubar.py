"""Menu bar status icon for STT app"""

import platform
import subprocess
import threading
from typing import TYPE_CHECKING, Callable

import rumps

if TYPE_CHECKING:
    from stt import AppState, STTApp

# SF Symbol names for each state
SF_SYMBOLS = {
    "idle": "mic",
    "recording": "mic.fill",
    "transcribing": "waveform",
}

# Text fallback for older macOS (< 11)
TEXT_TITLES = {
    "idle": "STT",
    "recording": "[REC]",
    "transcribing": "...",
}


def get_macos_version():
    """Get macOS major version number"""
    try:
        version = platform.mac_ver()[0]
        return int(version.split('.')[0])
    except (ValueError, IndexError):
        return 0


def is_sf_symbols_available():
    """Check if SF Symbols are available (macOS 11+)"""
    return get_macos_version() >= 11


class STTMenuBar(rumps.App):
    """Menu bar status icon app"""

    def __init__(
        self,
        stt_app: "STTApp",
        hotkey_name: str,
        provider_name: str,
        sound_enabled: bool,
        config_file: str,
        on_sound_toggle: Callable[[bool], None],
        on_quit: Callable[[], None],
    ):
        # Initialize with template=True for proper dark/light mode
        # quit_button=None disables rumps' automatic Quit item (we add our own)
        super().__init__("STT", template=True, quit_button=None)

        self.stt_app = stt_app
        self.hotkey_name = hotkey_name
        self.provider_name = provider_name
        self._sound_enabled = sound_enabled
        self.config_file = config_file
        self._on_sound_toggle = on_sound_toggle
        self._on_quit = on_quit

        # State synchronization
        self._pending_state = None
        self._state_lock = threading.Lock()

        # Register for state updates from STTApp
        stt_app.set_state_callback(self._on_state_change)

        # Build menu
        self._build_menu()

        # Set initial icon
        self._apply_state("idle")

    def _build_menu(self):
        """Build the menu items"""
        # Info items (disabled)
        hotkey_item = rumps.MenuItem(f"Hotkey: {self.hotkey_name}")
        hotkey_item.set_callback(None)

        provider_item = rumps.MenuItem(f"Provider: {self.provider_name}")
        provider_item.set_callback(None)

        # Toggle sound item
        self._sound_item = rumps.MenuItem("Sound", callback=self._toggle_sound)
        self._sound_item.state = self._sound_enabled

        # Open config item
        config_item = rumps.MenuItem("Open Config...", callback=self._open_config)

        # Quit item
        quit_item = rumps.MenuItem("Quit", callback=self._quit_app)

        self.menu = [
            hotkey_item,
            provider_item,
            None,  # Separator
            self._sound_item,
            config_item,
            None,  # Separator
            quit_item,
        ]

    def _on_state_change(self, state: "AppState"):
        """Called from keyboard callback thread - thread-safe"""
        with self._state_lock:
            self._pending_state = state.value

    @rumps.timer(0.05)
    def _poll_state(self, _):
        """Poll for state changes on main thread (every 50ms)"""
        with self._state_lock:
            if self._pending_state:
                self._apply_state(self._pending_state)
                self._pending_state = None

    def _apply_state(self, state: str):
        """Apply icon state (must be called on main thread)"""
        if is_sf_symbols_available():
            self._apply_sf_symbol(state)
        else:
            # Fallback to text for older macOS
            self.title = TEXT_TITLES.get(state, "STT")

    def _apply_sf_symbol(self, state: str):
        """Apply SF Symbol icon for the given state"""
        try:
            from AppKit import NSImage

            symbol_name = SF_SYMBOLS.get(state, "mic")
            image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                symbol_name, "STT Status"
            )
            if image:
                image.setTemplate_(True)
                # Access the underlying NSStatusItem button
                # rumps stores the status item after run() starts
                if hasattr(self, "_nsapp") and self._nsapp:
                    button = self._nsapp.nsstatusitem.button()
                    if button:
                        button.setImage_(image)
                        self.title = None  # Clear text when using icon
        except Exception:
            # Fallback to text if SF Symbols fail
            self.title = TEXT_TITLES.get(state, "STT")

    def _toggle_sound(self, sender):
        """Toggle sound on/off"""
        self._sound_enabled = not self._sound_enabled
        sender.state = self._sound_enabled
        self._on_sound_toggle(self._sound_enabled)

    def update_sound_enabled(self, enabled: bool):
        """Update sound enabled state (called from config reload)"""
        self._sound_enabled = enabled
        self._sound_item.state = enabled

    def update_provider_name(self, name: str):
        """Update provider name in menu (called from config reload)"""
        self.provider_name = name
        # Update the menu item title
        if len(self.menu) > 1:
            self.menu.values()[1].title = f"Provider: {name}"

    def _open_config(self, _):
        """Open the config file in default editor"""
        subprocess.run(["open", self.config_file])

    def _quit_app(self, _):
        """Quit the application"""
        self._on_quit()
        rumps.quit_application()

#!/usr/bin/env python3
"""
STT - Voice-to-text.

Entry-point module. Keep import-time dependencies lightweight so `import stt`
works in headless contexts (tests, harnesses, etc).
"""

from __future__ import annotations

import atexit
import fcntl
import os
import sys
import tempfile
import threading
from typing import Optional

from stt.app import AppState, STTApp
from stt.defaults import IS_LINUX, IS_MACOS, get_hotkey_display_name


HEADLESS = os.environ.get("STT_HEADLESS") == "1"

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("stt")
except Exception:
    __version__ = "0.0.0"

RELEASES_URL = "https://api.github.com/repos/jamesob/stt/releases/latest"
LOCK_FILE = os.path.join(tempfile.gettempdir(), "stt.lock")

_lock_file: Optional[object] = None


def acquire_lock() -> bool:
    """Ensure only one instance is running."""
    global _lock_file
    _lock_file = open(LOCK_FILE, "w")
    try:
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_file.write(str(os.getpid()))
        _lock_file.flush()
        return True
    except (BlockingIOError, OSError):
        try:
            _lock_file.close()
        except Exception:
            pass
        _lock_file = None
        return False


def release_lock() -> None:
    global _lock_file
    if _lock_file:
        try:
            fcntl.flock(_lock_file, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            _lock_file.close()
        except Exception:
            pass
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass
        _lock_file = None


def check_for_updates() -> None:
    """Check if a newer version is available on GitHub releases."""
    try:
        import requests
        from packaging.version import parse as parse_version

        response = requests.get(RELEASES_URL, timeout=5)
        if response.status_code != 200:
            return
        latest = str(response.json().get("tag_name", "")).lstrip("v")
        if latest and parse_version(latest) > parse_version(__version__):
            print(f"\n📦 Update available: {__version__} → {latest}", flush=True)
            print(
                "   Run: uv tool install --reinstall git+https://github.com/jamesob/stt.git",
                flush=True,
            )
    except Exception:
        pass


def check_accessibility_permissions() -> bool:
    """Check and request accessibility permissions on macOS."""
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions

        options = {"AXTrustedCheckOptionPrompt": True}
        return bool(AXIsProcessTrustedWithOptions(options))
    except ImportError:
        print("Could not check accessibility permissions")
        return True


def check_linux_deps() -> None:
    """Check Linux system dependencies and warn loudly about missing ones."""
    import grp
    import shutil

    from rich.console import Console
    console = Console()

    problems: list[str] = []
    arch_pkgs: list[str] = []
    deb_pkgs: list[str] = []

    # -- input group (evdev keyboard capture) --
    try:
        input_gid = grp.getgrnam("input").gr_gid
        if input_gid not in os.getgroups():
            problems.append(
                "User not in 'input' group -- keyboard capture "
                "will fail.\n"
                "  Fix: sudo usermod -aG input $USER && "
                "newgrp input"
            )
    except KeyError:
        problems.append("'input' group does not exist on this system.")

    # -- wtype (text injection) --
    if not shutil.which("wtype"):
        problems.append("'wtype' not found -- typed text will not appear.")
        arch_pkgs.append("wtype")
        deb_pkgs.append("wtype")

    # -- wl-clipboard (clipboard support) --
    if not shutil.which("wl-copy"):
        problems.append(
            "'wl-copy' not found (wl-clipboard) -- "
            "clipboard features unavailable."
        )
        arch_pkgs.append("wl-clipboard")
        deb_pkgs.append("wl-clipboard")

    # -- PortAudio (audio recording via sounddevice) --
    try:
        import ctypes.util
        if not ctypes.util.find_library("portaudio"):
            raise ImportError
    except (ImportError, OSError):
        problems.append(
            "libportaudio not found -- audio recording will fail."
        )
        arch_pkgs.append("portaudio")
        deb_pkgs.extend(["libportaudio2", "portaudio19-dev"])

    # -- PulseAudio / PipeWire (audio daemon) --
    if not shutil.which("paplay") and not shutil.which("aplay"):
        problems.append(
            "Neither 'paplay' nor 'aplay' found -- "
            "sound feedback will be silent."
        )
        arch_pkgs.append("pipewire-pulse")
        deb_pkgs.append("pipewire-pulse")

    # -- GTK4 + PyGObject (overlay UI) --
    # Don't actually import GTK here -- it initializes a Wayland
    # connection and must happen after the layer-shell preload.
    try:
        import importlib
        importlib.import_module("gi")
    except ImportError:
        problems.append(
            "PyGObject not available -- "
            "recording overlay will be disabled."
        )
        arch_pkgs.append("gobject-introspection")
        deb_pkgs.extend([
            "libgirepository1.0-dev",
            "gir1.2-gtk-4.0",
        ])

    # -- gtk4-layer-shell (overlay floating on Wayland) --
    try:
        import ctypes.util as _cu
        if not _cu.find_library("gtk4-layer-shell"):
            raise OSError
    except OSError:
        problems.append(
            "libgtk4-layer-shell not found -- "
            "overlay will not float above windows."
        )
        arch_pkgs.append("gtk4-layer-shell")
        deb_pkgs.extend([
            "gtk4-layer-shell-dev",
            "gir1.2-gtk4layershell-1.0",
        ])

    if not problems:
        return

    bar = "[bold red]" + "=" * 52 + "[/bold red]"
    console.print()
    console.print(bar)
    console.print(
        "[bold red]  MISSING LINUX DEPENDENCIES[/bold red]"
    )
    console.print(bar)
    console.print()
    for p in problems:
        console.print(f"  [red]* {p}[/red]")

    if arch_pkgs:
        console.print()
        console.print("[bold]Arch Linux:[/bold]")
        console.print(
            f"  sudo pacman -S {' '.join(arch_pkgs)}"
        )
    if deb_pkgs:
        console.print()
        console.print("[bold]Debian / Ubuntu:[/bold]")
        console.print(
            f"  sudo apt install {' '.join(deb_pkgs)}"
        )
    console.print()
    console.print(bar)
    console.print()


def _select_audio_device(*, saved_device_name: str, save_device_fn) -> str | None:
    """List and select an audio input device. Returns device NAME (not index)."""
    import sounddevice as sd

    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices.append((i, dev))

    if saved_device_name:
        for _, dev in input_devices:
            if dev["name"] == saved_device_name:
                return saved_device_name
        print(f"⚠️  Saved device '{saved_device_name}' not found, please select again")

    if len(input_devices) == 1:
        return None  # auto-select the only device

    print("\nAvailable input devices:")
    for i, dev in input_devices:
        marker = "*" if i == sd.default.device[0] else " "
        print(f"  {marker} [{i}] {dev['name']}")
    print("\n  (* = default)")

    while True:
        choice = input("\nSelect device number (or press Enter for default): ").strip()
        if choice == "":
            return None
        try:
            device_idx = int(choice)
            matching = [(i, d) for i, d in input_devices if i == device_idx]
            if matching:
                device_name = matching[0][1]["name"]
                save = input("Save this device for future use? [y/N]: ").strip().lower()
                if save == "y":
                    save_device_fn(device_name)
                return device_name
            print("Invalid device number")
        except ValueError:
            print("Please enter a number")


def main() -> None:
    if HEADLESS:
        print("STT_HEADLESS=1 set; UI disabled")
        raise SystemExit(1)

    # Heavy imports live inside main.
    import subprocess
    import time

    from stt.profiles import load_active_provider
    from stt.prompts_config import ensure_default_prompts
    from stt.config import (
        CONFIG_YAML,
        Config,
        ConfigWatcher,
        _PROVIDER_YAML_KEYS,
        _ENV_TO_YAML,
        get_active_profile_dict,
        is_first_run,
        load_env_startup,
        reload_env_files,
        mark_initialized,
        save_config,
        save_profile_key,
    )

    load_env_startup()
    cfg = Config.from_env()

    # --config: interactive wizard.
    if "--config" in sys.argv:
        from stt.onboarding import run_setup

        prof_name, prof_cfg = get_active_profile_dict()
        target_profile = prof_name or "default"

        def save(key: str, value: str):
            yaml_key = _ENV_TO_YAML.get(key, key.lower())
            if yaml_key in _PROVIDER_YAML_KEYS:
                save_profile_key(target_profile, yaml_key, value)
            else:
                save_config(key, value, force_global=True)
            os.environ[str(key)] = str(value)

        current_config = {
            "provider": prof_cfg.get("provider", ""),
            "model": prof_cfg.get("whisper_model", ""),
            "groq_api_key": prof_cfg.get("groq_api_key", ""),
            "hotkey": cfg.hotkey,
            "audio_device": cfg.audio_device,
            "openai_base_url": prof_cfg.get(
                "openai_base_url", ""),
            "openai_api_key": prof_cfg.get(
                "openai_api_key", ""),
            "openai_whisper_model": prof_cfg.get(
                "openai_whisper_model", ""),
            "whisper_cpp_http_url": prof_cfg.get(
                "whisper_cpp_http_url", ""),
        }
        run_setup(save, current_config=current_config, reconfigure=True)
        # Ensure active_profile is set after wizard.
        save_config(
            "active_profile", target_profile,
            force_global=True,
        )
        return

    # Dev-only permission flow testing (macOS only).
    if "--test-permissions" in sys.argv and IS_MACOS:
        from stt.onboarding import (
            console,
            get_terminal_app,
            open_accessibility_settings,
            open_input_monitoring_settings,
            show_permission_error,
        )
        from rich.prompt import Confirm

        show_permission_error()
        if Confirm.ask("Open Accessibility settings?", default=True):
            open_accessibility_settings()
            console.print(
                "\n[dim]Enable the permission, then come back here.[/dim]\n"
            )
            Confirm.ask("Done with Accessibility?", default=True)
        if Confirm.ask("Open Input Monitoring settings?", default=True):
            open_input_monitoring_settings()
            console.print(
                "\n[dim]Enable the permission, then come back here.[/dim]\n"
            )
            Confirm.ask("Done with Input Monitoring?", default=True)

        terminal = get_terminal_app()
        console.print("\n[green]Permission setup complete.[/green]")
        console.print(
            f"[yellow]Restart {terminal} and run STT again.[/yellow]\n"
        )
        return

    # Ensure only one instance.
    if not acquire_lock():
        from rich.console import Console

        Console().print(
            "[red]Another instance of STT is already running[/red]"
        )
        raise SystemExit(1)
    atexit.register(release_lock)

    # First-run onboarding.
    if is_first_run():
        from stt.onboarding import run_first_time_setup

        def save_and_update(key: str, value: str):
            yaml_key = _ENV_TO_YAML.get(key, key.lower())
            if yaml_key in _PROVIDER_YAML_KEYS:
                save_profile_key("default", yaml_key, value)
            else:
                save_config(key, value, force_global=True)
            os.environ[str(key)] = str(value)

        run_first_time_setup(save_and_update)
        save_config(
            "active_profile", "default", force_global=True,
        )
        mark_initialized()
        reload_env_files()
        cfg = Config.from_env()

    from rich.console import Console
    from rich.status import Status

    console = Console()
    console.print()
    console.print("[bold]STT[/bold] [dim]Voice-to-text[/dim]")
    console.print("[dim]https://github.com/jamesob/stt[/dim]")
    console.print()

    threading.Thread(target=check_for_updates, daemon=True).start()

    # Initialize provider via profiles.
    provider = None
    init_error = None

    status = Status(
        "[dim]Initializing...[/dim]", console=console, spinner="dots"
    )
    status.start()

    slow_timer = threading.Timer(
        2.0,
        lambda: status.update(
            "[dim]Initializing... first run may take ~30s[/dim]"
        ),
    )
    slow_timer.start()
    try:
        provider = load_active_provider(cfg.active_profile)
    except ValueError as e:
        init_error = e
    finally:
        slow_timer.cancel()
        status.stop()

    if init_error:
        console.print(f"[red]x[/red] {init_error}")
        raise SystemExit(1)

    if not provider.is_available():
        # Check if this is a groq profile missing an API key.
        _, prof_cfg = get_active_profile_dict()
        prov_name = prof_cfg.get("provider", "")
        if prov_name == "groq" and not prof_cfg.get("groq_api_key"):
            from stt.onboarding import run_setup

            target = cfg.active_profile or "default"

            def save(key: str, value: str):
                yaml_key = _ENV_TO_YAML.get(key, key.lower())
                if yaml_key in _PROVIDER_YAML_KEYS:
                    save_profile_key(target, yaml_key, value)
                else:
                    save_config(
                        key, value, force_global=True)
                os.environ[str(key)] = str(value)

            run_setup(
                save, current_config={
                    "provider": prov_name,
                    "groq_api_key": "",
                    "hotkey": cfg.hotkey,
                    "audio_device": cfg.audio_device,
                }, reconfigure=True,
            )
            reload_env_files()
            cfg = Config.from_env()
            provider = load_active_provider(cfg.active_profile)
        else:
            console.print(
                f"[red]x[/red] Provider not available: "
                f"{provider.name}"
            )
            if prov_name == "mlx":
                console.print(
                    "  [dim]Install with: "
                    "pip install mlx-whisper[/dim]"
                )
            raise SystemExit(1)

    assert provider is not None
    console.print(
        f"[green]>[/green] Provider: [cyan]{provider.name}[/cyan]"
    )
    provider.warmup()

    # Permission checks (platform-specific).
    if IS_MACOS:
        if not check_accessibility_permissions():
            from stt.onboarding import (
                get_terminal_app,
                open_accessibility_settings,
                prompt_open_settings,
                show_permission_error,
            )

            show_permission_error()
            if prompt_open_settings():
                open_accessibility_settings()
            terminal = get_terminal_app()
            console.print(
                f"\n[yellow]Restart {terminal} and run STT again.[/yellow]"
            )
            raise SystemExit(1)
    elif IS_LINUX:
        check_linux_deps()

    # Select audio device (uses saved device or prompts).
    def save_device(device_name: str) -> None:
        env_path = save_config("AUDIO_DEVICE", device_name)
        os.environ["AUDIO_DEVICE"] = device_name
        print(f"  (saved to {env_path})")

    device_name = _select_audio_device(
        saved_device_name=cfg.audio_device, save_device_fn=save_device
    )
    if device_name:
        console.print(
            f"[green]>[/green] Device: [cyan]{device_name}[/cyan]"
        )
    else:
        console.print(
            "[green]>[/green] Device: [cyan]System default[/cyan]"
        )

    console.print(
        "[bright_black]Hint:[/bright_black] "
        "[dim]Run[/dim] [grey70]stt --config[/grey70] "
        "[dim]to change settings[/dim]"
    )
    console.print()

    # Ensure default prompts exist before PromptOverlay loads.
    ensure_default_prompts()

    # Platform-specific UI wiring.
    from stt.input.controller import InputController

    if IS_LINUX:
        from stt.ui.linux_overlay import get_overlay

        overlay = get_overlay()

        # Freedesktop sound paths
        _SOUND_MAP = {
            "/System/Library/Sounds/Tink.aiff": (
                "/usr/share/sounds/freedesktop/stereo/message-new-instant.oga"
            ),
            "/System/Library/Sounds/Pop.aiff": (
                "/usr/share/sounds/freedesktop/stereo/message.oga"
            ),
            "/System/Library/Sounds/Basso.aiff": (
                "/usr/share/sounds/freedesktop/stereo/dialog-warning.oga"
            ),
        }

        def play_sound(sound_path: str) -> None:
            if not cfg.sound_enabled:
                return
            mapped = _SOUND_MAP.get(sound_path, sound_path)
            for cmd in ("paplay", "aplay"):
                try:
                    subprocess.Popen(
                        [cmd, mapped],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
                except FileNotFoundError:
                    continue

        from stt.text.linux_injector import paste_text as linux_paste

        def type_text(text: str, send_enter: bool = False) -> None:
            linux_paste(text, send_enter=send_enter)

    else:
        from stt.ui.overlay import get_overlay
        from stt.text.injector import paste_text

        overlay = get_overlay()

        def play_sound(sound_path: str) -> None:
            if not cfg.sound_enabled:
                return
            subprocess.Popen(
                ["afplay", sound_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        def type_text(text: str, send_enter: bool = False) -> None:
            paste_text(text, send_enter=send_enter, method="osascript")

    app = STTApp(
        device_name=device_name,
        provider=provider,
        overlay=overlay,
        sound_player=play_sound,
        text_injector=type_text,
        language=cfg.language,
        prompt=cfg.prompt,
        hotkey_id=cfg.hotkey,
        keep_recordings=cfg.keep_recordings,
    )

    # Wait for benchmark providers to finish warming up.
    wait_ready = getattr(provider, "wait_ready", None)
    if callable(wait_ready):
        wait_ready()

    hotkey_name = get_hotkey_display_name(cfg.hotkey)
    console.print(
        f"[bold green]Ready[/bold green] [dim]|[/dim] "
        f"Hold [cyan]{hotkey_name}[/cyan] to record, +Shift Enter, Esc cancel"
    )
    console.print()

    controller = InputController(app, hotkey_id=cfg.hotkey)

    config_watcher: Optional[ConfigWatcher] = None
    menubar = None  # Only set on macOS

    def cleanup():
        if config_watcher:
            config_watcher.stop()
        controller.stop()
        # Shut down provider workers (MLX subprocess, etc.)
        cancel = getattr(provider, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass

    atexit.register(cleanup)

    if IS_MACOS:
        from stt.ui.menubar import STTMenuBar

        def on_sound_toggle(enabled: bool):
            nonlocal cfg
            cfg.sound_enabled = enabled
            save_config("SOUND_ENABLED", str(enabled).lower())
            os.environ["SOUND_ENABLED"] = str(enabled).lower()

        menubar = STTMenuBar(
            stt_app=app,
            hotkey_name=hotkey_name,
            provider_name=provider.name,
            sound_enabled=cfg.sound_enabled,
            config_file=CONFIG_YAML,
            on_sound_toggle=on_sound_toggle,
            on_quit=cleanup,
        )

    # Config change handler.
    def on_config_change(new_cfg: Config, changes: dict):
        nonlocal cfg, provider, hotkey_name
        cfg = new_cfg

        if "AUDIO_DEVICE" in changes:
            app.device_name = cfg.audio_device or None
            print(f"   Audio device: {cfg.audio_device or 'default'}")
        if "LANGUAGE" in changes:
            app.language = cfg.language
            print(f"   Language: {cfg.language}")
        if "HOTKEY" in changes:
            controller.set_hotkey_id(cfg.hotkey)
            hotkey_name = get_hotkey_display_name(cfg.hotkey)
            if menubar:
                menubar.update_hotkey_name(hotkey_name)
            print(f"   Hotkey: {hotkey_name}")
        if "PROMPT" in changes:
            app.prompt = cfg.prompt
            print(f"   Prompt: {cfg.prompt or '(empty)'}")
        if "KEEP_RECORDINGS" in changes:
            app.keep_recordings = cfg.keep_recordings
            print(
                "   Keep recordings: "
                f"{'enabled' if cfg.keep_recordings else 'disabled'}"
            )
        if "SOUND_ENABLED" in changes:
            if menubar:
                menubar.update_sound_enabled(cfg.sound_enabled)
            print(
                "   Sound: "
                f"{'enabled' if cfg.sound_enabled else 'disabled'}"
            )

        # Profile or provider config changed.
        if "STT_PROFILE" in changes or "_PROFILES_CHANGED" in changes:
            try:
                provider = load_active_provider(
                    cfg.active_profile)
                if provider.is_available():
                    provider.warmup()
                    app.provider = provider
                    if menubar:
                        menubar.update_provider_name(
                            provider.name)
                    print(f"   Provider: {provider.name}")
                else:
                    print(
                        f"   Provider '{provider.name}' "
                        "not available"
                    )
            except Exception as e:
                print(
                    f"   Failed to reinitialize "
                    f"provider: {e}"
                )

    config_watcher = ConfigWatcher(on_config_change)
    config_watcher.start()

    controller.start()

    try:
        if menubar:
            menubar.run()
        elif IS_LINUX:
            try:
                from gi.repository import GLib
                loop = GLib.MainLoop()
                loop.run()
            except ImportError:
                threading.Event().wait()
        else:
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
        print("\nBye.")


if __name__ == "__main__":
    main()

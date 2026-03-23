from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from dotenv import dotenv_values
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from stt.defaults import IS_LINUX


CONFIG_DIR = os.path.expanduser("~/.config/stt")
CONFIG_FILE = os.path.join(CONFIG_DIR, ".env")
CONFIG_YAML = os.path.join(CONFIG_DIR, "config.yml")
INITIALIZED_MARKER = os.path.join(CONFIG_DIR, ".initialized")
LOCK_FILE = os.path.join(tempfile.gettempdir(), "stt.lock")

_BASE_ENV_KEYS = frozenset(os.environ.keys())
_MANAGED_ENV_KEYS: set[str] = set()


_DEFAULT_PROVIDER = "openai" if IS_LINUX else "mlx"

# Provider-specific YAML keys (used for migration only).
_PROVIDER_YAML_KEYS = frozenset({
    "provider", "groq_api_key", "whisper_model", "parakeet_model",
    "whisper_cpp_http_url", "openai_base_url", "openai_api_key",
    "openai_whisper_model",
})

# Mapping between YAML key names and ENV var names (general settings only).
_YAML_TO_ENV = {
    "audio_device": "AUDIO_DEVICE",
    "language": "LANGUAGE",
    "hotkey": "HOTKEY",
    "prompt": "PROMPT",
    "active_profile": "STT_PROFILE",
    "sound_enabled": "SOUND_ENABLED",
    "keep_recordings": "KEEP_RECORDINGS",
}
_ENV_TO_YAML = {v: k for k, v in _YAML_TO_ENV.items()}

# Canonical key order for YAML output.
_YAML_KEY_ORDER = [
    "language", "hotkey", "audio_device",
    "prompt", "sound_enabled", "keep_recordings",
    "active_profile", "profiles",
]


def _read_yaml() -> dict:
    """Read config.yml and return the raw dict."""
    try:
        import yaml

        with open(CONFIG_YAML) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _migrate_flat_to_profiles(data: dict) -> dict:
    """Migrate legacy flat provider keys into a default profile.

    If flat provider keys exist at the top level but no profiles
    section is present, move them into profiles.default and set
    active_profile.  Rewrites config.yml in place.
    """
    flat_keys = _PROVIDER_YAML_KEYS & set(data)
    if not flat_keys:
        return data
    if "profiles" in data and data["profiles"]:
        # Profiles already exist; strip stale flat keys.
        for k in flat_keys:
            data.pop(k, None)
        _write_yaml(data)
        return data

    profile: dict = {}
    for k in list(flat_keys):
        profile[k] = data.pop(k)
    data.setdefault("profiles", {})["default"] = profile
    data.setdefault("active_profile", "default")

    _write_yaml(data)
    print("Config migrated: provider keys moved to profiles.default")
    return data


def _ordered_yaml_dict(data: dict) -> dict:
    """Return data dict ordered by canonical key order."""
    ordered: dict = {}
    for k in _YAML_KEY_ORDER:
        if k in data:
            ordered[k] = data[k]
    for k, v in data.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _write_yaml(data: dict) -> str:
    """Write a full dict to config.yml."""
    import yaml

    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_YAML, "w") as f:
        yaml.dump(
            _ordered_yaml_dict(data), f,
            default_flow_style=False, sort_keys=False,
        )
    return CONFIG_YAML


def _save_to_yaml(key: str, value: str) -> str:
    """Save a single top-level config value to config.yml."""
    yaml_key = _ENV_TO_YAML.get(key, key.lower())

    data = _read_yaml() if os.path.exists(CONFIG_YAML) else {}

    if not value:
        data.pop(yaml_key, None)
    elif value.lower() in ("true", "false"):
        data[yaml_key] = value.lower() == "true"
    else:
        data[yaml_key] = value

    return _write_yaml(data)


def save_profile_key(
    profile_name: str, key: str, value: str,
) -> str:
    """Save a single key under profiles.<profile_name> in config.yml."""
    data = _read_yaml() if os.path.exists(CONFIG_YAML) else {}
    profiles = data.setdefault("profiles", {})
    profile = profiles.setdefault(profile_name, {})

    if not value:
        profile.pop(key, None)
    elif value.lower() in ("true", "false"):
        profile[key] = value.lower() == "true"
    else:
        profile[key] = value

    return _write_yaml(data)


def get_active_profile_dict() -> tuple[str, dict]:
    """Return (profile_name, profile_cfg) for the active profile.

    Reads from YAML config.  Returns ("", {}) if no profile is active.
    """
    data = _read_yaml()
    if not data:
        return "", {}
    data = _migrate_flat_to_profiles(data)
    name = (
        os.environ.get("STT_PROFILE", "")
        or data.get("active_profile", "")
        or data.get("active", "")
    )
    if not name:
        return "", {}
    profiles = data.get("profiles", {})
    return name, profiles.get(name, {})


@dataclass
class Config:
    active_profile: str = ""
    audio_device: str = ""
    language: str = "en"
    hotkey: str = "cmd_r"
    prompt: str = ""
    sound_enabled: bool = True
    keep_recordings: bool = False

    @staticmethod
    def from_env() -> "Config":
        return Config(
            active_profile=os.environ.get("STT_PROFILE", ""),
            audio_device=os.environ.get("AUDIO_DEVICE", ""),
            language=os.environ.get("LANGUAGE", "en"),
            hotkey=os.environ.get("HOTKEY", "cmd_r"),
            prompt=os.environ.get("PROMPT", ""),
            sound_enabled=os.environ.get(
                "SOUND_ENABLED", "true"
            ).lower() == "true",
            keep_recordings=os.environ.get(
                "KEEP_RECORDINGS", "false"
            ).lower() == "true",
        )

    def to_env_dict(self) -> dict[str, str]:
        return {
            "STT_PROFILE": self.active_profile,
            "AUDIO_DEVICE": self.audio_device,
            "LANGUAGE": self.language,
            "HOTKEY": self.hotkey,
            "PROMPT": self.prompt,
            "SOUND_ENABLED": str(self.sound_enabled).lower(),
            "KEEP_RECORDINGS": str(self.keep_recordings).lower(),
        }


def load_env_startup() -> None:
    """Load environment variables from config files.

    Precedence (highest to lowest):
    - OS env
    - local .env (cwd)
    - config.yml general settings
    - global ~/.config/stt/.env
    """
    # Run migration before loading env.
    if os.path.exists(CONFIG_YAML):
        data = _read_yaml()
        if data:
            _migrate_flat_to_profiles(data)
    _apply_env_files(is_startup=True)


def reload_env_files() -> None:
    """Reload env files and apply changes."""
    _apply_env_files(is_startup=False)


def _read_env_files() -> dict[str, str]:
    """Read config files without mutating the process environment.

    Only reads general (non-provider) settings from YAML.
    Falls back to .env for legacy setups.
    """
    data: dict[str, str] = {}

    if os.path.exists(CONFIG_YAML):
        yaml_data = _read_yaml()
        for yaml_key, env_key in _YAML_TO_ENV.items():
            if yaml_key in yaml_data:
                val = yaml_data[yaml_key]
                if isinstance(val, bool):
                    data[env_key] = str(val).lower()
                elif val is not None:
                    data[env_key] = str(val)
    elif os.path.exists(CONFIG_FILE):
        for k, v in dotenv_values(CONFIG_FILE).items():
            if v is None:
                continue
            data[str(k)] = str(v)

    local_env = os.path.join(os.getcwd(), ".env")
    if os.path.exists(local_env):
        for k, v in dotenv_values(local_env).items():
            if v is None:
                continue
            data[str(k)] = str(v)

    return data


def _apply_env_files(*, is_startup: bool) -> None:
    """Apply env file values into os.environ."""
    global _MANAGED_ENV_KEYS

    data = _read_env_files()
    next_managed = {k for k in data.keys() if k not in _BASE_ENV_KEYS}

    if not is_startup:
        for key in list(_MANAGED_ENV_KEYS):
            if key not in next_managed:
                os.environ.pop(key, None)

    for key in next_managed:
        os.environ[key] = data[key]

    _MANAGED_ENV_KEYS = next_managed


def is_first_run() -> bool:
    return not os.path.exists(INITIALIZED_MARKER)


def mark_initialized() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(INITIALIZED_MARKER, "w", encoding="utf-8") as f:
        f.write("")


def mask_api_key(key: str) -> str:
    if not key or len(key) < 8:
        return ""
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def save_config(key: str, value: str, *, force_global: bool = False) -> str:
    """Save a config value.

    Uses config.yml when it exists or when force_global is set.
    Falls back to .env for legacy setups.
    """
    if force_global or os.path.exists(CONFIG_YAML):
        return _save_to_yaml(key, value)

    local_env = os.path.join(os.getcwd(), ".env")
    if os.path.exists(local_env):
        env_path = local_env
    else:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        env_path = CONFIG_FILE

    lines: list[str] = []
    found = False

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                found = True
                break

    if not found:
        lines.append(f"{key}={value}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return env_path


class ConfigWatcher:
    """Watch config files for changes and trigger reload."""

    def __init__(
        self,
        on_config_change: Callable[[Config, dict[str, Any]], None],
    ):
        self._on_config_change = on_config_change
        self._observer: Optional[Observer] = None
        self._watched_files: set[str] = set()
        self._last_mtime: dict[str, float] = {}
        self._debounce_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._last_yaml_snapshot: str = ""

    def start(self):
        local_env = os.path.join(os.getcwd(), ".env")
        if os.path.exists(local_env):
            self._watched_files.add(local_env)
        if os.path.exists(CONFIG_YAML):
            self._watched_files.add(CONFIG_YAML)
        elif os.path.exists(CONFIG_DIR):
            self._watched_files.add(CONFIG_YAML)
        if os.path.exists(CONFIG_FILE):
            self._watched_files.add(CONFIG_FILE)
        elif os.path.exists(CONFIG_DIR):
            self._watched_files.add(CONFIG_FILE)

        if not self._watched_files:
            return

        # Snapshot YAML for profile change detection.
        self._last_yaml_snapshot = self._yaml_snapshot()

        for path in self._watched_files:
            if os.path.exists(path):
                self._last_mtime[path] = os.path.getmtime(path)

        self._observer = Observer()
        handler = _ConfigFileHandler(
            self._on_file_changed, self._watched_files,
        )

        watched_dirs = set()
        for path in self._watched_files:
            dir_path = os.path.dirname(path)
            if dir_path and dir_path not in watched_dirs:
                watched_dirs.add(dir_path)
                self._observer.schedule(
                    handler, dir_path, recursive=False,
                )

        self._observer.start()

    def stop(self):
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
        with self._lock:
            if self._debounce_timer:
                self._debounce_timer.cancel()
                self._debounce_timer = None

    def _on_file_changed(self, path: str):
        with self._lock:
            if os.path.exists(path):
                new_mtime = os.path.getmtime(path)
                old_mtime = self._last_mtime.get(path, 0)
                if new_mtime == old_mtime:
                    return
                self._last_mtime[path] = new_mtime

            if self._debounce_timer:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(
                0.1, self._reload_config,
            )
            self._debounce_timer.start()

    @staticmethod
    def _yaml_snapshot() -> str:
        """Hash-like snapshot of profiles + active_profile."""
        import json
        data = _read_yaml()
        relevant = {
            "active_profile": data.get("active_profile", ""),
            "profiles": data.get("profiles", {}),
        }
        return json.dumps(relevant, sort_keys=True)

    def _reload_config(self):
        old = Config.from_env().to_env_dict()
        old_yaml = self._last_yaml_snapshot

        _apply_env_files(is_startup=False)

        new_cfg = Config.from_env()
        new = new_cfg.to_env_dict()

        changes: dict[str, Any] = {}
        for k, old_v in old.items():
            if old_v != new.get(k, ""):
                if k in ("SOUND_ENABLED", "KEEP_RECORDINGS"):
                    changes[k] = (
                        new.get(k, "false").lower() == "true"
                    )
                else:
                    changes[k] = new.get(k, "")

        # Detect profile content changes.
        new_yaml = self._yaml_snapshot()
        if new_yaml != old_yaml:
            changes["_PROFILES_CHANGED"] = True
            self._last_yaml_snapshot = new_yaml

        if changes:
            print(f"Config reloaded: {', '.join(changes.keys())}")
            self._on_config_change(new_cfg, changes)


class _ConfigFileHandler(FileSystemEventHandler):
    def __init__(
        self, callback: Callable[[str], None],
        watched_files: set[str],
    ):
        self._callback = callback
        self._watched_files = watched_files

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path in self._watched_files:
            self._callback(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path in self._watched_files:
            self._callback(event.src_path)

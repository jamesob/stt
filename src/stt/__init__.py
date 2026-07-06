"""STT - Voice-to-text for macOS and Linux."""

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("stt")
except Exception:
    __version__ = "0.0.0"

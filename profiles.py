"""YAML-based provider profiles for STT.

Profiles allow named provider configurations switchable via
STT_PROFILE env var or the ``active`` key in profiles.yml.

Fallback entries reference other profiles by name::

    profiles:
      local:
        provider: mlx

      remote:
        provider: openai
        openai_base_url: http://stt.j.co:8200
        openai_whisper_model: Qwen/Qwen3-ASR-1.7B

      auto:
        fallback:
          - remote:
              connect_timeout: 2
          - local

    active: auto
"""

from __future__ import annotations

import os
from pathlib import Path

from stt_config import CONFIG_DIR


PROFILES_FILENAME = "profiles.yml"


def load_profiles() -> dict | None:
    """Load profiles from config.yml or profiles.yml.

    Checks config.yml first (unified config), then falls back
    to standalone profiles.yml.  Returns the parsed dict or
    None if no profiles are found.
    """
    import yaml

    # Unified config.yml (may contain a profiles: key)
    for path in [
        Path.cwd() / "config.yml",
        Path(CONFIG_DIR) / "config.yml",
    ]:
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "profiles" in data:
                return data

    # Standalone profiles.yml (legacy)
    for path in [
        Path.cwd() / PROFILES_FILENAME,
        Path(CONFIG_DIR) / PROFILES_FILENAME,
    ]:
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    return None


def get_active_profile(
    profiles_data: dict, profile_name: str = "",
) -> dict | None:
    """Resolve a profile config dict by name.

    *profile_name* (from env/config) takes precedence over the
    ``active`` key inside the YAML.
    """
    name = (
        profile_name
        or profiles_data.get("active", "")
    )
    if not name:
        return None
    return profiles_data.get("profiles", {}).get(name)


def provider_from_profile(
    profile_cfg: dict, all_profiles: dict,
):
    """Create a TranscriptionProvider from a profile dict.

    *all_profiles* is the ``profiles:`` mapping so that
    fallback entries can reference other profiles by name.
    """
    from providers import BenchmarkProvider, FallbackProvider

    if "fallback" in profile_cfg:
        chain = []
        for entry in profile_cfg["fallback"]:
            ref, timeout = _resolve_fallback_entry(
                entry, all_profiles)
            prov = _make_provider(ref)
            chain.append((prov, float(timeout)))
        primary = FallbackProvider(chain)
    else:
        primary = _make_provider(profile_cfg)

    # benchmark: list of profile names to run in parallel
    bench_refs = profile_cfg.get("benchmark")
    if bench_refs:
        others = []
        for ref in bench_refs:
            if isinstance(ref, str):
                cfg = all_profiles.get(ref)
                if cfg is None:
                    raise ValueError(
                        f"Benchmark references unknown "
                        f"profile: {ref!r}")
                others.append(_make_provider(cfg))
            elif isinstance(ref, dict):
                others.append(_make_provider(ref))
        if others:
            return BenchmarkProvider(primary, others)

    return primary


def _resolve_fallback_entry(
    entry, all_profiles: dict,
) -> tuple[dict, float]:
    """Resolve a fallback list item to (profile_cfg, timeout).

    Accepts three forms::

        - local                     # bare profile name
        - remote:                   # name with overrides
            connect_timeout: 2
        - provider: openai          # inline config (no ref)
            openai_base_url: ...
    """
    # Bare string: "- local"
    if isinstance(entry, str):
        cfg = all_profiles.get(entry)
        if cfg is None:
            raise ValueError(
                f"Fallback references unknown profile: "
                f"{entry!r}")
        return cfg, 0

    # Dict with a single key that matches a profile name:
    # "- remote:\n    connect_timeout: 2"
    if isinstance(entry, dict):
        profile_refs = [
            k for k in entry
            if k in all_profiles
        ]
        if len(profile_refs) == 1:
            ref_name = profile_refs[0]
            overrides = entry[ref_name] or {}
            if not isinstance(overrides, dict):
                overrides = {}
            timeout = overrides.get("connect_timeout", 0)
            return all_profiles[ref_name], timeout

        # Inline config (has "provider" key directly)
        if "provider" in entry:
            timeout = entry.get("connect_timeout", 0)
            return entry, timeout

        raise ValueError(
            f"Fallback entry not recognized: {entry!r}")

    raise ValueError(
        f"Fallback entry must be a string or dict: "
        f"{entry!r}")


def _make_provider(cfg: dict):
    """Instantiate a single provider from config keys."""
    from providers import (
        GroqProvider,
        MLXWhisperProvider,
        OpenAICompatibleProvider,
        ParakeetProvider,
        WhisperCPPHTTPProvider,
    )

    name = cfg.get("provider", "mlx")

    if name == "groq":
        return GroqProvider(api_key=cfg.get("groq_api_key"))
    elif name == "mlx":
        return MLXWhisperProvider(
            model=cfg.get("whisper_model"))
    elif name == "openai":
        return OpenAICompatibleProvider(
            base_url=cfg.get("openai_base_url"),
            api_key=cfg.get("openai_api_key"),
            model=cfg.get("openai_whisper_model"),
        )
    elif name == "whisper-cpp-http":
        return WhisperCPPHTTPProvider(
            base_url=cfg.get("whisper_cpp_http_url"))
    elif name == "parakeet":
        return ParakeetProvider(
            model=cfg.get("parakeet_model"))
    else:
        raise ValueError(
            f"Unknown provider in profile: {name}")

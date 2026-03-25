"""Backend loading for STT.

All provider configuration lives under the ``backends:`` key in
config.yml.  The ``order:`` key selects which backends to try, in
fallback order::

    backends:
      remote:
        provider: openai
        openai_base_url: http://gpu-server:8200
        openai_whisper_model: Qwen/Qwen3-ASR-1.7B
        connect_timeout: 2

      local:
        provider: mlx
        whisper_model: large-v3-turbo

    order:
      - remote
      - local
"""

from __future__ import annotations

from stt.config import get_backends_and_order


def load_provider(*, benchmark: bool = False):
    """Load and return the active TranscriptionProvider.

    Reads backends + order from config, builds a FallbackProvider
    chain if multiple backends are listed, or a single provider
    if only one.

    When *benchmark* is True, all backends run in parallel on each
    transcription and results are logged for comparison. The first
    backend in *order* is the primary (its result is used).
    """
    backends, order = get_backends_and_order()
    if not backends:
        raise ValueError(
            "No backends configured in config.yml. "
            "Run stt --config to set up."
        )
    if not order:
        raise ValueError(
            "No order configured in config.yml. "
            "Run stt --config to set up."
        )

    # Validate all referenced backends exist.
    for name in order:
        if name not in backends:
            raise ValueError(
                f"Order references unknown backend: {name!r}"
            )

    chain = []
    for name in order:
        cfg = backends[name]
        prov = _make_provider(cfg)
        timeout = float(cfg.get("connect_timeout", 0))
        chain.append((prov, timeout))

    if len(chain) == 1:
        return chain[0][0]

    from stt.providers import FallbackProvider

    if benchmark:
        from stt.providers import BenchmarkProvider
        primary = chain[0][0]
        others = [p for p, _ in chain[1:]]
        return BenchmarkProvider(primary, others)

    return FallbackProvider(chain)


def _make_provider(cfg: dict):
    """Instantiate a single provider from config keys."""
    from stt.providers import (
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
            f"Unknown provider in backend: {name}")

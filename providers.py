"""
Transcription providers for STT
"""

import os
from abc import ABC, abstractmethod


class TranscriptionProvider(ABC):
    """Base class for transcription providers"""

    @abstractmethod
    def transcribe(self, audio_file_path: str, language: str, prompt: str = None) -> str | None:
        """Transcribe audio file to text"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name"""
        pass

    def warmup(self) -> None:
        """Initialize/preload resources. Override if needed."""
        pass


class GroqProvider(TranscriptionProvider):
    """Groq Whisper API provider (cloud)"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")

    @property
    def name(self) -> str:
        return "Groq (cloud)"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def transcribe(self, audio_file_path: str, language: str, prompt: str = None) -> str | None:
        import requests

        print("🔄 Transcribing via Groq...")

        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        with open(audio_file_path, "rb") as audio_file:
            files = {
                "file": ("audio.wav", audio_file, "audio/wav")
            }
            data = {
                "model": "whisper-large-v3",
                "response_format": "text",
                "language": language,
            }
            if prompt:
                data["prompt"] = prompt

            try:
                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                return response.text.strip()
            except requests.exceptions.RequestException as e:
                print(f"❌ API Error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response: {e.response.text}")
                return None


class MLXWhisperProvider(TranscriptionProvider):
    """Local Whisper provider using MLX (Apple Silicon optimized)"""

    DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"

    def __init__(self, model: str = None):
        self.model = model or os.environ.get("WHISPER_MODEL", self.DEFAULT_MODEL)
        # Normalize model name if user provides short form
        if self.model and not self.model.startswith("mlx-community/"):
            if self.model in ("large-v3", "large", "medium", "small", "base", "tiny"):
                self.model = f"mlx-community/whisper-{self.model}-mlx"
        self._mlx_whisper = None

    @property
    def name(self) -> str:
        return f"MLX Whisper ({self.model.split('/')[-1]})"

    def is_available(self) -> bool:
        try:
            import mlx_whisper
            return True
        except ImportError:
            return False

    def warmup(self) -> None:
        """Pre-load the model at startup (downloads if needed)"""
        print(f"Loading MLX Whisper model ({self.model.split('/')[-1]})...", flush=True)

        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
        import mlx.core as mx
        # Load into ModelHolder cache so transcribe() reuses it
        ModelHolder.get_model(self.model, mx.float16)
        self._mlx_whisper = mlx_whisper
        print("Model loaded.")

    def transcribe(self, audio_file_path: str, language: str, prompt: str = None) -> str | None:
        if self._mlx_whisper is None:
            try:
                import mlx_whisper
                self._mlx_whisper = mlx_whisper
            except ImportError:
                print("❌ mlx-whisper not installed. Run: pip install mlx-whisper")
                return None

        print("🔄 Transcribing...")

        try:
            result = self._mlx_whisper.transcribe(
                audio_file_path,
                path_or_hf_repo=self.model,
                language=language,
                initial_prompt=prompt,
            )
            return result["text"].strip()
        except Exception as e:
            print(f"❌ MLX Whisper Error: {e}")
            return None


def get_provider(provider_name: str = None) -> TranscriptionProvider:
    """Get a transcription provider by name"""
    provider_name = provider_name or os.environ.get("PROVIDER", "mlx")
    provider_name = provider_name.lower()

    providers = {
        "groq": GroqProvider,
        "mlx": MLXWhisperProvider,
    }

    if provider_name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")

    return providers[provider_name]()

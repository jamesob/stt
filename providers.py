"""
Transcription providers for STT
"""

import atexit
import json
import os
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any


def _wav_to_opus(wav_path: str) -> str | None:
    """Compress WAV to OGG Opus for faster network transfer.

    Returns path to a temp .ogg file, or None if ffmpeg is missing.
    Caller is responsible for deleting the temp file.
    """
    if not shutil.which("ffmpeg"):
        return None
    fd, ogg_path = tempfile.mkstemp(suffix=".ogg")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-c:a", "libopus", "-b:a", "24k",
                "-application", "voip",
                "-vn", ogg_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if os.path.getsize(ogg_path) > 0:
            return ogg_path
    except Exception:
        pass
    try:
        os.unlink(ogg_path)
    except OSError:
        pass
    return None


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

    def warmup(self, quiet: bool = False) -> None:
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

        print("Transcribing...")

        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        ogg_path = _wav_to_opus(audio_file_path)
        send_path = ogg_path or audio_file_path
        fname = "audio.ogg" if ogg_path else "audio.wav"
        mime = "audio/ogg" if ogg_path else "audio/wav"

        try:
            with open(send_path, "rb") as audio_file:
                files = {
                    "file": (fname, audio_file, mime)
                }
                data = {
                    "model": "whisper-large-v3",
                    "response_format": "text",
                    "language": language,
                }
                if prompt:
                    data["prompt"] = prompt

                response = requests.post(
                    url, headers=headers, files=files,
                    data=data, timeout=(5, 20),
                )
                response.raise_for_status()
                return response.text.strip()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
        finally:
            if ogg_path:
                try:
                    os.unlink(ogg_path)
                except OSError:
                    pass


class WhisperCPPHTTPProvider(TranscriptionProvider):
    """Whisper.cpp HTTP provider (local server)"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.environ.get("WHISPER_CPP_HTTP_URL", "http://localhost:8080")

    @property
    def name(self) -> str:
        return "Whisper.cpp HTTP"

    def is_available(self) -> bool:
        return self.base_url is not None

    def transcribe(self, audio_file_path: str, language: str, prompt: str = None) -> str | None:
        import requests

        url = f"{self.base_url}/inference"

        with open(audio_file_path, "rb") as audio_file:
            files = {"file": ("audio.wav", audio_file)}
            data = {
            }
            if prompt:
                data["prompt"] = prompt

            try:
                response = requests.post(url, files=files, data=data, timeout=(5, 20))
                response.raise_for_status()
                result = response.json()
                return result.get("text", "").strip() or None
            except requests.exceptions.RequestException as e:
                print(f"❌ HTTP Error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        print(f"Response: {e.response.text}")
                    except Exception:
                        pass
                return None


class OpenAICompatibleProvider(TranscriptionProvider):
    """OpenAI-compatible API provider (local or remote servers)"""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
    ):
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get(
            "OPENAI_WHISPER_MODEL", "whisper-large-v3"
        )

    @property
    def name(self) -> str:
        return f"OpenAI-compat ({self.model})"

    def is_available(self) -> bool:
        return bool(self.base_url)

    def transcribe(
        self, audio_file_path: str, language: str, prompt: str = None
    ) -> str | None:
        import requests

        print("Transcribing...")

        url = f"{self.base_url}/v1/audio/transcriptions"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        ogg_path = _wav_to_opus(audio_file_path)
        send_path = ogg_path or audio_file_path
        fname = "audio.ogg" if ogg_path else "audio.wav"
        mime = "audio/ogg" if ogg_path else "audio/wav"

        try:
            with open(send_path, "rb") as audio_file:
                files = {
                    "file": (fname, audio_file, mime),
                }
                data: dict[str, str] = {
                    "model": self.model,
                    "response_format": "text",
                    "language": language,
                }
                if prompt:
                    data["prompt"] = prompt

                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=(5, 30),
                )
                response.raise_for_status()
                text = response.text.strip()
                # vLLM returns JSON even with response_format=text
                if text.startswith("{"):
                    try:
                        text = json.loads(text)["text"]
                    except (json.JSONDecodeError, KeyError):
                        pass
                return text.strip() if text else None
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    print(f"Response: {e.response.text}")
                except Exception:
                    pass
            return None
        finally:
            if ogg_path:
                try:
                    os.unlink(ogg_path)
                except OSError:
                    pass


class _MLXWorkerClient:
    _WRITE_TIMEOUT_S = 2.0
    _WRITE_LOCK_TIMEOUT_S = 2.0

    def __init__(self, model: str):
        self.model = model
        self._proc: subprocess.Popen[str] | None = None
        self._messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 1

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, startup_timeout_s: int) -> None:
        if self.is_running():
            return

        worker_path = os.path.join(os.path.dirname(__file__), "mlx_worker.py")
        if not os.path.exists(worker_path):
            raise FileNotFoundError(f"Missing MLX worker at {worker_path}")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        proc = subprocess.Popen(
            [sys.executable, "-u", worker_path, "--model", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit stderr to avoid pipe deadlocks
            text=True,
            bufsize=1,
            env=env,
        )

        messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        thread = threading.Thread(target=self._read_stdout, args=(proc, messages), daemon=True)
        thread.start()

        self._proc = proc
        self._messages = messages
        self._reader_thread = thread

        ready = self._wait_for(lambda m: m.get("type") in {"ready", "error"}, timeout_s=startup_timeout_s)
        if not ready:
            raise TimeoutError("MLX worker did not become ready in time")
        if ready.get("type") == "error":
            raise RuntimeError(ready.get("error") or "MLX worker failed to start")

    def stop(self, force: bool = False) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return

        # Close stdin first to signal worker to stop and unblock any writes
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        try:
            if not force and proc.poll() is None:
                # Give process a chance to exit gracefully
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass

            if proc.poll() is None:
                proc.terminate()
                # Close stdout to unblock reader thread before waiting
                try:
                    if proc.stdout:
                        proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        finally:
            # Ensure stdout is closed (may have been closed above, but that's ok)
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass

    def transcribe(
        self,
        audio_file_path: str,
        language: str,
        prompt: str | None,
        timeout_s: int,
    ) -> str:
        if not self.is_running():
            raise RuntimeError("MLX worker is not running")

        with self._lock:
            req_id = self._next_id
            self._next_id += 1

            payload = json.dumps(
                {
                    "type": "transcribe",
                    "id": req_id,
                    "audio_file_path": audio_file_path,
                    "language": language,
                    "prompt": prompt,
                },
                ensure_ascii=False,
            )
            if not self._write_lock.acquire(timeout=self._WRITE_LOCK_TIMEOUT_S):
                raise TimeoutError("Timed out waiting for MLX worker write lock")
            try:
                self._write_line(payload + "\n", timeout_s=self._WRITE_TIMEOUT_S)
            finally:
                self._write_lock.release()

            message = self._wait_for(
                lambda m: m.get("type") == "result" and m.get("id") == req_id,
                timeout_s=timeout_s,
            )
            if not message:
                raise TimeoutError("Timed out waiting for MLX worker response")
            if message.get("type") == "error":
                raise RuntimeError(message.get("error") or "MLX worker exited unexpectedly")
            error = message.get("error")
            if error:
                raise RuntimeError(str(error))
            return str(message.get("text") or "").strip()

    def _write_line(self, line: str, timeout_s: float) -> None:
        assert self._proc is not None
        assert self._proc.stdin is not None
        fd = self._proc.stdin.fileno()
        data = line.encode("utf-8")
        total = 0
        deadline = time.time() + timeout_s
        while total < len(data):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("Timed out writing to MLX worker stdin")
            _, writable, _ = select.select([], [fd], [], remaining)
            if not writable:
                raise TimeoutError("Timed out writing to MLX worker stdin")
            written = os.write(fd, data[total:])
            if written <= 0:
                raise RuntimeError("Failed to write to MLX worker stdin")
            total += written

    def _read_stdout(self, proc: subprocess.Popen[str], messages: "queue.Queue[dict[str, Any]]") -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                messages.put(json.loads(line))
            except json.JSONDecodeError:
                messages.put({"type": "stdout", "line": line})
        messages.put({"type": "eof"})

    def _wait_for(self, predicate, timeout_s: int) -> dict[str, Any] | None:
        deadline = time.time() + timeout_s if timeout_s > 0 else None
        while True:
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
            else:
                remaining = None

            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty:
                return None

            if message.get("type") == "eof":
                return {"type": "error", "error": "MLX worker exited unexpectedly"}

            if predicate(message):
                return message


class _WorkerClient:
    """Generic worker client for subprocess-based transcription."""

    def __init__(self, model: str, worker_script: str):
        self.model = model
        self.worker_script = worker_script
        self._proc: subprocess.Popen[str] | None = None
        self._messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._next_id = 1

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, startup_timeout_s: int) -> None:
        if self.is_running():
            return

        worker_path = os.path.join(os.path.dirname(__file__), self.worker_script)
        if not os.path.exists(worker_path):
            raise FileNotFoundError(f"Missing worker at {worker_path}")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        proc = subprocess.Popen(
            [sys.executable, "-u", worker_path, "--model", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )

        messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        thread = threading.Thread(target=self._read_stdout, args=(proc, messages), daemon=True)
        thread.start()

        self._proc = proc
        self._messages = messages
        self._reader_thread = thread

        ready = self._wait_for(lambda m: m.get("type") in {"ready", "error"}, timeout_s=startup_timeout_s)
        if not ready:
            raise TimeoutError("Worker did not become ready in time")
        if ready.get("type") == "error":
            raise RuntimeError(ready.get("error") or "Worker failed to start")

    def stop(self, force: bool = False) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return

        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        try:
            if not force and proc.poll() is None:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass

            if proc.poll() is None:
                proc.terminate()
                try:
                    if proc.stdout:
                        proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass

    def send_request(self, request: dict[str, Any], timeout_s: int) -> dict[str, Any]:
        if not self.is_running():
            raise RuntimeError("Worker is not running")

        with self._lock:
            req_id = self._next_id
            self._next_id += 1

            request["id"] = req_id
            assert self._proc is not None
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()

            message = self._wait_for(
                lambda m: m.get("type") == "result" and m.get("id") == req_id,
                timeout_s=timeout_s,
            )

            if not message:
                raise TimeoutError("Timed out waiting for worker response")

            return message

    def _read_stdout(self, proc: subprocess.Popen[str], messages: "queue.Queue[dict[str, Any]]") -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                messages.put(json.loads(line))
            except json.JSONDecodeError:
                messages.put({"type": "stdout", "line": line})
        messages.put({"type": "eof"})

    def _wait_for(self, predicate, timeout_s: int) -> dict[str, Any] | None:
        deadline = time.time() + timeout_s if timeout_s > 0 else None
        while True:
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
            else:
                remaining = None

            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty:
                return None

            if message.get("type") == "eof":
                return {"type": "error", "error": "Worker exited unexpectedly"}

            if predicate(message):
                return message


class MLXWhisperProvider(TranscriptionProvider):
    """Local Whisper provider using MLX (Apple Silicon optimized)"""

    # Short names that users can pass as WHISPER_MODEL.
    _SHORT_NAMES = {"large-v3", "large-v3-turbo", "large", "medium", "small", "base", "tiny"}
    # Repos that don't follow the default ``whisper-{name}-mlx`` convention.
    _REPO_OVERRIDES = {
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }
    DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"

    @staticmethod
    def mlx_repo_id(short_name: str) -> str:
        """Map a short model name (e.g. ``large-v3-turbo``) to its HF repo id."""
        override = MLXWhisperProvider._REPO_OVERRIDES.get(short_name)
        if override:
            return override
        return f"mlx-community/whisper-{short_name}-mlx"

    _WORKER_STARTUP_TIMEOUT_S = 1800
    _TRANSCRIBE_TIMEOUT_S = 30

    def __init__(self, model: str = None):
        self.model = model or os.environ.get("WHISPER_MODEL", self.DEFAULT_MODEL)
        # Normalize model name if user provides short form
        if self.model and not self.model.startswith("mlx-community/"):
            if self.model in self._SHORT_NAMES:
                self.model = self.mlx_repo_id(self.model)
        self._mlx_whisper = None
        self._use_worker = True
        self._worker: _MLXWorkerClient | None = None
        self._worker_startup_timeout_s = self._WORKER_STARTUP_TIMEOUT_S
        env_timeout = os.environ.get("WHISPER_TIMEOUT_S")
        if env_timeout:
            try:
                self._transcribe_timeout_s = max(1, int(env_timeout))
            except ValueError:
                self._transcribe_timeout_s = self._TRANSCRIBE_TIMEOUT_S
        else:
            self._transcribe_timeout_s = self._TRANSCRIBE_TIMEOUT_S
        self._cleanup_registered = False
        self._worker_lock = threading.Lock()
        self._last_error: str | None = None
        self._last_error_trace: str | None = None

    @property
    def name(self) -> str:
        return f"MLX Whisper ({self.model.split('/')[-1]})"

    def is_available(self) -> bool:
        try:
            import mlx_whisper
            return True
        except ImportError:
            return False

    def warmup(self, quiet: bool = False) -> None:
        """Pre-load the model at startup (downloads if needed)"""
        model_name = self.model.split('/')[-1]

        if quiet:
            try:
                if self._use_worker:
                    self._ensure_worker()
                else:
                    import mlx_whisper
                    from mlx_whisper.transcribe import ModelHolder
                    import mlx.core as mx
                    ModelHolder.get_model(self.model, mx.float16)
                    self._mlx_whisper = mlx_whisper
            except Exception:
                pass
            return

        from rich.console import Console
        from rich.status import Status
        console = Console()

        if self._use_worker:
            with Status(f"[dim]Loading {model_name}...[/dim]", console=console, spinner="dots"):
                try:
                    self._ensure_worker()
                except Exception as e:
                    console.print(f"[red]✗[/red] Failed to load model: {e}")
                    return
            console.print(f"[green]✓[/green] Model: [cyan]{model_name}[/cyan]")
            return

        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
        import mlx.core as mx

        with Status(f"[dim]Loading {model_name}...[/dim]", console=console, spinner="dots"):
            ModelHolder.get_model(self.model, mx.float16)
            self._mlx_whisper = mlx_whisper
        console.print(f"[green]✓[/green] Model: [cyan]{model_name}[/cyan]")

    def transcribe(self, audio_file_path: str, language: str, prompt: str = None) -> str | None:
        if self._use_worker:
            print("Transcribing...")
            try:
                self._last_error = None
                self._last_error_trace = None
                self._ensure_worker()
                assert self._worker is not None
                return self._worker.transcribe(
                    audio_file_path=audio_file_path,
                    language=language,
                    prompt=prompt,
                    timeout_s=self._transcribe_timeout_s,
                )
            except TimeoutError as e:
                print("❌ MLX transcription timed out. Restarting MLX worker...")
                self._last_error = f"timeout: {e}"
                self._last_error_trace = None
                self._stop_worker(force=True)
                return None
            except Exception as e:
                print(f"❌ MLX Whisper Error: {e}")
                self._last_error = str(e)
                self._last_error_trace = traceback.format_exc()
                self._stop_worker(force=True)
                return None

        if self._mlx_whisper is None:
            try:
                import mlx_whisper
                self._mlx_whisper = mlx_whisper
            except ImportError:
                print("❌ mlx-whisper not installed. Run: pip install mlx-whisper")
                return None

        print("Transcribing...")

        try:
            self._last_error = None
            self._last_error_trace = None
            result = self._mlx_whisper.transcribe(
                audio_file_path,
                path_or_hf_repo=self.model,
                language=language,
                initial_prompt=prompt,
            )
            return result["text"].strip()
        except Exception as e:
            print(f"❌ MLX Whisper Error: {e}")
            self._last_error = str(e)
            self._last_error_trace = traceback.format_exc()
            return None

    def cancel(self) -> None:
        """Best-effort cancellation of an in-flight transcription."""
        if self._use_worker:
            self._stop_worker(force=True)

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is None:
                self._worker = _MLXWorkerClient(model=self.model)
            if not self._worker.is_running():
                self._worker.start(startup_timeout_s=self._worker_startup_timeout_s)
            if not self._cleanup_registered:
                atexit.register(self._shutdown)
                self._cleanup_registered = True

    def _stop_worker(self, force: bool = False) -> None:
        with self._worker_lock:
            worker = self._worker
            if force:
                self._worker = None
        if worker is None:
            return
        worker.stop(force=force)

    def _shutdown(self) -> None:
        self._stop_worker(force=True)


class ParakeetProvider(TranscriptionProvider):
    """Nvidia Parakeet provider using MLX (Apple Silicon optimized, English-only)"""

    DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
    _WORKER_STARTUP_TIMEOUT_S = 1800
    _TRANSCRIBE_TIMEOUT_S = 30

    def __init__(self, model: str = None):
        self.model = model or os.environ.get("PARAKEET_MODEL", self.DEFAULT_MODEL)
        self._worker: _WorkerClient | None = None
        self._worker_startup_timeout_s = self._WORKER_STARTUP_TIMEOUT_S
        env_timeout = os.environ.get("PARAKEET_TIMEOUT_S")
        if env_timeout:
            try:
                self._transcribe_timeout_s = max(1, int(env_timeout))
            except ValueError:
                self._transcribe_timeout_s = self._TRANSCRIBE_TIMEOUT_S
        else:
            self._transcribe_timeout_s = self._TRANSCRIBE_TIMEOUT_S
        self._cleanup_registered = False
        self._worker_lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"Parakeet ({self.model.split('/')[-1]})"

    def is_available(self) -> bool:
        try:
            import parakeet_mlx
            return True
        except ImportError:
            return False

    def warmup(self, quiet: bool = False) -> None:
        """Pre-load the model at startup (downloads if needed)"""
        if quiet:
            try:
                self._ensure_worker()
            except Exception:
                pass
            return

        from rich.console import Console
        from rich.status import Status
        console = Console()
        model_name = self.model.split('/')[-1]

        with Status(f"[dim]Loading {model_name}...[/dim]", console=console, spinner="dots"):
            try:
                self._ensure_worker()
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to load model: {e}")
                return
        console.print(f"[green]✓[/green] Model: [cyan]{model_name}[/cyan]")

    def transcribe(self, audio_file_path: str, language: str, prompt: str = None) -> str | None:
        # Parakeet is English-only
        if language and language.lower() not in ("en", "english"):
            print(f"❌ Parakeet only supports English. Got: {language}")
            return None

        print("Transcribing...")
        try:
            self._ensure_worker()
            assert self._worker is not None
            result = self._worker.send_request(
                {"type": "transcribe", "audio_file_path": audio_file_path},
                timeout_s=self._transcribe_timeout_s,
            )
            error = result.get("error")
            if error:
                raise RuntimeError(str(error))
            text = str(result.get("text") or "").strip()

            # Apply phonetic correction using PROMPT vocabulary
            if prompt and text:
                from postprocess import parse_vocabulary, correct_text
                vocab = parse_vocabulary(prompt)
                if vocab:
                    text = correct_text(text, vocab)

            return text
        except TimeoutError:
            print("❌ Parakeet transcription timed out. Restarting worker...")
            self._stop_worker(force=True)
            return None
        except Exception as e:
            print(f"❌ Parakeet Error: {e}")
            self._stop_worker(force=True)
            return None

    def cancel(self) -> None:
        """Best-effort cancellation of an in-flight transcription."""
        self._stop_worker(force=True)

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is None:
                self._worker = _WorkerClient(model=self.model, worker_script="parakeet_worker.py")
            if not self._worker.is_running():
                self._worker.start(startup_timeout_s=self._worker_startup_timeout_s)
            if not self._cleanup_registered:
                atexit.register(self._shutdown)
                self._cleanup_registered = True

    def _stop_worker(self, force: bool = False) -> None:
        with self._worker_lock:
            worker = self._worker
            if force:
                self._worker = None
        if worker is None:
            return
        worker.stop(force=force)

    def _shutdown(self) -> None:
        self._stop_worker(force=True)


class FallbackProvider(TranscriptionProvider):
    """Try providers in order; fall back on connection failure."""

    def __init__(
        self,
        chain: list[tuple[TranscriptionProvider, float]],
    ):
        # chain: [(provider, connect_timeout_seconds), ...]
        self._chain = chain
        self._warmed: set[int] = set()

    @property
    def name(self) -> str:
        names = [p.name for p, _ in self._chain]
        return " -> ".join(names)

    def is_available(self) -> bool:
        return any(p.is_available() for p, _ in self._chain)

    def warmup(self, quiet: bool = False) -> None:
        # Only warm the primary provider. Fallback providers
        # warm lazily on first use to avoid wasting memory.
        for i, (prov, timeout) in enumerate(self._chain):
            if timeout and hasattr(prov, "base_url"):
                if not quiet:
                    print(f"Probing {prov.name}...")
                if not self._probe(prov.base_url, timeout):
                    if not quiet:
                        print(f"  unreachable, trying next")
                    continue
            if prov.is_available():
                prov.warmup(quiet=quiet)
                self._warmed.add(i)
                return

    def transcribe(
        self,
        audio_file_path: str,
        language: str,
        prompt: str = None,
    ) -> str | None:
        for i, (prov, timeout) in enumerate(self._chain):
            if not prov.is_available():
                continue

            # TCP probe for remote providers
            if timeout and hasattr(prov, "base_url"):
                if not self._probe(prov.base_url, timeout):
                    self._log_skip(i, "unreachable")
                    continue

            # Lazy warmup
            if i not in self._warmed:
                try:
                    prov.warmup()
                    self._warmed.add(i)
                except Exception as e:
                    self._log_skip(
                        i, f"warmup failed: {e}")
                    continue

            result = prov.transcribe(
                audio_file_path, language, prompt)
            if result is not None:
                return result

        return None

    def cancel(self) -> None:
        for prov, _ in self._chain:
            c = getattr(prov, "cancel", None)
            if callable(c):
                try:
                    c()
                except Exception:
                    pass

    def _log_skip(self, idx: int, reason: str):
        name = self._chain[idx][0].name
        print(f"{name}: {reason}")
        if idx < len(self._chain) - 1:
            nxt = self._chain[idx + 1][0].name
            print(f"Falling back to {nxt}...")

    @staticmethod
    def _probe(base_url: str, timeout: float) -> bool:
        """TCP connect probe to check server reachability."""
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (
            443 if parsed.scheme == "https" else 80
        )
        try:
            sock = socket.create_connection(
                (host, port), timeout=timeout)
            sock.close()
            return True
        except (socket.timeout, OSError):
            return False


class BenchmarkProvider(TranscriptionProvider):
    """Wraps a primary provider; runs all others in parallel and logs results."""

    def __init__(
        self,
        primary: TranscriptionProvider,
        others: list[TranscriptionProvider],
    ):
        self._primary = primary
        self._others = others
        self._print_lock = threading.Lock()
        self._warmup_threads: list[threading.Thread] = []

    @property
    def name(self) -> str:
        return f"{self._primary.name} [benchmark]"

    def is_available(self) -> bool:
        return self._primary.is_available()

    def warmup(self, quiet: bool = False) -> None:
        self._primary.warmup(quiet=quiet)
        # Warm others in background threads so we don't block startup
        self._warmup_threads = []
        for prov in self._others:
            t = threading.Thread(
                target=self._safe_warmup, args=(prov,),
                daemon=True,
            )
            t.start()
            self._warmup_threads.append(t)

    def wait_ready(self, timeout: float = 120) -> None:
        """Block until all benchmark providers have finished warmup."""
        for t in self._warmup_threads:
            t.join(timeout=timeout)
        self._warmup_threads.clear()

    def transcribe(
        self, audio_file_path: str, language: str,
        prompt: str = None,
    ) -> str | None:
        results: list[tuple[str, str | None, float, bool]] = []
        lock = threading.Lock()

        def _run(prov: TranscriptionProvider, primary: bool = False):
            t0 = time.monotonic()
            try:
                text = prov.transcribe(
                    audio_file_path, language, prompt)
            except Exception as e:
                text = f"[error: {e}]"
            elapsed = time.monotonic() - t0
            with lock:
                results.append((prov.name, text, elapsed, primary))

        # Launch others in background
        threads = []
        for prov in self._others:
            if prov.is_available():
                t = threading.Thread(
                    target=_run, args=(prov,), daemon=True)
                t.start()
                threads.append(t)

        # Run primary in foreground
        _run(self._primary, primary=True)

        # Wait for all benchmark threads before printing
        for t in threads:
            t.join(timeout=30)

        # Print all results together, primary first
        with self._print_lock:
            for name, text, elapsed, primary in sorted(
                results, key=lambda r: (not r[3], r[2])
            ):
                self._log(name, text, elapsed, primary=primary)

        # Return primary result
        for name, text, elapsed, primary in results:
            if primary:
                return text
        return None

    def cancel(self) -> None:
        c = getattr(self._primary, "cancel", None)
        if callable(c):
            c()
        for prov in self._others:
            c = getattr(prov, "cancel", None)
            if callable(c):
                try:
                    c()
                except Exception:
                    pass

    @staticmethod
    def _safe_warmup(prov: TranscriptionProvider):
        try:
            prov.warmup(quiet=True)
        except Exception as e:
            print(f"[benchmark] {prov.name} warmup failed: {e}")

    @staticmethod
    def _log(
        name: str, text: str | None, elapsed: float,
        primary: bool = False,
    ):
        from rich.console import Console
        console = Console(highlight=False)

        tag = ">>>" if primary else "   "
        sec = f"{elapsed:.2f}s"
        content = (text or "").strip()
        label = f"[bold]{name}[/bold]" if primary else name

        console.print(
            f"[dim]\\[benchmark][/dim] "
            f"[yellow]{sec:>7s}[/yellow] "
            f"{tag} {label}"
        )
        if content:
            console.print(
                f"          [dim]{content}[/dim]"
            )



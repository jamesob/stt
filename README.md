# STT

Hold a key, speak, release -- your words appear wherever your cursor is.

**Like SuperWhisper, but free. Like Wispr Flow, but local.**

![Demo](demo.gif)

- **Cross-platform** -- macOS (Apple Silicon) and Linux (Wayland: Sway, Hyprland, etc.)
- **Any backend** -- local MLX Whisper, any OpenAI-compatible server, whisper.cpp HTTP, or Groq cloud
- **Hold-to-record** -- global hotkey works in any application
- **Free & open source** -- no subscription, no cloud dependency required

## Install

```bash
uv tool install git+https://github.com/jamesob/stt.git
```

A setup wizard runs on first launch. To update:

```bash
uv tool install --reinstall git+https://github.com/jamesob/stt.git
```

### Linux dependencies

STT checks for missing dependencies at startup and prints install commands. For reference:

**Arch Linux:**
```bash
sudo pacman -S wtype wl-clipboard gtk4-layer-shell \
    gobject-introspection portaudio pipewire-pulse
```

**Debian / Ubuntu:**
```bash
sudo apt install wtype wl-clipboard gtk4-layer-shell-dev \
    libgirepository1.0-dev gir1.2-gtk-4.0 gir1.2-gtk4layershell-1.0 \
    libportaudio2 portaudio19-dev pipewire-pulse
```

Your user must be in the `input` group for keyboard capture:

```bash
sudo usermod -aG input $USER
newgrp input  # or log out and back in
```

### macOS permissions

Grant **Accessibility** and **Input Monitoring** (System Settings > Privacy & Security) to your terminal app -- not to "stt".

## Usage

```bash
stt
```

| Action | Keys |
|--------|------|
| Record | Hold trigger key (default: Right Cmd / Left Alt) |
| Record + Enter | Hold **Shift** while recording |
| Cancel | **ESC** |
| Quit | **Ctrl+C** |

## Configuration

Settings live in `~/.config/stt/config.yml`. Run `stt --config` to reconfigure. See [`config.sample.yml`](config.sample.yml) for all options.

```yaml
language: en
hotkey: cmd_r
sound_enabled: true

active_profile: default

profiles:
  default:
    provider: openai
    openai_base_url: http://localhost:8000
    openai_whisper_model: whisper-large-v3
```

### Providers

| Provider | Profile keys | Notes |
|----------|-------------|-------|
| `openai` | `openai_base_url`, `openai_api_key`, `openai_whisper_model` | Any OpenAI-compatible server (vLLM, faster-whisper, etc.) |
| `whisper-cpp-http` | `whisper_cpp_http_url` | Local whisper.cpp HTTP server |
| `mlx` | `whisper_model` | Apple Silicon, offline |
| `parakeet` | `parakeet_model` | Apple Silicon, English only, very fast |
| `groq` | `groq_api_key` | Cloud, requires [API key](https://console.groq.com) |

### Fallback chains

Profiles can be chained with automatic failover. If a server is unreachable, STT falls back to the next provider:

```yaml
profiles:
  qwen:
    provider: openai
    openai_base_url: http://gpu-server:8200
    openai_whisper_model: Qwen/Qwen3-ASR-1.7B
  local:
    provider: mlx
    whisper_model: large-v3-turbo
  auto:
    fallback:
      - qwen:
          connect_timeout: 2
      - local
```

**Benchmark mode** runs additional providers in parallel and logs timing for comparison:

```yaml
  bench:
    fallback:
      - qwen:
          connect_timeout: 2
      - local
    benchmark:
      - whisper
```

### Prompt tuning

The `prompt` setting helps Whisper recognize domain-specific terms:

```yaml
prompt: Claude, Anthropic, TypeScript, React, API endpoint
```

## Development

```bash
git clone https://github.com/jamesob/stt.git
cd stt
uv sync
uv run stt
```

## License

MIT

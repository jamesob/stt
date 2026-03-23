# STT

Fork of [bokan/stt](https://github.com/bokan/stt) with multi-backend benchmarking, profile-based fallback, and Linux improvements.

**Like SuperWhisper, but free. Like Wispr Flow, but local.**

Hold a key, speak, release -- your words appear wherever your cursor is. Built for vibe coding and conversations with AI agents.

![Demo](demo.gif)

- **Free & open source** -- no subscription, no cloud dependency
- **Runs locally** on Apple Silicon via MLX Whisper or Parakeet
- **Or use cloud** (Groq) or any OpenAI-compatible server
- **Cross-platform** -- macOS and Wayland Linux (Sway, Hyprland)
- **One command install** -- `uv tool install git+https://github.com/jamesob/stt.git`

## Features

- **Global hotkey** -- works in any application, configurable trigger key
- **Hold-to-record** -- no start/stop buttons, just hold and speak
- **Auto-type** -- transcribed text is typed directly into the active field
- **Shift+record** -- automatically sends Enter after typing (great for chat interfaces)
- **Audio feedback** -- subtle system sounds confirm recording state (can be disabled)
- **Silence detection** -- automatically skips transcription when no speech detected
- **Slash commands** -- say "slash help" to type `/help`
- **Context prompts** -- improve accuracy with domain-specific vocabulary
- **Auto-updates** -- notifies when a new version is available

## Requirements

### macOS

- Apple Silicon (M1/M2/M3/M4)
- [UV](https://docs.astral.sh/uv/) package manager
- **For cloud mode (optional):** [Groq API key](https://console.groq.com)

### Linux (Wayland)

- A wlroots-based compositor (Sway, Hyprland, etc.)
- [UV](https://docs.astral.sh/uv/) package manager
- System packages (see [install commands](#linux-system-dependencies) below)
- User must be in the `input` group for keyboard capture
- A transcription server (OpenAI-compatible endpoint, Whisper.cpp HTTP, or Groq cloud)

## Installation

```bash
uv tool install git+https://github.com/jamesob/stt.git
```

On first run, a setup wizard will guide you through configuration.

To update:

```bash
uv tool install --reinstall git+https://github.com/jamesob/stt.git
```

### Linux system dependencies

Text injection requires `wtype` and `wl-clipboard`. The recording overlay needs GTK4 and `gtk4-layer-shell`. Audio capture uses PortAudio via the `sounddevice` Python package.

#### Arch Linux

```bash
sudo pacman -S wtype wl-clipboard gtk4-layer-shell \
    gobject-introspection portaudio pipewire-pulse
```

#### Debian / Ubuntu

```bash
sudo apt install wtype wl-clipboard gtk4-layer-shell-dev \
    libgirepository1.0-dev gir1.2-gtk-4.0 gir1.2-gtk4layershell-1.0 \
    libportaudio2 portaudio19-dev pipewire-pulse
```

#### Keyboard capture (all distros)

STT uses evdev to capture the hotkey globally. Your user must be in the `input` group:

```bash
sudo usermod -aG input $USER
newgrp input  # or log out and back in
```

## Permissions

### macOS

STT needs macOS permissions to capture the global hotkey and type text into other apps.

Grant these to **your terminal app** (iTerm2, Terminal, Warp, etc.) -- not "stt":

- **Accessibility** -- System Settings > Privacy & Security > Accessibility
- **Input Monitoring** -- System Settings > Privacy & Security > Input Monitoring

### Linux

STT uses evdev for keyboard capture, which requires membership in the `input` group. Text injection uses `wtype` and `wl-clipboard`, which work without special permissions on Wayland.

## Usage

```bash
stt
```

| Action | Keys |
|--------|------|
| Record | Hold **Right Command** (default) |
| Record + Enter | Hold **Shift** while recording |
| Cancel recording / stuck transcription | **ESC** |
| Quit | **Ctrl+C** |

## Configuration

Settings are stored in `~/.config/stt/config.yml`. Run `stt --config` to reconfigure, or edit directly. See [`config.sample.yml`](config.sample.yml) for all options.

Provider configuration lives under named **profiles**. General settings stay at the top level:

```yaml
language: en
hotkey: cmd_r
sound_enabled: true
# prompt: Claude, Anthropic, React

active_profile: default

profiles:
  default:
    provider: openai
    openai_base_url: http://localhost:8000
    openai_whisper_model: whisper-large-v3
```

Old flat-key configs are automatically migrated to a `default` profile on first load.

### Providers

| Provider | Profile keys | Notes |
|----------|-------------|-------|
| `mlx` | `whisper_model` | macOS default, Apple Silicon, offline |
| `parakeet` | `parakeet_model` | macOS, English only, very fast |
| `openai` | `openai_base_url`, `openai_api_key`, `openai_whisper_model` | Linux default, any OpenAI-compatible server |
| `whisper-cpp-http` | `whisper_cpp_http_url` | Local whisper.cpp server |
| `groq` | `groq_api_key` | Cloud, requires [API key](https://console.groq.com) |

### Fallback and Benchmark

Multiple profiles can be chained with automatic fallback. If a remote server is unreachable, STT falls back to the next provider:

```yaml
active_profile: auto

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

**Benchmark mode** runs additional providers in parallel and logs timing + text for comparison, while returning results from the primary:

```yaml
  bench:
    fallback:
      - qwen:
          connect_timeout: 2
      - local
    benchmark:
      - whisper
```

### Prompt examples

The `prompt` setting helps Whisper recognize domain-specific terms. Parakeet uses it for phonetic post-correction.

```yaml
# Programming
prompt: Claude, Anthropic, TypeScript, React, useState, API endpoint

# AI tools
prompt: Claude Code, WezTerm, Groq, LLM
```

## Prompt Overlay (Optional)

STT includes a prompt overlay (triggered by Right Option by default) for quickly pasting common prompts.

Prompts live in `~/.config/stt/prompts/*.md`.

## Development

```bash
git clone https://github.com/jamesob/stt.git
cd stt
uv sync
uv run stt
```

## License

MIT

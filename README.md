# STT

**Like SuperWhisper, but free. Like Wispr Flow, but local.**

Hold a key, speak, release -- your words appear wherever your cursor is. Built for vibe coding and conversations with AI agents.

![Demo](demo.gif)

- **Free & open source** -- no subscription, no cloud dependency
- **Runs locally** on Apple Silicon via MLX Whisper or Parakeet
- **Or use cloud** (Groq) or any OpenAI-compatible server
- **Cross-platform** -- macOS and Wayland Linux (Sway, Hyprland)
- **One command install** -- `uv tool install git+https://github.com/bokan/stt.git`

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
- System packages: `gtk4-layer-shell`, `wtype`, `wl-clipboard`, `libgirepository1.0-dev`, `gir1.2-gtk-4.0`
- User must be in the `input` group for keyboard capture: `sudo usermod -aG input $USER && newgrp input`
- A transcription server (OpenAI-compatible endpoint, Whisper.cpp HTTP, or Groq cloud)

## Installation

```bash
uv tool install git+https://github.com/bokan/stt.git
```

On first run, a setup wizard will guide you through configuration.

To update:

```bash
uv tool install --reinstall git+https://github.com/bokan/stt.git
```

### Linux system dependencies

```bash
# Debian/Ubuntu
sudo apt install gtk4-layer-shell wtype wl-clipboard \
    libgirepository1.0-dev gir1.2-gtk-4.0 gir1.2-gtk4layershell-1.0 \
    pulseaudio-utils

# Arch
sudo pacman -S gtk4-layer-shell wtype wl-clipboard

# Add yourself to the input group (required for keyboard capture)
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

Settings are stored in `~/.config/stt/.env`. Run `stt --config` to reconfigure, or edit directly:

```bash
# Transcription provider: "mlx" (macOS default), "openai" (Linux default),
#   "whisper-cpp-http", "parakeet", or "groq"
PROVIDER=mlx

# OpenAI-compatible server (default provider on Linux)
OPENAI_BASE_URL=http://localhost:8000
OPENAI_API_KEY=           # optional
OPENAI_WHISPER_MODEL=whisper-large-v3

# Local HTTP server URL (default: http://localhost:8080)
WHISPER_CPP_HTTP_URL=http://localhost:8080

# Required for cloud mode only
GROQ_API_KEY=gsk_...

# Audio device (saved automatically after first selection; device name, not index)
AUDIO_DEVICE=MacBook Pro Microphone

# Language code for transcription
LANGUAGE=en

# Trigger key: cmd_r, cmd_l, alt_r, alt_l, ctrl_r, ctrl_l, shift_r
HOTKEY=cmd_r

# Context prompt to improve accuracy for specific terms
PROMPT=Claude, Anthropic, TypeScript, React, Python

# Disable audio feedback sounds
SOUND_ENABLED=true
```

## Prompt Overlay (Optional)

STT includes a prompt overlay (triggered by Right Option by default) for quickly pasting common prompts.

Prompts live in:

`~/.config/stt/prompts/*.md`

### Local Mode (MLX Whisper) — Default

Local transcription uses Apple Silicon GPU acceleration via MLX. On first run, the Whisper large-v3 model (~3GB) will be downloaded and cached. Subsequent runs load from cache.

Runs completely offline — no API key required. Supports 99 languages and context prompts.

### Local Mode (Parakeet)

Nvidia's Parakeet model via MLX. Faster than Whisper (~3000x realtime factor) with comparable accuracy.

```bash
PROVIDER=parakeet
```

On first run, the model (~2.5GB) will be downloaded and cached.

**Limitations:**
- English only

**Phonetic correction:** While Parakeet doesn't support Whisper-style prompts, it uses the `PROMPT` setting for phonetic post-processing. Terms like `Claude Code, WezTerm` will correct sound-alike ASR errors (e.g., "cloud code" → "Claude Code", "Vez term" → "WezTerm").

### Cloud Mode (Groq)

To use cloud transcription instead:

```bash
PROVIDER=groq
GROQ_API_KEY=gsk_...
```

Requires a [Groq API key](https://console.groq.com) (free tier available).

### HTTP Mode (Local Server)

Run a local HTTP server with Whisper transcription. Useful for performance or custom integration.

```bash
PROVIDER=whisper-cpp-http
WHISPER_CPP_HTTP_URL=http://localhost:8080
```

**Start the server:**

```bash
# Terminal 1: Start the whisper.cpp server
./whisper-server -m models/ggml-large-v3.bin -f

# Or run in background with a custom port
./whisper-server -m models/ggml-large-v3.bin -f -t 4 -ngl 32 -p 8080
```

The server provides a whisper.cpp-compatible endpoint:

```bash
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "language=en"
```

**Benefits:**
- Fast HTTP API for integrating with other services
- Reuse whisper.cpp model across multiple applications
- Hardware accelerated on CPU/NVIDIA
- Configurable temperature, model, and decoding options

### OpenAI-compatible Mode

Works with any server that implements the OpenAI `/v1/audio/transcriptions` API (e.g., faster-whisper-server, whisper.cpp with OpenAI compat, LocalAI).

```bash
PROVIDER=openai
OPENAI_BASE_URL=http://localhost:8000
OPENAI_WHISPER_MODEL=whisper-large-v3
# OPENAI_API_KEY=sk-...  # optional, if your server requires auth
```

This is the default provider on Linux.

### Prompt Examples

The `PROMPT` setting helps Whisper recognize domain-specific terms:

```bash
# Programming
PROMPT=TypeScript, React, useState, async await, API endpoint

# AI tools
PROMPT=Claude, Anthropic, OpenAI, Groq, LLM, GPT
```

## Development

```bash
git clone https://github.com/bokan/stt.git
cd stt
uv sync
uv run stt
```

## License

MIT

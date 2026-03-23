"""Prompt configuration for STT.

Loads custom prompts from ~/.config/stt/prompts/*.md files.
Falls back to built-in defaults if directory empty/missing.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import shutil

PROMPTS_DIR = Path.home() / ".config" / "stt" / "prompts"
OLD_PROMPTS_DIR = Path.home() / "go" / "src" / "github.com" / "jamesob" / "dotfiles" / "stt-prompts"

DEFAULT_ICON = "•"


@dataclass
class PromptItem:
    key: str
    label: str
    text: str
    icon: str = DEFAULT_ICON
    enter: bool = False


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content.

    Returns (metadata_dict, body_text).
    """
    pattern = r'^---\s*\n(.*?)\n---\s*\n?(.*)$'
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return {}, content

    frontmatter_text = match.group(1)
    body = match.group(2)

    # Simple YAML parsing (key: value pairs only)
    metadata = {}
    for line in frontmatter_text.split('\n'):
        line = line.strip()
        if ':' in line:
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip()
            # Remove quotes if present
            if value and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            metadata[key] = value

    return metadata, body


def _default_prompts() -> list[PromptItem]:
    """Built-in default prompts."""
    return [
        PromptItem(key="1", icon="⚡", label="Fix this", text="Fix the following code:\n\n"),
        PromptItem(key="2", icon="📝", label="Explain", text="Explain this code in detail:\n\n"),
        PromptItem(key="3", icon="🔄", label="Refactor", text="Refactor this code for better readability:\n\n"),
        PromptItem(key="4", icon="🧪", label="Add tests", text="Write unit tests for:\n\n"),
        PromptItem(key="5", icon="📖", label="Document", text="Add documentation comments to:\n\n"),
    ]

def _migrate_prompts_if_needed() -> None:
    """One-time copy from old dotfiles path if new prompts dir is empty."""
    try:
        if PROMPTS_DIR.exists():
            existing = list(PROMPTS_DIR.glob("*.md"))
            if existing:
                return
        if not OLD_PROMPTS_DIR.exists():
            return
        old_files = list(OLD_PROMPTS_DIR.glob("*.md"))
        if not old_files:
            return
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        for src in old_files:
            dst = PROMPTS_DIR / src.name
            if dst.exists():
                continue
            shutil.copy2(src, dst)
    except Exception:
        # Migration is best-effort; fall back to defaults.
        return


def ensure_default_prompts() -> None:
    """Create prompts dir and default .md files if dir empty/missing."""
    _migrate_prompts_if_needed()
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if dir has any .md files
    existing = list(PROMPTS_DIR.glob("*.md"))
    if existing:
        return

    # Write defaults
    for prompt in _default_prompts():
        filename = f"{prompt.key}_{prompt.label.lower().replace(' ', '_')}.md"
        filepath = PROMPTS_DIR / filename

        content = f"""---
key: {prompt.key}
label: {prompt.label}
icon: {prompt.icon}
---
{prompt.text}"""

        filepath.write_text(content)


def load_prompts() -> list[PromptItem]:
    """Load prompts from PROMPTS_DIR.

    Returns list of PromptItem sorted by key.
    Falls back to defaults if dir missing/empty.
    """
    _migrate_prompts_if_needed()
    if not PROMPTS_DIR.exists():
        return _default_prompts()

    md_files = list(PROMPTS_DIR.glob("*.md"))
    if not md_files:
        return _default_prompts()

    prompts = []
    for filepath in md_files:
        content = filepath.read_text()
        metadata, body = _parse_frontmatter(content)

        # Key required - use filename stem as fallback
        key = metadata.get("key", filepath.stem)

        # Label defaults to filename
        label = metadata.get("label", filepath.stem.replace("_", " ").title())

        # Icon optional; normalize to non-empty string.
        icon = (metadata.get("icon") or "").strip()
        if not icon or icon.lower() == "none":
            icon = DEFAULT_ICON

        # Enter flag - triggers on enter key
        enter = metadata.get("enter", "").lower() == "true"

        prompts.append(PromptItem(
            key=str(key),
            label=label,
            text=body or "",
            icon=icon,
            enter=enter,
        ))

    # Sort by key
    prompts.sort(key=lambda p: p.key)
    return prompts

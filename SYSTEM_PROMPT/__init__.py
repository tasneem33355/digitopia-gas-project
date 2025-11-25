"""Loader for the system prompt used by the chatbot.

This module reads the full prompt from `prompt.txt` (same folder) so the
prompt text can be edited without modifying Python code. It exposes a
`system_prompt` string plus small helpers for quick verification.
"""

from pathlib import Path

__all__ = ["system_prompt", "get_system_prompt_preview", "print_system_prompt_preview"]

_PROMPT_PATH = Path(__file__).with_name("prompt.txt")


def _read_prompt() -> str:
    try:
        text = _PROMPT_PATH.read_text(encoding="utf-8")
        # If the prompt file was accidentally wrapped in Markdown fences
        # (``` or ```plaintext), remove the first/last fence lines.
        if text.lstrip().startswith("```"):
            parts = text.splitlines()
            # drop the first fence line
            parts = parts[1:]
            # drop trailing fence line if present
            if parts and parts[-1].strip().startswith("```"):
                parts = parts[:-1]
            text = "\n".join(parts)
        return text.strip()
    except Exception:
        # Minimal fallback to avoid import-time failures
        return (
            "You are a SCADA diagnostic assistant. Return concise JSON diagnosis "
            "for Pump Speed Overshoot, Valve Blockage, Compressor Instability, "
            "or System Overload. Prioritize safety."
        )


system_prompt: str = _read_prompt()


def get_system_prompt_preview(n: int = 500) -> str:
    """Return the first `n` characters of the system prompt."""

    return system_prompt[:n]


def print_system_prompt_preview(n: int = 500) -> None:
    """Print a short preview of the system prompt to stdout."""

    print(get_system_prompt_preview(n))
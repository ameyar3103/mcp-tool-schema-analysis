"""Token counting. Estimates drive admission decisions; measured API counts drive results."""

from __future__ import annotations

import functools

# Anthropic publishes no tokenizer, so one public BPE serves as the yardstick and
# each ModelSpec carries an empirically calibrated scale factor off it.
_REFERENCE = "Qwen/Qwen2.5-7B"
_FALLBACK_CHARS_PER_TOKEN = 3.7


@functools.lru_cache(maxsize=1)
def _tokenizer():
    """Load once. Returns None when offline so planning degrades instead of failing."""
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_pretrained(_REFERENCE)
    except Exception:  # noqa: BLE001 - hub errors are many and none are recoverable here
        return None


def count(text: str) -> int:
    """Reference-tokenizer length of one string."""
    tok = _tokenizer()
    if tok is None:
        return round(len(text) / _FALLBACK_CHARS_PER_TOKEN)
    return len(tok.encode(text, add_special_tokens=False).ids)


def estimate(text: str, scale: float = 1.0) -> int:
    """Reference length rescaled to a target model's tokenizer."""
    return round(count(text) * scale)

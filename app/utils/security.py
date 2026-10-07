"""Layer 1 prompt-injection defense: reject suspicious phrases before the LLM call."""
import re

SUSPICIOUS_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"system\s+prompt",
    r"you\s+are\s+now",
    r"forget\s+(all\s+)?(previous|prior|your)\s+instructions",
    r"reveal\s+(your|the)\s+(prompt|instructions)",
    r"</?\s*external_data\s*>",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]


def is_suspicious(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)

"""Deterministic input/output guardrails.

Design note: NVIDIA NeMo Guardrails was evaluated for this project, but its
dialogue rails require a live LLM backend, so they cannot run in this
project's offline test suite or in environments without an LLM API key.
These rails are pure-Python, dependency-free, and fully unit-testable:

- **input rail:** blocks prompt-injection / jailbreak attempts
- **input rail:** redacts PII before anything reaches the LLM (PII isolation)
- **output rail:** redacts any PII that appears in retrieved context or in
  the generated answer

If you later add an LLM-backed rail layer (e.g. NeMo), keep these
deterministic rails as the first line of defence — they are cheap, fast,
and have zero false-negative risk on the patterns they cover.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Prompt-injection / jailbreak patterns (input rail)
# ---------------------------------------------------------------------------
INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"reveal\s+(your\s+)?(system|initial)\s+prompt", re.I),
    re.compile(r"show\s+me\s+your\s+(system\s+)?instructions", re.I),
    re.compile(r"print\s+your\s+(system\s+)?prompt", re.I),
    re.compile(r"\bdan\s+mode\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"you\s+are\s+now\s+(dan|unrestricted|unfiltered|evil)", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(unrestricted|unfiltered|not\s+bound)", re.I),
    re.compile(r"override\s+your\s+(safety|content)\s+(filters|guardrails|policy)", re.I),
    re.compile(r"bypass\s+(your\s+)?(safety|content)\s+(filters|filter|policy)", re.I),
    re.compile(r"developer\s+mode\s+(enabled|on)\b", re.I),
)

# ---------------------------------------------------------------------------
# PII patterns (input + output rails)
# ---------------------------------------------------------------------------
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # 13–19 digits with optional separators (phone numbers are checked first
    # and only have 10 digits, so they never match here).
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
}

INPUT_REFUSAL = (
    "I can't help with that request — it looks like an attempt to override "
    "my instructions. Please ask a question about the indexed documents."
)


def contains_injection(text: str) -> bool:
    """Return True if *text* matches a known prompt-injection pattern."""
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def find_pii(text: str) -> list[str]:
    """Return the kinds of PII found in *text* (e.g. ``["email"]``)."""
    return [kind for kind, pattern in PII_PATTERNS.items() if pattern.search(text)]


def redact_pii(text: str) -> str:
    """Replace PII in *text* with typed placeholders like ``[REDACTED:email]``."""
    redacted = text
    for kind, pattern in PII_PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED:{kind}]", redacted)
    return redacted


@dataclass
class RailDecision:
    """Outcome of the input rails."""

    allowed: bool
    refusal: str | None = None
    safe_question: str | None = None


def apply_input_rails(question: str) -> RailDecision:
    """Run the input rails on a user question.

    - Prompt-injection attempts are blocked with a refusal.
    - PII is redacted (isolation) and the question is allowed through.
    """
    if contains_injection(question):
        return RailDecision(allowed=False, refusal=INPUT_REFUSAL)
    return RailDecision(allowed=True, safe_question=redact_pii(question))


def apply_output_rails(answer: str) -> str:
    """Run the output rails: redact any PII leaking into the final answer."""
    return redact_pii(answer)

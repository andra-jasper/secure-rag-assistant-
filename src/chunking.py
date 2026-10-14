"""Overlapping text chunking.

A small dependency-free splitter used before embedding: the text is first
broken on paragraph / line / word boundaries into pieces no longer than
``chunk_size``, then a sliding window re-slices the joined text so that
consecutive chunks share exactly ``chunk_overlap`` characters of context.

Trade-off vs. fancier splitters: chunk boundaries may cut mid-sentence.
For embedding + retrieval this is acceptable and keeps the behaviour
exact and easy to test (every chunk is <= ``chunk_size`` and the overlap
between neighbours is exactly ``chunk_overlap`` characters).
"""
from __future__ import annotations


def _split_recursive(
    text: str, chunk_size: int, separators: tuple[str, ...]
) -> list[str]:
    """Recursively split *text* into pieces each no longer than *chunk_size*."""
    for i, sep in enumerate(separators):
        if sep == "":
            # Last resort: hard character split.
            return [
                text[j : j + chunk_size] for j in range(0, len(text), chunk_size)
            ]
        if sep in text:
            pieces: list[str] = []
            current = ""
            rest = separators[i + 1 :]
            for part in text.split(sep):
                candidate = part if not current else current + sep + part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        pieces.append(current)
                        current = ""
                    # Oversized part: split it with the remaining separators.
                    if len(part) > chunk_size:
                        pieces.extend(_split_recursive(part, chunk_size, rest))
                    else:
                        current = part
            if current:
                pieces.append(current)
            return pieces
    return [text]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: tuple[str, ...] = ("\n\n", "\n", " ", ""),
) -> list[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* chars.

    Returns a list of chunk strings. Empty/whitespace-only input yields [].
    Raises ValueError if ``chunk_overlap`` >= ``chunk_size``.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []

    joined = "\n\n".join(_split_recursive(text, chunk_size, separators))

    chunks: list[str] = []
    start = 0
    while start < len(joined):
        end = min(start + chunk_size, len(joined))
        chunks.append(joined[start:end])
        if end == len(joined):
            break
        start = end - chunk_overlap
    return chunks

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    chunk_index: int


def chunk_text(text: str, chunk_size: int = 2200, overlap: int = 300) -> list[TextChunk]:
    """Split page text into overlapping, whitespace-aligned character chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(TextChunk(normalized[start:end].strip(), len(chunks)))
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks

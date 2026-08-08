import pytest

from rag.chunking import chunk_text


def test_empty_text_has_no_chunks():
    assert chunk_text("  \n ") == []


def test_short_text_is_normalized_into_one_chunk():
    chunks = chunk_text("Tighten   the bolt\ncarefully.", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "Tighten the bolt carefully."
    assert chunks[0].chunk_index == 0


def test_long_text_is_split_with_overlap():
    chunks = chunk_text("one two three four five six", chunk_size=15, overlap=4)
    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)


def test_invalid_chunk_configuration_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("manual", chunk_size=10, overlap=10)

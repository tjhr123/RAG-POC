from rag.retrieval import format_context


def test_format_context_adds_source_labels_and_metadata():
    context, sources = format_context(
        [
            {
                "text": "Use the specified lifting points.",
                "metadata": {"manual": "Workshop Manual", "page": 12, "chunk": 0},
                "distance": 0.2,
            }
        ]
    )
    assert "[Source 1: Workshop Manual, PDF page 12]" in context
    assert sources == [{"manual": "Workshop Manual", "page": 12, "chunk": 0}]

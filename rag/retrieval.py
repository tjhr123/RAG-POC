from pathlib import Path

COLLECTION_NAME = "service_manuals"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ManualRetriever:
    def __init__(self, database_path: Path | str):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(database_path))
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def count(self) -> int:
        return self.collection.count()

    def search(self, question: str, result_count: int = 5) -> list[dict]:
        embedding = self.model.encode([question], normalize_embeddings=True)[0].tolist()
        result = self.collection.query(
            query_embeddings=[embedding], n_results=min(result_count, self.count())
        )
        return [
            {"text": document, "metadata": metadata, "distance": distance}
            for document, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]


def format_context(matches: list[dict]) -> tuple[str, list[dict]]:
    excerpts = []
    sources = []
    for position, match in enumerate(matches, 1):
        metadata = match["metadata"]
        excerpts.append(
            f"[Source {position}: {metadata['manual']}, PDF page {metadata['page']}]\n"
            f"{match['text']}"
        )
        sources.append(
            {
                "manual": metadata["manual"],
                "page": metadata["page"],
                "chunk": metadata["chunk"],
            }
        )
    return "\n\n".join(excerpts), sources

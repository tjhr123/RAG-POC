import argparse
import hashlib
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from rag.chunking import chunk_text
from rag.retrieval import COLLECTION_NAME, EMBEDDING_MODEL


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_pdf(path: Path, collection, model: SentenceTransformer) -> tuple[int, int]:
    digest = file_digest(path)
    existing = collection.get(where={"document_hash": digest}, limit=1)
    if existing["ids"]:
        print(f"Skipping unchanged manual: {path.name}")
        return 0, 0

    reader = PdfReader(path)
    records = []
    empty_pages = 0
    for page_number, page in enumerate(reader.pages, 1):
        chunks = chunk_text(page.extract_text() or "")
        if not chunks:
            empty_pages += 1
            continue
        for chunk in chunks:
            records.append(
                {
                    "id": f"{digest}:p{page_number}:c{chunk.chunk_index}",
                    "text": chunk.text,
                    "metadata": {
                        "document_hash": digest,
                        "manual": path.stem,
                        "file_name": path.name,
                        "page": page_number,
                        "chunk": chunk.chunk_index,
                    },
                }
            )

    if not records:
        print(f"WARNING: no text found in {path.name}; it may require OCR.")
        return 0, empty_pages

    print(f"Embedding {len(records)} chunks from {path.name}...")
    embeddings = model.encode(
        [record["text"] for record in records],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    for start in range(0, len(records), 500):
        batch = records[start : start + 500]
        collection.upsert(
            ids=[record["id"] for record in batch],
            documents=[record["text"] for record in batch],
            metadatas=[record["metadata"] for record in batch],
            embeddings=embeddings[start : start + len(batch)].tolist(),
        )
    return len(records), empty_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF service manuals for semantic search.")
    parser.add_argument("--input", type=Path, default=Path("data/manuals"))
    parser.add_argument("--database", type=Path, default=Path("vector_store/chroma"))
    args = parser.parse_args()

    pdfs = sorted(args.input.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {args.input}")

    collection = chromadb.PersistentClient(path=str(args.database)).get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    model = SentenceTransformer(EMBEDDING_MODEL)
    total_chunks = 0
    total_empty_pages = 0
    for pdf in pdfs:
        chunks, empty_pages = ingest_pdf(pdf, collection, model)
        total_chunks += chunks
        total_empty_pages += empty_pages
    print(
        f"Done. Added {total_chunks} chunks. "
        f"Pages without extractable text: {total_empty_pages}."
    )


if __name__ == "__main__":
    main()

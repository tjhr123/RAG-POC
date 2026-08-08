import hashlib
from pathlib import Path

from rag.chunking import chunk_text
from rag.retrieval import COLLECTION_NAME, EMBEDDING_MODEL


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_pdf(path: Path, collection, model) -> tuple[int, int]:
    from pypdf import PdfReader

    digest = file_digest(path)
    existing = collection.get(where={"document_hash": digest}, limit=1)
    if existing["ids"]:
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
        return 0, empty_pages

    embeddings = model.encode(
        [record["text"] for record in records],
        batch_size=32,
        show_progress_bar=False,
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


def index_directory(manuals_path: Path, database_path: Path) -> dict:
    import chromadb
    from sentence_transformers import SentenceTransformer

    manuals_path = manuals_path.expanduser().resolve()
    database_path = database_path.expanduser().resolve()
    database_path.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(manuals_path.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files were found in {manuals_path}")

    collection = chromadb.PersistentClient(path=str(database_path)).get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    model = SentenceTransformer(EMBEDDING_MODEL)
    total_chunks = 0
    total_empty_pages = 0
    for pdf in pdfs:
        chunks, empty_pages = ingest_pdf(pdf, collection, model)
        total_chunks += chunks
        total_empty_pages += empty_pages
    return {
        "manuals": len(pdfs),
        "new_chunks": total_chunks,
        "empty_pages": total_empty_pages,
        "total_chunks": collection.count(),
    }

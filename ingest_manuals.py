import argparse
from pathlib import Path

from rag.indexing import index_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF service manuals for semantic search.")
    parser.add_argument("--input", type=Path, default=Path("data/manuals"))
    parser.add_argument("--database", type=Path, default=Path("vector_store/chroma"))
    args = parser.parse_args()

    result = index_directory(args.input, args.database)
    print(
        f"Done. Found {result['manuals']} manual(s), added {result['new_chunks']} chunks, "
        f"and found {result['empty_pages']} pages without extractable text. "
        f"The index now contains {result['total_chunks']} chunks."
    )


if __name__ == "__main__":
    main()

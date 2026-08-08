# Service Manual Assistant

This Streamlit prototype retrieves relevant pages from PDF workshop manuals and gives those
excerpts to Llama through Groq. Answers include the manual name and PDF page used as evidence.

## First-time setup

Use Python 3.11 or 3.12; some local embedding and vector-database packages do not yet support
newer Python releases.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` (do not commit it):

```text
GROQ_API_KEY=your_key_here
```

## Index manuals

Put permitted PDF manuals in `data/manuals/`, then run:

```bash
python ingest_manuals.py
```

The generated Chroma database is stored under `vector_store/chroma/`. Re-running the command
skips a manual when the same file content has already been indexed. A high count of pages without
extractable text indicates that the PDF probably needs OCR.

The app also shows a **Build manual index** button when the database is empty. This is useful on
Streamlit Community Cloud, where generated files are not included in Git and the local filesystem
can be replaced whenever the app is redeployed or rebooted. If that happens, use the button again.

## Start the chat

```bash
streamlit run GeminiApp.py
```

Paths are resolved from the application directory rather than the process's current directory, so
the same setup works when Streamlit launches the app from another working directory. Set
`CHROMA_PATH` or `MANUALS_PATH` only when you intentionally keep these files elsewhere.

The first indexing or search run downloads the local embedding model. The Groq API is used only
to compose the final answer from retrieved excerpts.

> This is a prototype reference tool. Mechanics must verify safety-critical procedures and
> specifications against the official manual before carrying out work.

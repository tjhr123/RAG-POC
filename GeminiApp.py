import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag.retrieval import ManualRetriever, format_context


load_dotenv()
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Service Manual Assistant", page_icon="🔧", layout="wide")
st.title("🔧 Service Manual Assistant")
st.caption("Answers grounded in the indexed workshop manuals")

api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not api_key:
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Groq API Key", type="password")
        if not api_key:
            st.warning("Provide a Groq API key to continue.")
            st.stop()

client = Groq(api_key=api_key)

st.sidebar.header("Manual search")
result_count = st.sidebar.slider("Sources to retrieve", 2, 8, 5)
st.sidebar.info(
    "Run `python ingest_manuals.py` after adding or changing a PDF. "
    "The assistant will say when the indexed sources do not contain an answer."
)

database_path = Path(os.getenv("CHROMA_PATH", APP_DIR / "vector_store" / "chroma"))
manuals_path = Path(os.getenv("MANUALS_PATH", APP_DIR / "data" / "manuals"))
try:
    retriever = ManualRetriever(database_path)
    indexed_chunks = retriever.count()
except Exception as error:
    st.error(
        f"Could not open the manual index at `{database_path}` "
        f"({type(error).__name__}: {error})."
    )
    st.code("python ingest_manuals.py", language="bash")
    st.info("On Streamlit Cloud, reboot the app after confirming the command finishes.")
    st.stop()

if indexed_chunks == 0:
    st.warning(
        "The manual database is empty. Index the PDFs once before asking questions."
    )
    available_pdfs = sorted(manuals_path.glob("*.pdf"))
    st.write(f"Found **{len(available_pdfs)} PDF manual(s)** in `{manuals_path}`.")
    if st.button("Build manual index", type="primary", disabled=not available_pdfs):
        from rag.indexing import index_directory

        with st.spinner("Reading the manual and creating embeddings. This can take several minutes..."):
            try:
                result = index_directory(manuals_path, database_path)
            except Exception as error:
                st.error(f"Indexing failed ({type(error).__name__}: {error}).")
            else:
                st.success(
                    f"Index ready: {result['total_chunks']:,} chunks from "
                    f"{result['manuals']} manual(s)."
                )
                st.rerun()
    if not available_pdfs:
        st.error(f"Add at least one PDF to `{manuals_path}` and redeploy the app.")
    st.stop()

st.sidebar.success(f"{indexed_chunks:,} manual chunks indexed")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['manual']}**, PDF page {source['page']} "
                        f"(chunk {source['chunk']})"
                    )

if prompt := st.chat_input("Ask a question about the service manual..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            matches = retriever.search(prompt, result_count)
            context, sources = format_context(matches)
            recent_chat = [
                {"role": item["role"], "content": item["content"]}
                for item in st.session_state.messages[-6:]
            ]
            request_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an automotive service-manual assistant for trained mechanics. "
                        "Answer only from the supplied manual excerpts. Never invent a procedure, "
                        "specification, part number, warning, or torque value. Preserve units exactly. "
                        "Mention relevant warnings. If the excerpts are insufficient, clearly say so. "
                        "Cite claims inline using the provided labels, for example [Source 1]."
                    ),
                },
                *recent_chat,
                {
                    "role": "user",
                    "content": f"Service-manual excerpts:\n\n{context}\n\nQuestion: {prompt}",
                },
            ]

            response_box = st.empty()
            full_response = ""
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=request_messages,
                temperature=0.1,
                stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    full_response += text
                    response_box.markdown(full_response + "▌")
            response_box.markdown(full_response)

            with st.expander("Sources used"):
                for source in sources:
                    st.markdown(
                        f"- **{source['manual']}**, PDF page {source['page']} "
                        f"(chunk {source['chunk']})"
                    )
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "sources": sources}
            )
        except Exception as error:
            st.error(f"Could not generate an answer: {error}")

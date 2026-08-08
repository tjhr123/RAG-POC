import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag.retrieval import ManualRetriever, format_context


load_dotenv()

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

database_path = Path(os.getenv("CHROMA_PATH", "vector_store/chroma"))
try:
    retriever = ManualRetriever(database_path)
    indexed_chunks = retriever.count()
except Exception as error:
    st.error(f"Could not open the manual index: {error}")
    st.stop()

if indexed_chunks == 0:
    st.warning(
        "No manual pages have been indexed yet. Run `python ingest_manuals.py` "
        "from the project directory, then restart this app."
    )
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

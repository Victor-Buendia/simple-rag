import pymupdf4llm
import pymupdf
import streamlit as st
from yaml import safe_load
from rag import generate_embeddings, query_topk_embeddings_indices
from log import logger

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_vault" not in st.session_state:
    st.session_state.rag_vault = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

config = safe_load(open("./config.yaml"))


def upload_rag():
    uploaded_file = st.file_uploader(
        label="Upload a file to generate RAG embeddings", type=["pdf"],
    )
    if uploaded_file:
        document = pymupdf.Document(
            filename=uploaded_file, stream=uploaded_file.getvalue()
        )
        md_text = pymupdf4llm.to_markdown(document)
        for chunk in split_text(md_text):
            st.session_state.rag_vault[chunk] = generate_embeddings(
                text=chunk, model=config["ollama"]["embeddings_model"]
            )

        st.success("Uploaded successfully!")
        st.session_state.uploaded_files.append(uploaded_file)
        st.balloons()

    if st.session_state["uploaded_files"]:
        st.markdown("## RAG Vault")
        for doc in st.session_state.uploaded_files:
            st.write(doc.name)


def split_text(text: str, chunk_size=250, overlap_size=50):
    words = text.split()
    for i in range(overlap_size, len(words), chunk_size):
        yield " ".join(words[(i - overlap_size) : (i + chunk_size + overlap_size)])


pg = st.navigation(
    [
        st.Page("chat.py", title="Chat", icon="💬"),
        st.Page(upload_rag, title="RAG Vault", icon="🔒"),
    ]
)
pg.run()

import streamlit as st
from application.rag import (
    generate_embeddings,
    generate_text_chunks,
)
from interfaces.pymupdf import extract_markdown_from_pdf
from config import embeddings_model

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_vault" not in st.session_state:
    st.session_state.rag_vault = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def upload_rag():
    uploaded_file = st.file_uploader(
        label="Upload a file to generate RAG embeddings",
        type=["pdf"],
    )
    if uploaded_file:
        md_text = extract_markdown_from_pdf(uploaded_file=uploaded_file)
        text_chunks = generate_text_chunks(text=md_text)

        for chunk in text_chunks:
            embeddings = generate_embeddings(text=chunk, model=embeddings_model)
            st.session_state.rag_vault[chunk] = embeddings

        st.session_state.uploaded_files.append(uploaded_file)
        st.success("Uploaded successfully!")
        st.balloons()

    if st.session_state["uploaded_files"]:
        st.markdown("## RAG Vault")
        for doc in st.session_state.uploaded_files:
            st.write(doc.name)


pg = st.navigation(
    [
        st.Page("application/chat/chat.py", title="Chat", icon="💬"),
        st.Page(upload_rag, title="RAG Vault", icon="🔒"),
    ]
)
pg.run()

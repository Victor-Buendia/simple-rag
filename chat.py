import torch
import ollama

from log import logger
from rag import generate_embeddings, query_topk_embeddings_indices
from yaml import safe_load
import streamlit as st


def stream_ollama(stream):
    for chunk in stream:
        yield chunk["message"]["content"]


avatars = {
    "user": "👨🏽‍💻",
    "assistant": "🦙",
}
headers = {
    "user": "**User**: ",
    "assistant": "**Assistant**: ",
}


config = safe_load(open("./config.yaml"))

st.title("Llama 3")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatars[message["role"]]):
        st.markdown(headers[message["role"]] + message["content"])

if prompt := st.chat_input("Enter your question..."):
    user_prompt_embeddings = generate_embeddings(
        text=prompt, model=config["ollama"]["embeddings_model"]
    )
    vault_embeddings_tensor = list(st.session_state.rag_vault.values())
    relevant_rag_documents_indexes = query_topk_embeddings_indices(
        input_embeddings_tensor=user_prompt_embeddings,
        vault_embeddings_tensor=vault_embeddings_tensor,
        topk=3,
    )
    relevant_rag_documents = [
        list(st.session_state.rag_vault.keys())[i]
        for i in relevant_rag_documents_indexes
    ]
    docs = "\n\n".join(relevant_rag_documents)
    logger.info(f"Relevant RAG documents: \n{docs}")

    user_message = {"role": "user", "content": prompt}
    if relevant_rag_documents:
        user_message["rag_docs"] = f"\nUse this content to generate a response: {docs}"

    st.chat_message("user", avatar=avatars["user"]).markdown("**User:** " + prompt)
    st.session_state.messages.append(user_message)

    logger.debug(f"User message: {user_message}")

    response = ollama.chat(
        model=config["ollama"]["chat_model"],
        messages=[
            {"role": chat["role"], "content": chat["content"] + chat.get("rag_docs", "")}
            for chat in st.session_state.messages
        ],
        stream=True,
    )
    with st.chat_message("assistant", avatar=avatars["assistant"]):
        complete_answer = ""
        answer_placeholder = st.empty()
        for chunk in stream_ollama(response):
            complete_answer += chunk
            answer_placeholder.markdown("**Assistant:** " + complete_answer)
    st.session_state.messages.append({"role": "assistant", "content": complete_answer})

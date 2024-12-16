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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatars[message["role"]]):
        st.markdown(headers[message["role"]] + message["content"])

if prompt := st.chat_input("Enter your question..."):
    user_message = {"role": "user", "content": prompt}
    st.chat_message("user", avatar=avatars["user"]).markdown("**User:** " + prompt)
    st.session_state.messages.append(user_message)
    response = ollama.chat(
        model=config["ollama"]["chat_model"],
        messages=st.session_state.messages,
        stream=True,
    )
    with st.chat_message("assistant", avatar=avatars["assistant"]):
        complete_answer = ""
        answer_placeholder = st.empty()
        for chunk in stream_ollama(response):
            complete_answer += chunk
            answer_placeholder.markdown("**Assistant:** " + complete_answer)
    st.session_state.messages.append(
        {"role": "assistant", "content": complete_answer}
    )

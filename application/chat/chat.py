import ollama
import streamlit as st

from interfaces.ollama import ollama_client, stream_ollama
from interfaces.log import logger
from application.rag import generate_embeddings, query_topk_embeddings_indices
from application.chat import avatars, headers
from config import embeddings_model, chat_model


def load_chat_history():
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role, avatar=avatars[role]):
            st.markdown(headers[role] + content)


st.title("Llama 3")
load_chat_history()

if prompt := st.chat_input("Enter your question..."):
    user_prompt_embeddings = generate_embeddings(text=prompt, model=embeddings_model)
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

    response = ollama_client.chat(
        model=chat_model,
        messages=[
            {
                "role": chat["role"],
                "content": chat["content"] + chat.get("rag_docs", ""),
            }
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

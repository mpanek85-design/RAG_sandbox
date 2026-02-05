from os import getenv
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from rag_utils import load_documents, rag_stream, llm_stream

load_dotenv()


def init_page():
    st.set_page_config(page_title="RAG evaluation sandbox", 
                    layout="wide", 
                    initial_sidebar_state="expanded")


    st.html("<h1 style='text-align: center;'>RAG evaluation sandbox</h1>")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello there!"}
        ]

    if "context_precision" not in st.session_state:
        st.session_state.context_precision = 0.0

    if "reranker" not in st.session_state:
        st.session_state.reranker = False

    if "retriever_k" not in st.session_state:
        st.session_state.retriever_k = 5

    if "retriever_lambda_mult" not in st.session_state:
        st.session_state.retriever_lambda_mult = 0.2

#### Main logic ####

init_page()

model = init_chat_model(getenv("LLM_MODEL"))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        if not st.session_state.rag:
            st.write_stream(llm_stream(model, st.session_state.messages[-1]['content']))
        else:
            st.write_stream(rag_stream(model, st.session_state.messages[-1]['content']))

with st.sidebar:
    st.file_uploader(
            "Upload PDF file(s)", 
            type=["pdf"],
            accept_multiple_files=True,
            on_change=load_documents,
            key="uploaded_files",
        )
    cols0 = st.columns(1)
    with cols0[0]:
        is_vector_store_loaded = ("vector_store" in st.session_state and st.session_state.vector_store is not None)
        st.toggle(
            "Use RAG", 
            value=is_vector_store_loaded, 
            key="rag", 
            disabled=not is_vector_store_loaded,
        )
        st.toggle(
            "Use Reranker",
            value=st.session_state.reranker,
            key="reranker",
            disabled=not st.session_state.rag,
        )
    
    st.divider()
    if st.session_state.rag:
        st.subheader("Retriever Settings")
        st.slider(
            "Number of documents (k)",
            min_value=1,
            max_value=20,
            value=st.session_state.retriever_k,
            key="retriever_k",
        )
        st.slider(
            "Diversity factor (lambda_mult)",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            value=st.session_state.retriever_lambda_mult,
            key="retriever_lambda_mult",
        )
    
    st.divider()
    st.metric("Context Precision", f"{st.session_state.context_precision:.4f}")




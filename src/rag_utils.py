from os import getenv, makedirs

import openai
import streamlit as st
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import LLMContextPrecisionWithoutReference


def join_retrieved_content(content):
    return "\n\n".join(doc.page_content for doc in content)

def llm_stream(model, messages):
    response_message = ""

    for chunk in model.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


def stream_llm_rag_response_old(llm_stream, message):
    rag_chain = get_rag_chain(llm_stream)
    response_message = rag_chain.invoke(message)

    for chunk in rag_chain.stream(message):
        response_message += chunk
        yield chunk
    st.session_state.messages.append({"role": "assistant", 
                                      "content": response_message})


def rag_stream(model, message):
    rag_chain = get_rag_chain(model)
    
    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": st.session_state.retriever_k, "lambda_mult": st.session_state.retriever_lambda_mult})
    retrieved_docs = retriever.invoke(message)
    retrieved_strs = [doc.page_content for doc in retrieved_docs]
    
    response_message = ""
    for chunk in rag_chain.stream(message):
        response_message += chunk
        yield chunk
    
    # Calculate context utilization after response is generated
    calculate_context_precision(retrieved_strs, message, response_message)

def calculate_context_precision(retrieved_strs, message, response_message):
    try:
        evaluator_llm = llm_factory("gpt-4o-mini", client=openai.OpenAI())
        context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
        sample = SingleTurnSample(
            user_input=message,
            response=response_message,
            retrieved_contexts=retrieved_strs,
        )
        context_precision_score = context_precision.single_turn_score(sample)
    except Exception as e:
        st.warning(f"Could not calculate context precision: {e}")
        context_precision_score = None

    st.session_state.context_precision = context_precision_score
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_message,
        "context_precision": context_precision_score
    })

def load_documents():

    if not st.session_state.uploaded_files:
        return
    documents=[]

    for uploaded_file in st.session_state.uploaded_files:
        makedirs("source_files", exist_ok=True)
        file_path = f"./source_files/{uploaded_file.name}"
        with open(file_path, "wb") as file:
            file.write(uploaded_file.read()) 
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
        # st.session_state.uploaded_files.append(file_path)

    if documents:
        upload_to_vector_store(documents)


def upload_to_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=1000,
        )
    document_chunks = text_splitter.split_documents(documents)

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_vectore_store(documents)
    else:
        st.session_state.vector_store.add_documents(document_chunks)

def create_vectore_store(documents):
    embeddings = OpenAIEmbeddings(model=getenv("EMBEDDINGS_MODEL"))
    return InMemoryVectorStore.from_documents(documents, embeddings)



def get_rag_chain(model):
    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt")

    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": st.session_state.retriever_k, 
                                          "lambda_mult": st.session_state.retriever_lambda_mult})
    
    if st.session_state.reranker:
        compressor = FlashrankRerank()
        retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return (
        {
            "context": retriever | join_retrieved_content,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
        )
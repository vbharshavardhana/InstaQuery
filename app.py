# Required imports
import streamlit as st
import google.generativeai as genai
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
import uvicorn

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# FastAPI app setup
fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Retrieves a retriever object by processing the text data from a given URL
# This involves loading the webpage, splitting it into chunks, and creating a BM25Retriever for information retrieval.
def get_retriever_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    retriever = BM25Retriever.from_documents(document_chunks)
    return retriever


# Creates a context-aware retriever chain to enhance information retrieval
# Uses a custom GeminiLLM and ChatPromptTemplate to account for chat history when responding to user queries.
def get_context_retriever_chain(retriever):
    class GeminiLLM:
        def __call__(self, input, **kwargs):
            chat_history = kwargs.get("chat_history", "")
            full_input = f"Given the conversation history: {chat_history}\nUser Query: {input}"
            response = model.generate_content(full_input)
            return response.text

    gemini_llm = GeminiLLM()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    retriever_chain = create_history_aware_retriever(gemini_llm, retriever, prompt)
    return retriever_chain


# Constructs a conversational RAG (Retrieval-Augmented Generation) chain
# Combines retrieval capabilities and document-aware generation for providing contextually enriched responses.
def get_conversational_rag_chain(retriever_chain):
    class GeminiLLM:
        def __call__(self, input, **kwargs):
            chat_history = kwargs.get("chat_history", "")
            context = kwargs.get("context", "")
            full_input = (
                f"Context: {context}\n"
                f"Conversation History: {chat_history}\n"
                f"User Query: {input}"
            )
            response = model.generate_content(full_input)
            return response.text

    gemini_llm = GeminiLLM()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(gemini_llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


# Handles the overall process of generating a response to user input
# Utilizes both retriever chain and conversational RAG chain to provide relevant, context-aware answers.
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.retriever)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']


# FastAPI endpoint for querying the chatbot
# Accepts user input and returns a response after verifying that a retriever has been initialized.
@fastapi_app.post("/query")
async def query(data: dict):
    user_input = data.get("user_input", "")
    if "retriever" not in st.session_state:
        return {"error": "No retriever initialized. Provide a URL first."}
    response = get_response(user_input)
    return {"answer": response}


# Starts a FastAPI server in a separate thread
# Enables handling API requests asynchronously while allowing the main application to run independently.
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)


# Main Streamlit application configuration
# Sets up the UI, manages session states for the retriever, chat history, and handles user input.
st.set_page_config(page_title="InstaQuery", page_icon="🤖")
st.title("InstaQuery")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    if "website_url" not in st.session_state:
        st.session_state.website_url = ""
    website_url = st.text_input("Website URL", st.session_state.website_url)
    if website_url != st.session_state.website_url:
        st.session_state.website_url = website_url
        if "retriever" in st.session_state:
            del st.session_state.retriever

# Main logic for handling chatbot interactions
if st.session_state.website_url.strip() == "":
    st.info("Please enter a website URL in the sidebar.")
else:
    website_url = st.session_state.website_url
    if "retriever" not in st.session_state:
        st.session_state.retriever = get_retriever_from_url(website_url)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="How can I help you?")]

    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

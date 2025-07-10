
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

st.set_page_config(page_title="Smart Blockchain Tutor", layout="centered")
st.title("🤖 Suhasini's Smart Learning Assistant (SSLA) – AI Tutor")

st.markdown("Upload your notes and ask questions related to Blockchain, ML, or Python!")

# Sidebar file upload
st.sidebar.header("📚 Upload Notes")
uploaded_file = st.sidebar.file_uploader("Upload a .txt file with your notes", type=["txt"])

if uploaded_file:
    # Save uploaded file temporarily
    with open("notes.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process notes
    loader = TextLoader("notes.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Create vector store from docs
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)

    # QA system setup
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), chain_type="stuff", retriever=db.as_retriever())

    # Ask a question
    st.subheader("🧠 Ask me a question from your uploaded notes:")
    user_question = st.text_input("Enter your question here")

    if user_question:
        response = qa_chain.run(user_question)
        st.success(response)
else:
    st.warning("Please upload your notes in .txt format to begin.")

st.markdown("---")
st.markdown("Made with 💙 by Suhasini using LangChain + ChromaDB + OpenAI")

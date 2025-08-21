import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

# ---------------- PDF Functions ----------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)

def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def build_qa_chain(vector_store_path="faiss_index"):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever()
    llm = Ollama(model="llama3.2")  # Use LLaMA 3.2 via Ollama
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)
    return qa_chain

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot with FAISS + LLaMA 3.2")
st.write("Upload a PDF and then ask questions about its content.")

# Store conversation history
if "history" not in st.session_state:
    st.session_state.history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload your PDF file", type="pdf")

if uploaded_file is not None:
    pdf_path = f"uploaded/{uploaded_file.name}"
    os.makedirs("uploaded", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = extract_text_from_pdf(pdf_path)

    st.info("⚙️ Creating FAISS vector store...")
    create_faiss_vector_store(text)

    st.info("🤖 Initializing chatbot...")
    st.session_state.qa_chain = build_qa_chain()
    st.success("✅ Chatbot is ready! You can now ask questions below.")

# Always show question box
question = st.text_input("💬 Ask a question about the uploaded PDF:")

if question:
    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload a PDF first before asking questions.")
    else:
        st.info("🔎 Querying the document...")
        answer = st.session_state.qa_chain.run(question)

        # Save to history
        st.session_state.history.append({"question": question, "answer": answer})

# Show chat history
if st.session_state.history:
    st.subheader("📝 Conversation History")
    for i, chat in enumerate(st.session_state.history, 1):
        st.markdown(f"**Q{i}:** {chat['question']}")
        st.markdown(f"**A{i}:** {chat['answer']}")
        st.markdown("---")

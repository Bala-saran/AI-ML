import os
import time
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
import google.api_core.exceptions

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Use Gemini model
LLM_MODEL = "gemini-1.5-pro"


# ---------------- PDF Utils ----------------
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF"""
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=12000, overlap=500):
    """Split text into overlapping chunks to avoid token overload"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------- Gemini Query ----------------
def ask_gemini(question, context, retries=3):
    """Send question + PDF text to Gemini with retries"""
    model = genai.GenerativeModel(LLM_MODEL)
    prompt = f"""
You are a helpful assistant. Answer the question using only the provided PDF text.

Question: {question}

PDF Text:
{context}
    """
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except google.api_core.exceptions.ResourceExhausted as e:
            wait_time = 40  # default
            if hasattr(e, "retry_delay") and e.retry_delay.seconds > 0:
                wait_time = e.retry_delay.seconds
            st.warning(f"Quota exceeded. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{retries})")
            time.sleep(wait_time)
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    st.error("Failed after multiple retries. Please check your API quota.")
    return None


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="PDF Q&A with Gemini", page_icon="📄🤖")
st.title("📄 Ask Questions About Your PDF with Gemini")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        pdf_chunks = chunk_text(pdf_text)
    st.success("PDF loaded successfully!")

    question = st.text_input("Ask a question about the PDF:")
    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            answers = []
            for chunk in pdf_chunks:
                answer = ask_gemini(question, chunk)
                if answer:
                    answers.append(answer)

            # Combine answers (basic merge — can improve with ranking)
            final_answer = "\n\n".join(answers) if answers else "No answer found."
        st.markdown("### ✅ Answer")
        st.write(final_answer)
else:
    st.info("Upload a PDF to begin.")

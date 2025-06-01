import streamlit as st
import subprocess
import os
import sqlite3
from rag import RAGSystem
import db_utils
from pdf2image import convert_from_bytes
import pytesseract
import PyPDF2
import io
from datetime import datetime

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the database at startup
try:
    db_utils.init_db()  # Ensure the database and table are created
except Exception as e:
    st.error(f"Failed to initialize the database: {e}")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to fetch available Ollama models
def get_available_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = [
            line.split(" ")[0] for line in result.stdout.strip().split("\n")
            if line and "NAME" not in line and "embed" not in line.lower()
        ]
        return models
    except subprocess.CalledProcessError as e:
        print(f"Error fetching models: {e}")
        return []

# Function to extract text from PDF
def extract_text_from_pdf(pdf_content):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        raise Exception(f"PDF text extraction failed: {e}")

# Function to convert PDF to images for OCR
def convert_pdf_to_images(pdf_content):
    try:
        images = convert_from_bytes(pdf_content)
        return images
    except Exception as e:
        raise Exception(f"PDF to image conversion failed: {e}")

# Function to extract text from images using OCR
def extract_text_from_images(images):
    try:
        text = ""
        for img in images:
            img = img.convert('L')  # Convert to grayscale
            text += pytesseract.image_to_string(img, config='--psm 6') + "\n"
        return text
    except Exception as e:
        raise Exception(f"OCR failed: {e}")

# Function to chunk text
def chunk_text(text, chunk_size=1024):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Sidebar for document upload and settings
with st.sidebar:
    st.header("💬 NALCO Chatbot")
    st.markdown("A document-based chatbot for NALCO, developed by Sumit Kumar.")

    # LLM Provider Selection
    llm_provider = st.selectbox("Select LLM Provider", ["Ollama", "Sambanova"], index=0)

    # Model Selection based on Provider
    if llm_provider == "Ollama":
        available_models = get_available_models()
        if not available_models:
            st.error("No installed Ollama models found. Please install one using `ollama pull <model_name>`.")
            st.stop()
        selected_model = st.selectbox("Select Ollama Model", available_models, index=0)
        llm_name = None
        api_key = None
    else:
        llm_name = st.selectbox("Select Sambanova Model", ["QwQ-32B", "DeepSeek-R1-Distill-Llama-70B"], index=0)
        api_key = st.text_input("Enter OpenRouter API Key", type="password")
        selected_model = None

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

    # Clear database buttons (for debugging)
    if st.button("Clear ChromaDB Collection"):
        try:
            rag_system = RAGSystem(collection_name="pdf_content", db_path="./PDF_ChromaDB")
            rag_system.delete_collection()
            st.success("ChromaDB collection cleared. Please re-upload documents.")
        except Exception as e:
            st.error(f"Failed to clear ChromaDB collection: {e}")

    if st.button("Clear Stored Documents"):
        try:
            if os.path.exists("nalco_chatbot.db"):
                os.remove("nalco_chatbot.db")
                st.success("Stored documents cleared. Please re-upload documents.")
                # Reinitialize the database after clearing
                db_utils.init_db()
            else:
                st.info("No SQLite database found to clear.")
        except Exception as e:
            st.error(f"Failed to clear SQLite database: {e}")

    # Display stored documents with timestamp
    documents = db_utils.load_documents_from_db()
    if documents:
        st.subheader("Stored Documents")
        for doc in documents:
            file_name = doc[0]
            timestamp = doc[3]
            st.write(f"{file_name} (Uploaded on {timestamp})")
    else:
        st.info("No documents uploaded yet. Please upload a PDF or image.")

# Document processing
chunk_size = 1024
if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            file_name = uploaded_file.name
            file_content = uploaded_file.read()

            # Extract text based on file type
            if file_name.lower().endswith(".pdf"):
                try:
                    extracted_text = extract_text_from_pdf(file_content)
                except Exception as e:
                    st.warning(f"Simple extraction failed: {e}. Using OCR...")
                    images = convert_pdf_to_images(file_content)
                    extracted_text = extract_text_from_images(images)
            elif file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                extracted_text = extract_text_from_images([file_content])
            else:
                st.error("Unsupported file type. Please upload a PDF or image.")
                st.stop()

            if not extracted_text.strip():
                st.error("No text could be extracted from the document.")
                st.stop()

            # Store document in SQLite and ChromaDB
            db_utils.store_document(file_name, extracted_text)
            rag_system = RAGSystem(collection_name="pdf_content", db_path="./PDF_ChromaDB")
            chunks = chunk_text(extracted_text, chunk_size=chunk_size)
            
            # Generate embeddings and store in ChromaDB with metadata
            embeddings = []
            for i, chunk in enumerate(chunks):
                embedding = rag_system._generate_embeddings(chunk)
                if not embedding:
                    st.error(f"Failed to generate embedding for chunk {i}: {chunk[:50]}...")
                    st.stop()
                embeddings.append(embedding)
            
            rag_system.collection.add(
                documents=chunks,
                ids=[f"{file_name}_{i}" for i in range(len(chunks))],
                embeddings=embeddings,
                metadatas=[{"file_name": file_name} for _ in range(len(chunks))]
            )
            
            # Clear file content to save space
            file_content = None
            uploaded_file = None

            st.success(f"Document '{file_name}' uploaded and processed successfully!")
        except Exception as e:
            st.error(f"Document processing failed: {e}")
            st.stop()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about the document"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            rag_system = RAGSystem(collection_name="pdf_content", db_path="./PDF_ChromaDB")
            if llm_provider == "Ollama":
                llm_response, time, docs_nbrs, input_token_count, output_token_count = rag_system.generate_response(
                    query.strip(), selected_model
                )
            else:
                llm_response, time, docs_nbrs, input_token_count, output_token_count = rag_system.generate_response2(
                    query.strip(), llm_name, api_key=api_key
                )
            st.markdown(llm_response)
            st.markdown(f"----\nLLM Name: {selected_model or llm_name} | Response Time: {time} | "
                        f"Input Tokens Count: {input_token_count} | Output Tokens Count: {output_token_count} | "
                        f"Number of Retrieved Documents: {docs_nbrs}")
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
        except Exception as e:
            st.error(f"Query processing failed: {e}")
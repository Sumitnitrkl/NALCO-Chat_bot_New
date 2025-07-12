# app.py

import streamlit as st
import subprocess
import os
import sqlite3
from rag import RAGSystem
import db_utils
import pypdfium2 as pdfium
import io
from datetime import datetime
from PIL import Image
import re
import pytesseract
from ocr_utils import extract_text_from_image, extract_text_from_scanned_pdf
import traceback

# --- NEW: Function for intelligent metadata extraction ---
def extract_metadata_from_chunk(chunk: str) -> dict:
    """
    Analyzes a text chunk to extract structured metadata.
    This is the "thinking" part of the ingestion process.
    """
    metadata = {}
    
    # 1. Extract Page Number
    page_match = re.search(r'Page (\d+)', chunk)
    metadata['page_number'] = page_match.group(1) if page_match else "N/A"
    
    # 2. Extract Section Number (e.g., 5.4, 5.7.1)
    # This looks for a number pattern at the start of the chunk.
    section_match = re.search(r'^(\d+(?:\.\d+)+)', chunk.strip())
    metadata['section_number'] = section_match.group(1) if section_match else "N/A"
    
    # 3. Extract Section Title (assumes the first line is the title)
    lines = chunk.strip().split('\n')
    if lines:
        # Clean up the title by removing the section number if it's there
        title = re.sub(r'^(\d+(?:\.\d+)+)\s*', '', lines[0]).strip()
        metadata['section_title'] = title
    else:
        metadata['section_title'] = "N/A"
        
    return metadata

# --- The rest of the setup is the same ---
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    version = pytesseract.get_tesseract_version()
except Exception as e:
    st.error(f"Tesseract not found or misconfigured: {e}.")
    st.stop()
try:
    db_utils.init_db()
except Exception as e:
    st.error(f"Failed to initialize the database: {e}")
    st.stop()
if "messages" not in st.session_state: st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem(collection_name="pdf_content", db_path="./PDF_ChromaDB")
def get_available_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True, shell=True)
        return [line.split()[0] for line in result.stdout.strip().split("\n") if line and "NAME" not in line and "embed" not in line.lower()]
    except Exception: return []

# --- Sidebar UI is the same ---
with st.sidebar:
    st.header("💬 NALCO Chatbot")
    # ... (all sidebar UI code remains unchanged)
    llm_provider = st.selectbox("Select LLM Provider", ["Ollama", "Sambanova"], index=0)
    if llm_provider == "Ollama":
        available_models = get_available_models()
        selected_model = st.selectbox("Select Ollama Model", available_models, index=0 if available_models else -1) if available_models else None
    else:
        llm_name = st.selectbox("Select Sambanova Model", ["QwQ-32B", "DeepSeek-R1-Distill-Llama-70B"], index=0)
        api_key = st.text_input("Enter OpenRouter API Key", type="password")
        selected_model = None
    uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
    st.subheader("Upload Status")
    status_placeholder = st.empty()
    if st.button("Clear All Data"):
        try:
            st.session_state.rag_system.delete_collection()
            if os.path.exists("nalco_chatbot.db"): os.remove("nalco_chatbot.db")
            db_utils.init_db()
            status_placeholder.success("All data cleared and reset.")
            st.rerun()
        except Exception as e: status_placeholder.error(f"Failed to clear data: {e}")
    st.subheader("Stored Documents")
    documents = db_utils.load_documents_from_db()
    if documents:
        for doc in documents: st.write(f"📄 {doc[0]}")
    else: st.info("No documents uploaded yet.")

# --- MODIFIED: Document processing now uses the "thinking" function ---
if uploaded_file:
    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            file_name = uploaded_file.name
            if file_name in [doc[0] for doc in db_utils.load_documents_from_db()]:
                status_placeholder.warning(f"'{file_name}' already exists.")
                st.stop()
            
            # Text extraction (no changes here)
            status_placeholder.info("1/4 - Reading document...")
            file_content = uploaded_file.read()
            if file_name.lower().endswith(".pdf"):
                pdf = pdfium.PdfDocument(io.BytesIO(file_content))
                extracted_text = "\n".join([f"Page {i+1}\n{page.get_textpage().get_text_range()}" for i, page in enumerate(pdf)])
            else: # Image
                doc = extract_text_from_image(file_content)
                extracted_text = doc.page_content if doc else ""
            if not extracted_text.strip():
                status_placeholder.error("Failed to extract text.")
                st.stop()

            # Chunking (no changes here, uses the semantic chunker in rag.py)
            status_placeholder.info("2/4 - Structuring content...")
            chunks = st.session_state.rag_system.chunk_text(extracted_text)
            
            # Storing and Indexing with rich metadata
            status_placeholder.info("3/4 - Indexing document (thinking)...")
            doc_chunks, doc_ids, doc_embeddings, doc_metadatas = [], [], [], []
            for i, chunk in enumerate(chunks):
                # --- THIS IS THE KEY CHANGE ---
                # The system "thinks" about each chunk and extracts metadata
                metadata = extract_metadata_from_chunk(chunk)
                metadata['file_name'] = file_name # Add file name to each chunk's metadata
                
                embedding = st.session_state.rag_system._generate_embeddings(chunk)
                if not embedding: continue
                
                doc_chunks.append(chunk)
                doc_ids.append(f"{file_name}_chunk_{i}")
                doc_embeddings.append(embedding)
                doc_metadatas.append(metadata) # Append the rich metadata dictionary
            
            status_placeholder.info("4/4 - Saving to database...")
            if doc_chunks:
                st.session_state.rag_system.collection.add(
                    embeddings=doc_embeddings, documents=doc_chunks, metadatas=doc_metadatas, ids=doc_ids
                )
                # Store only filename and timestamp in SQLite for listing
                db_utils.store_document(file_name, "", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(chunks))
                status_placeholder.success(f"✅ '{file_name}' understood and indexed!")
            
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(traceback.format_exc())

# --- Chat interface is the same ---
st.header("Chat with your Documents")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question..."):
    # ... (all chat logic code remains unchanged)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                rag_system = st.session_state.rag_system
                if llm_provider == "Ollama":
                    if not selected_model: st.error("Please select an Ollama model.")
                    else: llm_response, time, docs_nbrs, _, _ = rag_system.generate_response(query.strip(), selected_model)
                else: 
                    if not api_key: st.error("Please enter your OpenRouter API Key.")
                    else: llm_response, time, docs_nbrs, _, _ = rag_system.generate_response2(query.strip(), llm_name, api_key=api_key)
                
                st.markdown(llm_response)
                if "No documents" not in llm_response and "No relevant" not in llm_response:
                    st.markdown(f"----\n*Retrieved from **{docs_nbrs}** document chunks in **{time}***")
                st.session_state.messages.append({"role": "assistant", "content": llm_response})
            except Exception as e: st.error(f"Query processing failed: {e}")

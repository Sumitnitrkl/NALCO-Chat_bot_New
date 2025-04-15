import os, re, io
import shutil
import logging
import subprocess
from PIL import Image
import streamlit as st
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain.schema import Document 
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from rag import RAGSystem
from md_convertor import Convert2Markdown

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set up logging
logger = logging.getLogger(__name__)
logger.info(f"Streamlit app is running")

# Function to generate PDF
def generate_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 40  # Start position for text
    max_width = width - 80  # Margin for text wrapping

    c.setFont("Helvetica-Bold", 16)
    header_text = "Conversation History"
    text_width = c.stringWidth(header_text, "Helvetica-Bold", 16)
    c.drawString((width - text_width) / 2, height - 40, header_text)

    y -= 30  # Adjust position after header
    c.setFont("Helvetica", 12)

    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "LLM"
        text = f"{role}: {msg['content']}"

        # Separate user questions with a line
        if msg["role"] == "user":
            y -= 10
            c.setStrokeColorRGB(0, 0, 0)
            c.line(40, y, width - 40, y)
            y -= 20  

        # Wrap text within max_width
        wrapped_lines = simpleSplit(text, c._fontname, c._fontsize, max_width)

        for line in wrapped_lines:
            c.drawString(40, y, line)
            y -= 20
            
            # Handle page breaks
            if y < 40:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = height - 40  # Reset position after new page

    c.save()
    buffer.seek(0)
    return buffer

def get_available_models():
    """Fetches the installed Ollama models, excluding 'NAME' and models containing 'embed'."""
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

def remove_tags(text):
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

# Fetch available models
available_models = get_available_models()
if not available_models:
    st.error("No installed Ollama models found. Please install one using `ollama pull <model_name>`.")

image2 = Image.open('imgs/ChatPDF3.png')
st.set_page_config(page_title="Chat with PDF", page_icon=image2)

st.subheader("💬 Chat with PDF :")

with st.sidebar:
    st.header("💬 Chat with your PDF Locally Using Ollama/OpenRouter")
    image = Image.open('imgs/ChatPDF3.png')
    st.image(image)

    # File uploader for PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")

# Initialize session state variables
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None

with st.sidebar:
    # If a new PDF is uploaded, reset session state
    if pdf is not None:
        new_pdf_name = os.path.splitext(pdf.name)[0]
        
        if st.session_state.pdf_name != new_pdf_name:
            st.session_state.pdf_name = new_pdf_name
            st.session_state.processing_complete = False

            # Remove collection from previous vector database
            rag_system2 = RAGSystem(collection_name="pdf_content")
            rag_system2.delete_collection()
            st.success("Previous chat and collection deleted. Please start a new chat.")

    if pdf is not None and not st.session_state.processing_complete:
        markdown_path = f"./tmp/{st.session_state.pdf_name}.md"  # Define output path

        # Choose processing mode
        processing_mode = st.radio("Choose processing mode:", ("Simple Processing", "Advanced Processing"))

        # Button to start processing
        start_button = st.button("Start Processing")

        if start_button:
            # Set processing_complete to True when processing starts
            st.session_state.processing_complete = True
            st.session_state.processing_mode = processing_mode  # Store the selected mode

            text = ""

            with st.spinner("Processing PDF..."):  # Landing spinner
                # Process PDF based on the selected mode
                if st.session_state.processing_mode == "Simple Processing":
                    # Extract text from the uploaded PDF
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                elif st.session_state.processing_mode == "Advanced Processing":
                    # Remove the tmp folder if it exists
                    if os.path.exists("./tmp"):
                        shutil.rmtree("./tmp")

                    # Recreate the tmp folder
                    os.makedirs("./tmp", exist_ok=True)

                    pdf_path = f"./tmp/{pdf.name}"

                    # Save the uploaded file to disk
                    with open(pdf_path, "wb") as f:
                        f.write(pdf.getbuffer())

                    artifact_dict = create_model_dict()
                    converter = PdfConverter(artifact_dict=artifact_dict)
                    convert2md = Convert2Markdown()
                    convert2md.pdf_to_markdown(marker_converter=converter, input_pdf=pdf_path, output_directory="./tmp/")
                    text = convert2md._load_file(markdown_path)

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)

            # Convert chunks into Document objects
            documents = [Document(page_content=chunk) for chunk in chunks]

            with st.spinner("Generating embeddings..."):
                # Create embeddings and a vector database
                vector_db = Chroma.from_documents(
                    documents=documents,
                    embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
                    collection_name="pdf_content",
                    persist_directory="./PDF_ChromaDB",
                )

            st.success("Processing Complete!")

    # User selects the model Provider
    llm_provider = st.selectbox("Select LLM Provider:", ['Ollama', 'Openrouter'], index=0)

    if llm_provider == 'Ollama' :

        selected_model = st.selectbox("Select an Ollama model:", available_models, index=0)

    else : 
        llm_name = st.text_input("Enter LLM Name", value='qwen/qwq-32b:free')
        # openrouter_api_key = st.text_input("Enter Openrouter API Key")

    # Slider to choose the number of retrieved results
    n_results = st.slider("Number of retrieved documents", min_value=1, max_value=15, value=5)

    # Button to download PDF
    if st.button("Download Chat as PDF"):
        pdf_buffer = generate_pdf()
        st.download_button(
            label="Download",
            data=pdf_buffer,
            file_name="chat_history.pdf",
            mime="application/pdf"
        )
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
        
    with st.expander("PDF Proccesing Methods"):
        st.info("""
##### There're 2 method to process PDF after uploading: 
    - Simple Processing : extract the text directly from the pdf if the pdf searchable. (Faster)
    - Advanced Processing : extract the text by converting the pdf to markdown using OCR and then search the markdown file. (Slower)
        
-> The chatbot depends on your performence of your labtop, so please be patient!
""")

# Initialize the RAG system
rag_system = RAGSystem(collection_name="pdf_content", db_path="PDF_ChromaDB", n_results=n_results)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "max_messages" not in st.session_state:
    st.session_state.max_messages = 40  # 20 user + 20 assistant messages

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("----")

# Stop if max messages are reached
if len(st.session_state.messages) >= st.session_state.max_messages:
    st.info("Notice: The maximum message limit has been reached. clear the chat plz!")
else:
    if prompt := st.chat_input("What is up?"):
        if pdf is None or st.session_state.processing_complete != True :
            st.error("Please upload at least one PDF file.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    metadata = None
                    with st.spinner("Thinking..."):

                        if llm_provider == 'Ollama' :
                            # Stream LLM response
                            response_placeholder = st.empty()
                            streamed_response = ""

                            for chunk in rag_system.generate_response(prompt, selected_model):  # Stream response
                                if isinstance(chunk, dict):  # Ensure metadata is correctly assigned
                                    metadata = chunk
                                else:
                                    streamed_response = chunk  # Accumulate response text
                                    response_placeholder.markdown(streamed_response)  # Update UI

                            st.write(f"""\n\n----
                            Token Count: {metadata.get('token_count', 'N/A')} | Response Time: {metadata.get('response_time', 'N/A')} | n_results of context: {metadata.get('n_results', 'N/A')} | LLM Name : {selected_model} """)

                            response = f"""
                            {remove_tags(streamed_response)}
                            \n\n----
                            Token Count: {metadata.get('token_count', 'N/A')} | Response Time: {metadata.get('response_time', 'N/A')} | n_results of context: {metadata.get('n_results', 'N/A')} | LLM Name : {selected_model}
                            """
                        else :
                            llm_response = rag_system.generate_response2(prompt)
                            response = f"""
                            {llm_response}
                            \n----
                            LLM Name : {llm_name}
                            """

                        # Store assistant response
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
                    st.rerun()

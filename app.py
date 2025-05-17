import os, re, io
import shutil
import logging
import subprocess
from PIL import Image
import streamlit as st
# -------------------------------
import pymupdf4llm
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
# -------------------------------
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain.schema import Document 
# -------------------------------
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
# -------------------------------
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
    return re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()

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

    # chunk_size for splittig
    chunk_size = st.number_input('chunk size :', min_value=128, max_value=2560, value=512, step=128)

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
        markdown_path = f"./tmp/{st.session_state.pdf_name}.md"

        # Choose processing mode
        processing_mode = st.radio("Choose processing mode:", ("Simple Processing", "Advanced Processing"))

        # Button to start processing
        start_button = st.button("Start Processing")

        if start_button:
            # Set processing_complete to True when processing starts
            st.session_state.processing_complete = True
            st.session_state.processing_mode = processing_mode  # Store the selected mode

            text = ""

            # put the target pdf in the local folder (in tmp folder)
            # Remove the tmp folder if it exists
            if os.path.exists("./tmp"):
                shutil.rmtree("./tmp")

            # Recreate the tmp folder
            os.makedirs("./tmp", exist_ok=True)

            pdf_path = f"./tmp/{pdf.name}"

            # Save the uploaded file to disk
            with open(pdf_path, "wb") as f:
                f.write(pdf.getbuffer())

            with st.spinner("Processing PDF..."): 
                # Process PDF based on the selected mode
                if st.session_state.processing_mode == "Simple Processing":
                    # Extract text from the uploaded PDF as markdown
                    text = pymupdf4llm.to_markdown(pdf_path)

                elif st.session_state.processing_mode == "Advanced Processing":

                    artifact_dict = create_model_dict()
                    converter = PdfConverter(artifact_dict=artifact_dict)
                    convert2md = Convert2Markdown()
                    convert2md.pdf_to_markdown(marker_converter=converter, input_pdf=pdf_path, output_directory="./tmp/")
                    text = convert2md._load_file(markdown_path)

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
            chunks = text_splitter.split_text(text)
            logger.info(f'Chunks Numbers for {pdf.name} is : {len(chunks)}')

            # Convert chunks into Document objects
            documents = [Document(page_content=chunk) for chunk in chunks]

            with st.spinner("Generating embeddings..."):
                # Create embeddings and a vector database
                vector_db = Chroma.from_documents(
                    documents=documents,
                    embedding=OllamaEmbeddings(model="nomic-embed-text:latest"),
                    collection_name="pdf_content",
                    persist_directory="./PDF_ChromaDB",
                )

            st.success("Processing Complete!")

    # User selects the model Provider
    llm_provider = st.selectbox("Select LLM Provider:", ['Sambanova', 'Ollama'], index=0)

    if llm_provider == 'Ollama' :
        selected_model = st.selectbox("Select an Ollama model:", available_models, index=0)
    else : 
        llm_name = st.selectbox("Enter LLM Name:", ['DeepSeek-R1-Distill-Llama-70B', 'DeepSeek-V3-0324', 'DeepSeek-R1', 'Qwen3-32B', 'QwQ-32B'], index=0)
        api_key = st.text_input("Enter Sambanova API Key", type="password", value=os.getenv("API_KEY"))

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
    - Simple Processing : extract the text as markdown from the pdf if the pdf searchable. (Faster)
    - Advanced Processing : Efficient for Complex PDFs (like Research Papers), extract the text by converting the pdf to markdown using OCR and then search the markdown file. (Slower)
""")

# Initialize the RAG system
rag_system = RAGSystem(collection_name="pdf_content", db_path="PDF_ChromaDB", n_results=n_results)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "max_messages" not in st.session_state:
    st.session_state.max_messages = 50  # 25 user + 25 assistant messages

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("----")

# Stop if max messages are reached
if len(st.session_state.messages) >= st.session_state.max_messages:
    st.info("Notice: The maximum message limit has been reached. clear the chat plz!")
else:
    if query := st.chat_input("Ask me ..."):
        if pdf is None or st.session_state.processing_complete != True :
            st.error("Please upload at least one PDF file.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):

                        if llm_provider == 'Ollama' :

                            llm_response, time, docs_nbrs , input_token_count, output_token_count = rag_system.generate_response(query.strip(), selected_model)

                            response = f"""
                            {remove_tags(llm_response)}

                            \n----
                            LLM Name: {selected_model} | Response Time: {time} | Input Tokens Count : {input_token_count} | Output Tokens Count : {output_token_count} | Number of Retrieved Documents: {docs_nbrs}
                            """
                            st.markdown(response)

                        else :
                            llm_response, time, docs_nbrs , input_token_count, output_token_count  = rag_system.generate_response2(query.strip(), llm_name)

                            if llm_response == "":
                                llm_response = "No, response, Maybe the API not available or the model is not installed"

                            if 'Error' in llm_response:
                                st.error(llm_response)
                                response = llm_response

                            elif 'No Document retrieved' in llm_response:
                                st.error("No Document retrieved, make sure to add documents to the Vector DB")
                                response = llm_response

                            else:
                                response = f"""
                                {remove_tags(llm_response)}

                                \n----
                                LLM Name: {llm_name} | Response Time: {time} | Input Tokens Count : {input_token_count} | Output Tokens Count : {output_token_count} | Number of Retrieved Documents: {docs_nbrs}
                                """
                                st.markdown(response)

                        # Store assistant response
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
                    st.rerun()

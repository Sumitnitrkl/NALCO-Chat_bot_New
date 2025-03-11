import os, time, re
import shutil
import logging
import chromadb
import subprocess
from PIL import Image
import streamlit as st
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain.schema import Document 

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set up logging
logger = logging.getLogger(__name__)
logger.info(f"Streamlit app is running")

class Convert2Markdown:
    """
    Convert2Markdown: Converts PDF and HTML files to Markdown.
    """
    def _markdown_remove_images(self, markdown: str) -> str:
        """Remove image references from Markdown text."""
        return re.sub(r'!\[.*?\]\(.*?\)', '', markdown)

    def _remove_images_from_directory(self, directory: str, extension: str = ".jpeg") -> None:
        """Delete all images with a specific extension in a directory."""
        for file in os.listdir(directory):
            if file.endswith(extension):
                os.remove(os.path.join(directory, file))

    def _load_file(self, file_path: str) -> str:
        """Load and return the content of a file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _save_file(self, file_path: str, content: str) -> None:
        """Save content to a file."""
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
            
    def _format_time(self, response_time):
        hours = response_time // 3600
        minutes = (response_time % 3600) // 60
        seconds = response_time % 60

        if hours:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"Time: {int(seconds)}s"

    def pdf_to_markdown(self, marker_converter, input_pdf: str, output_directory: str, remove_images: bool = True) -> None:
        """Convert a PDF file to Markdown using the Marker tool."""
        
        if not marker_converter:
            raise ValueError("marker_converter instance is required.")
        if not input_pdf or not output_directory:
            raise ValueError("Both input PDF path and output directory are required.")
        if not os.path.exists(input_pdf):
            raise FileNotFoundError(f"Input PDF '{input_pdf}' does not exist.")
        
        os.makedirs(output_directory, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(input_pdf))[0]
        markdown_path = os.path.join(output_directory, f"{base_filename}.md")
        
        start_time = time.time()
        
        try:
            rendered = marker_converter(input_pdf)
            save_output(rendered, output_directory, f"{base_filename}")
            
            # if remove_images and os.path.exists(markdown_path):
            if remove_images :
                markdown_content = self._load_file(markdown_path)
                markdown_content = self._markdown_remove_images(markdown_content)
                self._save_file(markdown_path, markdown_content)
                self._remove_images_from_directory(output_directory, extension=".jpeg")
                
            print(f"Markdown saved: '{markdown_path}', Conversion completed in {self._format_time(time.time() - start_time)} seconds")
        except Exception as e:
            print(f"Error during PDF conversion: {e}")
            raise

class RAGSystem :
    """
    RAG System
    """
    def __init__(
            self, collection_name: str, 
            db_path: str ="PDF_chroma_db", 
            ollama_model: str='deepseek-r1:7b', 
            n_results: int =5
        ) :

        self.collection_name = collection_name
        self.db_path = db_path
        self.ollama_llm = OllamaLLM(model=ollama_model)
        self.n_results = n_results

        if not self.collection_name:
            raise ValueError("'collection_name' parameter is required.")

        self.logger = self._setup_logging()
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        # self.logger.info("*** RAGSystem initialized ***")
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        log_file = Path("./RAG.log")
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger
    
    def _format_time(self, response_time):
        minutes = response_time // 60
        seconds = response_time % 60
        return f"{int(minutes)}m {int(seconds)}s" if minutes else f"Time: {int(seconds)}s"
    
    def _generate_embeddings(self, text: str):
        return self.embedding_model.embed_query(text)
    
    def _retrieve(self, user_text: str):
        """Retrieves relevant documents based on user input."""
        embedding = self._generate_embeddings(user_text)
        results = self.collection.query(query_embeddings=[embedding], n_results=self.n_results)
        
        if not results['documents']:
            return []
        
        return results['documents'][0]
    
    def generate_response(self, query: str):
        """Generates a response using retrieved documents and an LLM."""
        self.logger.info("--> Generate Response from LLM")
        retrieved_docs = self._retrieve(query)
        if not retrieved_docs:
            return "No relevant information found."
        
        context = "\n-----\n".join(retrieved_docs)
        
        prompt = f"""
        You are an AI assistant specialized in answering questions based **only** on the provided context.  
        The context is structured with sections separated by `-----`.  

        ### **Context:**  
        '''  
        {context}  
        '''  

        ### **Question:**  
        "{query}"  

        ### **Instructions:**  
        - Answer concisely and accurately using only the given context.  
        - Put what you find from the context **without summarizing**, and **Expand the answer**.
        - Answer directly and concisely. (without writing 'answer :')
        - If the context is unclear, just give me what did you find.
        - If the context is missing state: "The provided context does not contain enough information." (but try to answer) 

        ### **Answer:**
        """
        self.logger.info(f"-> User Query : {query}")
        self.logger.info(f"-> Context : {prompt}")
        
        token_count = self.ollama_llm.get_num_tokens(prompt)
        start_time = time.time()

        streamed_response = ""
        for chunk in self.ollama_llm.stream(prompt): 
            streamed_response += chunk
            yield streamed_response 

        response_time = time.time() - start_time
        self.logger.info(f"-> LLM Response : {streamed_response}")
        self.logger.info(f"-> input token count : {token_count}  |  response time : {self._format_time(response_time)}")
        metadata = {
            'n_results': self.n_results,
            'token_count': token_count,
            'response_time': self._format_time(response_time)
        }
        yield metadata

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)

# --- Streamlit App---

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
st.set_page_config(page_title="Chat with your PDF", page_icon=image2)

st.header("💬 Chat with your PDF")

with st.sidebar:
    st.header("💬 Chat with your PDF Locally Using Ollama")
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
                    persist_directory="./PDF_chroma_db",
                )

            st.success("Processing Complete!")
            # st.snow()
            # st.balloons()


    # User selects the model
    selected_model = st.selectbox("Select an Ollama model:", available_models, index=0)

    # Slider to choose the number of retrieved results
    n_results = st.slider("Number of retrieved documents", min_value=1, max_value=10, value=5)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.info("""
##### There're 2 method to process PDF after uploading: 
    - Simple Processing : extract the text directly from the pdf if the pdf searchable. (Faster)
    - Advanced Processing : extract the text by converting the pdf to markdown using OCR and then search the markdown file. (Slower)
        
-> The chatbot depends on your performence of your labtop, so please be patient!
""")

# Initialize the RAG system
rag_system = RAGSystem(collection_name="pdf_content", db_path="PDF_chroma_db", ollama_model=selected_model, n_results=n_results)


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
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                metadata = None
                with st.spinner("Thinking..."):
                    # Stream LLM response
                    response_placeholder = st.empty()
                    streamed_response = ""

                    for chunk in rag_system.generate_response(prompt):  # Stream response
                        if isinstance(chunk, dict):  # Ensure metadata is correctly assigned
                            metadata = chunk
                        else:
                            streamed_response = chunk  # Accumulate response text
                            response_placeholder.markdown(streamed_response)  # Update UI

                logger.info(f"Metadata: {metadata}")

                # if metadata:
                st.write(f"""
                    \n\n----
                    Token Count: {metadata.get('token_count', 'N/A')} | Response Time: {metadata.get('response_time', 'N/A')} | n_results of context: {metadata.get('n_results', 'N/A')}  
                    """)

                response = f"""
                    {remove_tags(streamed_response)}

                    \n----
                    Token Count: {metadata.get('token_count', 'N/A')}, 
                    Response Time: {metadata.get('response_time', 'N/A')}, 
                    n_results of context: {metadata.get('n_results', 'N/A')} 
                    """

                    # Store assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
                st.rerun()

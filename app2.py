import streamlit as st  # type: ignore
from PIL import Image
from PyPDF2 import PdfReader  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_ollama import OllamaEmbeddings  # type: ignore
from langchain_community.vectorstores import Chroma  # type: ignore
from langchain_ollama.chat_models import ChatOllama  # type: ignore
from langchain.schema import Document  # type: ignore # Import the Document class
from langchain.chains import RetrievalQA # type: ignore


# Load and display the image
image = Image.open('images/bala-ollama.webp')
st.image(image, caption='Run LLM Locally using Ollama')

# App title
st.title("Ask your PDF Using RAG - Ollama")

# File uploader for PDF
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    # Extract text from the uploaded PDF
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

        # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Convert chunks into Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create embeddings and a vector database
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="local-rag",
        persist_directory="./chroma_langchain_db",
    )

    # Input for user question
    st.header("Ask your PDF 💬")
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        # Initialize the LLM
        local_model = "huihui_ai/qwen2.5-1m-abliterated:latest"
        llm = ChatOllama(model=local_model)

        # Create a retriever
        retriever = vector_db.as_retriever()

        # Define the chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False  #use True If you want to Know the Docs that your Query Similaire to.
        )

        # Get the response from the model
        try:
            res = chain.invoke({"query": user_question})
            st.write("IA Response : \n",res["result"])
                
        except Exception as e:
            st.error(f"Error during retrieval or processing: {e}")

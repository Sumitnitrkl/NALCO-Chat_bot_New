# Chat with Your PDF Locally 🤖

## Overview

"ChatPDF" is an interactive web application that allows you to upload a PDF file and engage with it by querying specific information. The app processes the PDF using various techniques, such as extracting text directly or converting it to Markdown, and stores the information in a local database to generate responses to user queries. The app is powered by Ollama and integrates with the Langchain framework to generate accurate answers based on the content within the PDF. It's a powerful local RAG (Retrieval Augmented Generation) application that lets you chat with your PDF documents.

**ChatPDF** is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them through a chatbot interface. The system uses advanced embedding models and a local vector store for efficient and accurate question-answering.

<img src='./imgs/ChatPDF3.png'></img>

<video src="./imgs/video.mp4" autoplay controls loop></video>


## Features

- **PDF Processing**: Extract text from PDFs directly or use advanced processing to convert PDFs to Markdown.
- **RAG Workflow**: Combines retrieval and generation for high-quality responses.
- **Customizable Retrieval**: Adjust the number of retrieved results (`n_results`) for context.
- **Memory Management**: Easily clear vector store and retrievers to reset the system.
- **Question Answering**: Use the processed PDF content to answer queries through a chatbot interface.
- **Model Selection**: Choose from a list of available Ollama models for different tasks.
- **Text Retrieval**: Retrieve relevant documents from the database based on the user's query.
- **Chat Interface**: Easy-to-use chat interface for interacting with the PDF content.

## 🚀 Getting Started

1. **Clone the Repository**:
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/drisskhattabi6/Chat-with-PDF-Locally.git
   cd Chat-with-PDF-Locally
   ```


2. **Create a Virtual Environment** : (Optional)

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**:
   Install the necessary dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**:
   The app requires Ollama for language models. Follow the [Ollama installation instructions](https://ollama.com/) to install it.

   - Pull required models:
     ```bash
     ollama pull deepseek-r1:7b  # or your preferred model
     ollama pull mxbai-embed-large:latest
     ```

### 🎮 Running the Application

Run the app using the following command:

   ```bash
   streamlit run app.py
   ```

Then open your browser to `http://localhost:8501` (it will open automatically)

<img src='./imgs/img0.png' style='max-width=400px'></img>
<br>
<img src='./imgs/QA.jpg'></img>


## Project Structure

```
.
├── app.py                  # Streamlit app for the user interface
├── requirements.txt        # List of required Python dependencies
├── imgs/                   # some screanshots, logo and video
├── PDF_chroma_db/          # Local persistent vector store (auto-generated)
└── README.md               # Project documentation
```

---

## How it Works

1. **Upload PDF**: 
   After you upload a PDF file, the app will process the content using two modes:
   - **Simple Processing**: Extracts the text directly from the PDF (faster).
   - **Advanced Processing**: Converts the PDF into Markdown format using OCR and extracts the text (slower).

<img src='./imgs/img1.png' style='max-width=400px'></img>

2. **Model Selection** and **Customizable Retrieval**: 
   Once the PDF is processed, the app will use an Ollama language model for text analysis and question answering. You can select a model from a list of available Ollama models. and you can adjust the number of retrieved results (`n_results`) for context.

<img src='./imgs/img2.png' style='max-width=400px'></img>

3. **Query the Content**:
   After the processing is complete, you can ask questions based on the content of the PDF. The app will use the Chroma vector database to search for relevant information and generate an accurate response using the selected Ollama model.

4. **Embedding & Vector Database**:
   The app generates embeddings from the PDF content and stores them in a Chroma vector database. This allows fast retrieval of relevant text based on user queries.

<img src='./imgs/Pre-processing.png' style='max-width=600px'></img>
<br>
<img src='./imgs/img6.png' style='max-width=600px'></img>

## Features Breakdown

- **PDF Conversion**: The app uses the Marker library to convert PDFs to Markdown format. The conversion can be customized to remove images.
- **Text Chunking**: Large documents are split into manageable chunks for easier processing.
- **Embedded Models**: The app supports Ollama models for document embeddings and generating answers based on the content.
- **Chroma Vector Store**: All the processed documents are stored in a Chroma vector store for efficient retrieval.
- **RAG**: Advanced RAG implementation using LangChain

## How to Use

1. **Upload PDF**: Upload a PDF file using the file uploader in the sidebar.
2. **Choose Processing Mode**: Select between "Simple Processing" and "Advanced Processing."
3. **Start Processing**: Click the "Start Processing" button to begin the conversion and embedding process.
4. **Ask Questions**: After processing is complete, ask questions related to the content of the PDF.
5. **Select Model**: Choose the Ollama model to generate the answers.
6. **Clear Chat**: Clear the chat history using the "Clear Chat" button.

### Some Screenshots :

<img src='./imgs/img3.png' style='max-width=600px'></img>
<br>
<img src='./imgs/img4.png' style='max-width=600px'></img>
<br>
<img src='./imgs/img5.png' style='max-width=600px'></img>

## Requirements

- Python 3.8+
- Pip
- Ollama models installed via `ollama pull`
- Marker library for PDF to Markdown conversion
- Chroma for storing vector embeddings

## Troubleshooting

- If no Ollama models are found, ensure that Ollama is properly installed and models are pulled using `ollama pull <model_name>`.
- Ensure that the PDF file uploaded is valid and can be processed by the app.

---

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://github.com/streamlit/streamlit)
- [Ollama](https://ollama.ai/)

---

Follow me on [LinkedIn](https://www.linkedin.com/in/idriss-khattabi-b3a266235/)
import logging
import chromadb
import tiktoken
import time, os
import numpy as np
from nltk import download
from nltk.data import find
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm_inference import LLMInference

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')

try:
    find('tokenizers/punkt')
except LookupError:
    download('punkt')

try:
    find('tokenizers/punkt_tab')
except LookupError:
    download('punkt_tab')

# Load SentenceTransformer model for semantic similarity (Contriever-style)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class RAGSystem :
    def __init__(
            self, collection_name: str, 
            db_path: str ="PDF_ChromaDB", 
            n_results: int = 5
        ) :

        self.collection_name = collection_name
        self.db_path = db_path
        self.n_results = n_results

        if not self.collection_name:
            raise ValueError("'collection_name' parameter is required.")

        self.llm_inference = LLMInference()

        self.logger = self._setup_logging()
        # self.embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger
    
    def _format_time(self, response_time):
        minutes = response_time // 60
        seconds = response_time % 60
        return f"{int(minutes)}m {int(seconds)}s" if minutes else f"Time: {int(seconds)}s"
    
    def _generate_embeddings(self, text: str):
        return self.llm_inference._generate_embeddings(input_text=text, model_name="nomic-embed-text:latest")
        # return self.llm_inference._generate_embeddings(input_text=text, model_name="mxbai-embed-large:latest")
    
    def _get_tokens_count(self, text: str):
        """Returns the number of tokens in the given text."""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    
    def _retrieve(self, user_text: str, n_results:int=10):
        """Retrieves relevant documents based on user input."""
        embedding = self._generate_embeddings(user_text)
        results = self.collection.query(query_embeddings=[embedding], n_results=n_results)
        
        if not results['documents']:
            return []
        
        return results['documents'][0]
    
    def _rerank_docs(self, chunks: list, query: str, top_k: int = 5):
        """Retrieves and ranks text chunks using BM25, semantic similarity"""
        # chunks = [chunk['content'] for chunk in chunks] 
        # ----- BM25 Lexical Ranking -----
        tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(word_tokenize(query.lower()))
        
        # ----- Semantic Ranking -----
        chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
        query_embedding = embedder.encode([query], convert_to_tensor=True)
        semantic_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # ----- Combine Scores -----
        bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-5)
        sem_norm = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores) + 1e-5)
        combined_scores = 0.5 * bm25_norm + 0.5 * sem_norm

        # ----- Top-K Selection -----
        ranked_indices = np.argsort(combined_scores)[::-1]  # from best to worst

        final_chunks = [chunks[i] for i in ranked_indices]
        final_chunks = final_chunks[:top_k]

        return final_chunks
    
    def _get_prompt(self, query, context) :
        prompt = f"""
    You are an AI assistant specialized in answering questions based **only** on the provided context. 
    The context is a part of a document (PDF).
    The context is structured with sections separated by `########`. 

    ### **Context:**  
        '''  
        {context}  
        '''  

    ### **Question:**  
        "{query}"  

    ### **Instructions:**  
        - Answer concisely and accurately using only the given context.  
        - Put what you find from the context **without summarizing**.
        - Answer directly and concisely.

    ### **Answer:**

    """
        return prompt
    
            # - If the context is unclear, just give me what did you find.
        # - If the context is missing state: "The provided context does not contain enough information." (but try to answer) 
    
    def generate_response(self, query: str, ollama_model):
        """Generates a response using retrieved documents and an LLM."""

        if not ollama_model :
            return "Error: Choose an ollama LLM"

        self.logger.info(f"--> Generate Response Using Ollama LLM : {ollama_model}")

        retrieved_docs = self._retrieve(query, n_results=20)
        
        if not retrieved_docs:
            return "No relevant information found, may be the data base is empty."
        
        reranked_retrieved_docs = self._rerank_docs(chunks=retrieved_docs, query=query, top_k=self.n_results)

        context = "\n\n########\n\n".join(reranked_retrieved_docs)
        
        prompt = self._get_prompt(query, context)

        self.logger.info(f"-> User Query : {query}")
        self.logger.info(f"-> Context : {prompt}")

        # input_query_token_count = self.ollama_llm.get_num_tokens(query)
        input_prompt_token_count = self._get_tokens_count(prompt)
        start_time = time.time()

        response = self.llm_inference.generate_text(prompt=prompt, model_name=ollama_model, llm_provider='Ollama')

        output_token_count = self._get_tokens_count(response)
        response_time = time.time() - start_time
        self.logger.info(f"-> LLM Response : {response}")
        self.logger.info(f"-> Output token count : {output_token_count} | Input token count : {input_prompt_token_count} | response time : {self._format_time(response_time)}")

        return response, self._format_time(response_time), self.n_results , input_prompt_token_count, output_token_count
    

    def generate_response2(self, query, llm_name='QwQ-32B', api_key=None) :

        if not api_key and not API_KEY :
            return "Set OpenRouter API Key"
        
        api_key = api_key if api_key else API_KEY

        self.logger.info(f"--> Generate Response Using OpenRouter LLM : {llm_name}")

        retrieved_docs = self._retrieve(query, n_results=20)
        self.logger.info(f"-> type Retrieved documents : {type(retrieved_docs)}")
        self.logger.info(f"-> api_key : {api_key}")
        if not retrieved_docs:
            return "No relevant information found."
        
        reranked_retrieved_docs = self._rerank_docs(chunks=retrieved_docs, query=query, top_k=self.n_results)

        context = "\n\n########\n\n".join(reranked_retrieved_docs)

        prompt = self._get_prompt(query, context)

        start_time = time.time()

        response = self.llm_inference.generate_text(prompt=prompt, model_name=llm_name, llm_provider='Sambanova')

        input_prompt_token_count = response[1]
        output_token_count = self._get_tokens_count(response[0])
        response_time = time.time() - start_time
        self.logger.info(f"-> LLM Response : {response[0]}")
        self.logger.info(f"-> Output token count : {output_token_count} | Input token count : {input_prompt_token_count}  |  response time : {self._format_time(response_time)}")
        
        return response[0], self._format_time(response_time), self.n_results , input_prompt_token_count, output_token_count

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)

if __name__ == "__main__":
    rag_system = RAGSystem(collection_name="pdf_content", db_path="PDF_ChromaDB", n_results=5)
    print(rag_system.generate_response2("What is the name of the book ?", "QwQ-32B"))
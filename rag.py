import logging
import chromadb
import tiktoken
import time, os
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from llm_inference import LLMInference

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')

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
    
    def _get_tokens_count(self, text: str):
        """Returns the number of tokens in the given text."""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    
    def _retrieve(self, user_text: str, n_results: int = 10):

        embedding = self._generate_embeddings(user_text)

        # Include document texts and their embeddings
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "embeddings"]
        )

        if not results['documents']:
            return []

        chunks = results['documents'][0]
        embeddings = results['embeddings'][0]

        return chunks, embeddings
    
    def _rerank_docs(self, chunks: list[str], embeddings: list[list[float]], query: str, top_k: int = 5):
        """Ranks text chunks using BM25 and precomputed semantic similarity."""
        # ----- BM25 -----
        tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(word_tokenize(query.lower()))

        # ----- Semantic similarity using Ollama embeddings -----
        query_embedding = np.array(self._generate_embeddings(query))
        chunk_embeddings = np.array(embeddings)

        # Cosine similarity
        dot_product = np.dot(chunk_embeddings, query_embedding)
        query_norm = np.linalg.norm(query_embedding)
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
        semantic_scores = dot_product / (chunk_norms * query_norm + 1e-10)

        # ----- Score normalization -----
        bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-5)
        sem_norm = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores) + 1e-5)
        combined_scores = 0.5 * bm25_norm + 0.5 * sem_norm

        # ----- Top-k selection -----
        ranked_indices = np.argsort(combined_scores)[::-1]
        return [chunks[i] for i in ranked_indices[:top_k]]
    
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

        chunks, embeddings = self._retrieve(query, n_results=20)
        
        if not chunks:
            return "No relevant information found, may be the data base is empty."
        
        reranked_retrieved_docs = self._rerank_docs(chunks=chunks, embeddings=embeddings, query=query, top_k=self.n_results)

        context = "\n\n########\n\n".join(reranked_retrieved_docs)
        
        prompt = self._get_prompt(query, context)

        self.logger.info(f"-> User Query : {query}")
        self.logger.info(f"-> Context : {context}")

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

        chunks, embeddings = self._retrieve(query, n_results=20)

        if not chunks:
            return "No relevant information found, may be the data base is empty."
        
        reranked_retrieved_docs = self._rerank_docs(chunks=chunks, embeddings=embeddings, query=query, top_k=self.n_results)

        context = "\n\n########\n\n".join(reranked_retrieved_docs)

        self.logger.info(f"-> Context : {context}")

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
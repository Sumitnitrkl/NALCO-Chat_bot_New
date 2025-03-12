import time
import logging
import chromadb
from pathlib import Path
from langchain_ollama import OllamaEmbeddings, OllamaLLM

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

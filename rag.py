import logging
import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import time, os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

class RAGSystem :
    def __init__(
            self, collection_name: str, 
            db_path: str ="PDF_chroma_db", 
            n_results: int =5
        ) :

        self.collection_name = collection_name
        self.db_path = db_path
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
    
    def _rerank_docs(self, retrieved_docs:list) :
        return retrieved_docs
    
    def _get_prompt(self, query, context) :
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
        return prompt
    
    def generate_response(self, query: str, ollama_model):
        """Generates a response using retrieved documents and an LLM."""

        if not ollama_model :
            return "Choose and ollama LLM"

        self.logger.info("--> Generate Response Using Ollama : ", ollama_model)

        retrieved_docs = self._retrieve(query)
        
        if not retrieved_docs:
            return "No relevant information found."
        
        reranked_retrieved_docs = self._rerank_docs(retrieved_docs)

        context = "\n-----\n".join(reranked_retrieved_docs)
        
        prompt = self._get_prompt(query, context)

        self.logger.info(f"-> User Query : {query}")
        self.logger.info(f"-> Context : {prompt}")

        ollama_llm = OllamaLLM(model=ollama_model)
        
        token_count = ollama_llm.get_num_tokens(prompt)
        start_time = time.time()

        streamed_response = ""
        for chunk in ollama_llm.stream(prompt): 
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

    def generate_response2(self, query, llm_name='qwen/qwq-32b:free') :

        self.logger.info("--> Generate Response Using OpenRouter : ", llm_name)

        retrieved_docs = self._retrieve(query)
        
        if not retrieved_docs:
            return "No relevant information found."
        
        reranked_retrieved_docs = self._rerank_docs(retrieved_docs)

        context = "\n-----\n".join(reranked_retrieved_docs)

        prompt = self._get_prompt(query, context)

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": llm_name,
                "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
                ],
            })
        )

        response_data = json.loads(response.text)
        # print(response_data)
        return response_data["choices"][0]["message"]["content"]

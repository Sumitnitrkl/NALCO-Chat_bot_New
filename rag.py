# rag.py

import logging
import chromadb
import re
import time
import os
import numpy as np
from llm_inference import LLMInference
import db_utils

from dotenv import load_dotenv
load_dotenv()

class RAGSystem:
    def __init__(self, collection_name: str, db_path: str = "PDF_ChromaDB", n_results: int = 10):
        self.collection_name = collection_name
        self.db_path = db_path
        self.n_results_to_retrieve = 20
        self.n_results_to_use = n_results
        self.llm_inference = LLMInference()
        self.logger = self._setup_logging()
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.logger.info(f"Initialized Hybrid RAG System for collection '{self.collection_name}'.")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _format_time(self, response_time):
        return f"{response_time:.2f}s"
    
    def _generate_embeddings(self, text: str):
        try:
            return self.llm_inference._generate_embeddings(input_text=text, model_name="nomic-embed-text:latest") or []
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return []
    
    def chunk_text(self, text: str):
        pattern = r'\n(\d+(?:\.\d+)+)\s'
        parts = re.split(pattern, text)
        chunks = [parts[0].strip()] if parts[0] and parts[0].strip() else []
        for i in range(1, len(parts), 2):
            chunks.append(f"{parts[i]} {parts[i+1].strip()}")
        return chunks

    def _parse_query_for_direct_lookup(self, query: str) -> dict:
        pattern = r'\b(sec|section|secton|clause|point|no)\b\.?\s*(\d+(?:\.\d+)+)'
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            section_number = match.group(2)
            self.logger.info(f"Direct lookup detected for section '{section_number}'.")
            return {"section_number": section_number}
        return None

    def _generate_hypothetical_answer(self, query: str) -> str:
        prompt = f"""Please write a short, one-paragraph, hypothetical answer to the user query. This is for an internal search system; the answer should be dense with concepts related to the query.

User Query: "{query}"

Hypothetical Answer:"""
        hypothetical_answer, _, _ = self.llm_inference.generate_text(prompt=prompt, model_name="mistral", llm_provider="Ollama")
        return hypothetical_answer

    def _rerank_with_mmr(self, query_embedding: np.ndarray, chunks: list, chunk_embeddings: np.ndarray, top_k: int, diversity: float = 0.5):
        if not chunks: return []
        query_chunk_similarity = np.dot(chunk_embeddings, query_embedding)
        chunk_chunk_similarity = np.dot(chunk_embeddings, chunk_embeddings.T)
        
        selected_indices, candidate_indices = [], list(range(len(chunks)))
        if not candidate_indices: return []

        best_initial_index = np.argmax(query_chunk_similarity)
        selected_indices.append(best_initial_index)
        candidate_indices.remove(best_initial_index)
        
        while len(selected_indices) < top_k and candidate_indices:
            mmr_scores = []
            for index in candidate_indices:
                relevance_score = (1 - diversity) * query_chunk_similarity[index]
                diversity_penalty = diversity * np.max(chunk_chunk_similarity[index, selected_indices])
                mmr_scores.append(relevance_score - diversity_penalty)
            
            best_mmr_index = np.argmax(mmr_scores)
            selected_index = candidate_indices[best_mmr_index]
            selected_indices.append(selected_index)
            candidate_indices.remove(selected_index)
            
        return [chunks[i] for i in selected_indices]

    def _generate_final_response(self, query: str, llm_provider_details: dict):
        if self.collection.count() == 0:
            return "No documents available. Please upload a document first.", "N/A", 0, 0, 0

        final_chunks = []
        
        # --- PATH A: THE "SHORTCUT" ---
        where_filter = self._parse_query_for_direct_lookup(query)
        if where_filter:
            self.logger.info(f"Executing direct metadata lookup with filter: {where_filter}")
            # --- START OF FIX ---
            # Use collection.get() for direct metadata filtering, not query()
            results = self.collection.get(where=where_filter, limit=1, include=["documents"])
            # --- END OF FIX ---
            
            if results and results.get('documents'):
                self.logger.info("Direct lookup successful. Using precise result.")
                final_chunks = results['documents']

        # --- PATH B: THE GENERAL-PURPOSE PIPELINE ---
        if not final_chunks:
            self.logger.info("No direct lookup match. Executing general-purpose HyDE + MMR pipeline.")
            
            hypothetical_answer = self._generate_hypothetical_answer(query)
            search_embedding = self._generate_embeddings(hypothetical_answer)
            if not search_embedding: search_embedding = self._generate_embeddings(query)
            
            n_to_retrieve = min(self.collection.count(), self.n_results_to_retrieve)
            results = self.collection.query(query_embeddings=[search_embedding], n_results=n_to_retrieve, include=["documents", "embeddings"])
            
            if not (results and results['documents'] and results['documents'][0]):
                return "No relevant information found for your query.", "N/A", 0, 0, 0
            
            candidate_chunks = results['documents'][0]
            candidate_embeddings = np.array(results['embeddings'][0])

            query_embedding = self._generate_embeddings(query)
            if not query_embedding:
                final_chunks = candidate_chunks[:self.n_results_to_use]
            else:
                final_chunks = self._rerank_with_mmr(np.array(query_embedding), candidate_chunks, candidate_embeddings, top_k=self.n_results_to_use)
        
        if not final_chunks:
            return "Could not construct a relevant context for the query.", "N/A", 0, 0, 0

        # --- Final answer generation ---
        context = "\n\n---\n\n".join(final_chunks)
        prompt = f"""You are an expert AI assistant. Based *only* on the provided context, provide a clear and direct answer to the user's query.

### Context:
{context}

### User's Query:
"{query}"

### Answer:
"""
        
        start_time = time.time()
        response, input_tokens, output_tokens = self.llm_inference.generate_text(**llm_provider_details, prompt=prompt)
        response_time = time.time() - start_time
        
        return response, self._format_time(response_time), len(final_chunks), input_tokens, output_tokens
    
    def generate_response(self, query: str, ollama_model):
        if not ollama_model: return "Error: Choose an Ollama LLM", "N/A", 0, 0, 0
        details = {"llm_provider": "Ollama", "model_name": ollama_model}
        return self._generate_final_response(query, details)
    
    def generate_response2(self, query: str, llm_name='QwQ-32B', api_key=None):
        API_KEY = os.getenv('API_KEY')
        if not (api_key or API_KEY): return "Set OpenRouter API Key", "N/A", 0, 0, 0
        details = {"llm_provider": "Sambanova", "model_name": llm_name}
        return self._generate_final_response(query, details)

    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception: pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

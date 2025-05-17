import os
import json
import ollama
import requests

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
API_URL = "https://api.sambanova.ai/v1/chat/completions"

class LLMInference:
    
    def _format_time(self, response_time):
        minutes = response_time // 60
        seconds = response_time % 60
        return f"{int(minutes)}m {int(seconds)}s" if minutes else f"Time: {int(seconds)}s"
    
    def _generate_embeddings(self, input_text: str, model_name:str):
        # print('model_name : ', model_name)
        return ollama.embeddings(model=model_name, prompt=input_text).get("embedding", [])
    
    def generate_text_ollama(self, prompt: str, model_name: str):

        if not prompt or not model_name:
            return {"error": "Both 'prompt' and 'model_name' are required"}

        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.get("message", {}).get("content", "")
        except Exception as e:
            return {"error": "Failed to generate response using Ollama", "details": str(e)}
   
    def generate_text_sambanova(self, prompt: str, model_name: str):
        if not prompt or not model_name:
            return {"error": "Both 'prompt' and 'model_name' are required"}

        response = requests.post(
            url= API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
            })
        )

        response_data = json.loads(response.text)
        print('response_data : ', response_data)
        # Check if there's an error key in the response
        if "error" in response_data:
            error_message = response_data["error"].get("message", "Unknown error")
            raise Exception(f"API Error: {error_message}")
        
        return response_data["choices"][0]["message"]["content"], response_data["usage"]["total_tokens"]

    def generate_text(self, prompt: str, model_name: str, llm_provider: str):
        if llm_provider == 'Ollama':
            return self.generate_text_ollama(prompt, model_name)
        elif llm_provider == 'Sambanova':
            return self.generate_text_sambanova(prompt, model_name)
        else:
            raise ValueError("Unsupported LLM provider. Choose either 'Ollama' or 'Sambanova'.")

if __name__ == "__main__":
    llm = LLMInference()
    prompt = "What is the capital of France?"
    
    # response = llm._generate_embeddings(prompt, model_name="mxbai-embed-large:latest")
    response = llm.generate_text(prompt, model_name="QwQ-32B", llm_provider='Sambanova')
    print(response)

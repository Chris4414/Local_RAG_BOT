import requests
from haystack.core.component import component

@component
class OllamaGenerator:
    def __init__(self, model="phi3:mini", base_url = "HTTP:\\localhost:11434"):
        self.model = model
        self.base_url = base_url

    def run(self, prompt: str) -> dict:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        data = response.json()
        return {"replies": [data["response"].strip()]}
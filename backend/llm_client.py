import json
import logging
import requests
from typing import Generator, Optional
from .config import OLLAMA_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = LLM_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def check_health(self) -> dict:
        """Verify Ollama connection and check available models."""
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                return {
                    "connected": True,
                    "models": models,
                    "target_model_available": any(self.model in m for m in models)
                }
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
        return {
            "connected": False,
            "models": [],
            "target_model_available": False
        }

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        """Generate response from local Llama 3.2 model via Ollama."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            else:
                return f"[Error: Ollama returned status {response.status_code}: {response.text}]"
        except requests.exceptions.Timeout:
            return "[Error: Ollama generation timed out after 60 seconds.]"
        except Exception as e:
            return f"[Error communicating with Ollama: {str(e)}]"

import requests
import json
import base64
from typing import Optional, Dict, Any, List
import os

# use the `gemma3:27b` model by default

class OllamaClient:
    """
    Simple client to interact with local Ollama instance (vision + JSON).
    """
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:27b", num_ctx: Optional[int] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx or int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        self.session = requests.Session()

    def test_connection(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=3)
            return r.ok
        except Exception:
            return False

    def generate(self, prompt: str, system: str = "", stream: bool = False) -> Optional[str]:
        """
        Use /api/generate with JSON output and enlarged context window.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "format": "json",
            "options": {
                "num_ctx": self.num_ctx
            },
            "stream": stream,
        }
        try:
            r = self.session.post(f"{self.base_url}/api/generate", json=payload, timeout=None)
            r.raise_for_status()
            data = r.json()
            # When stream=false, Ollama returns a single JSON with 'response'
            return data.get("response")
        except Exception as e:
            print(f"Ollama error: {e}")
            return None

    def generate_with_images(
        self,
        prompt: str,
        image_paths: List[str],
        stream: bool = False,
        system: Optional[str] = None,
        schema: Optional[dict] = None
    ) -> Optional[str]:
        """
        Send a prompt with images; optional system prompt + JSON schema.
        """
        url = f"{self.base_url}/api/generate"
        images = []
        for image_path in image_paths:
            try:
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    images.append(image_data)
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                continue
        if not images:
            print("No images could be loaded")
            return None
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "images": images,
        }
        if system:
            payload["system"] = system
        if schema is not None:
            payload["format"] = schema
        else:
            payload["format"] = "json"
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'response' in data:
                            full_response += data['response']
                return self._extract_json(full_response)
            else:
                data = response.json()
                return self._extract_json(data.get('response', ''))
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing Ollama response: {e}")
            return None
    
    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        if not text:
            return None
        try:
            json.loads(text)
            return text.strip()
        except Exception:
            pass
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                return None
        return None
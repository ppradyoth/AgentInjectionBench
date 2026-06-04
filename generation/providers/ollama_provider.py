"""Ollama provider — runs local models via Ollama (ollama.ai)."""

import json
import urllib.error
import urllib.request

from generation.providers import BaseLLMProvider, register_provider

OLLAMA_BASE_URL = "http://localhost:11434"


@register_provider("ollama")
class OllamaProvider(BaseLLMProvider):

    def __init__(self, model: str = "qwen2.5:7b", **kwargs):
        self.model = model
        self._base_url = kwargs.get("base_url", OLLAMA_BASE_URL)
        self._temperature = kwargs.get("temperature", 0.8)
        self._max_tokens = kwargs.get("max_tokens", 8192)
        self._verify_connection()

    def _verify_connection(self):
        try:
            req = urllib.request.urlopen(f"{self._base_url}/api/tags", timeout=5)
            req.read()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. "
                f"Is Ollama running? Try: ollama serve\nError: {e}"
            )

    @property
    def name(self) -> str:
        return "ollama"

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self._temperature),
                "num_predict": kwargs.get("max_tokens", self._max_tokens),
                "num_ctx": 8192,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
        return result["message"]["content"]

    def generate_batch(
        self, prompts: list[str], system: str | None = None, **kwargs
    ) -> list[str]:
        return [self.generate(p, system=system, **kwargs) for p in prompts]

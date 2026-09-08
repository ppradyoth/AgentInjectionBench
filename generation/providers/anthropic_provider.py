"""Anthropic Claude provider with prompt caching support."""

from generation.providers import BaseLLMProvider, register_provider


@register_provider("anthropic")
class AnthropicProvider(BaseLLMProvider):

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires the anthropic package. "
                "Install with: pip install agent-injection-bench[anthropic]"
            ) from exc
        self.model = model
        self._client = anthropic.Anthropic()
        self._max_tokens = kwargs.get("max_tokens", 4096)
        self._temperature = kwargs.get("temperature", 0.8)

    @property
    def name(self) -> str:
        return "anthropic"

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        msg_kwargs = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            # prompt caching on system prompt for batch efficiency
            msg_kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        response = self._client.messages.create(**msg_kwargs)
        return response.content[0].text

    def generate_batch(
        self, prompts: list[str], system: str | None = None, **kwargs
    ) -> list[str]:
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, system=system, **kwargs))
        return results

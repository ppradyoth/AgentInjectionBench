"""OpenAI provider stub — install with `pip install agent-injection-bench[openai]`."""

from generation.providers import BaseLLMProvider, register_provider


@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):

    def __init__(self, model: str = "gpt-4o", **kwargs):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install agent-injection-bench[openai]"
            )
        self.model = model
        self._client = openai.OpenAI()
        self._max_tokens = kwargs.get("max_tokens", 4096)
        self._temperature = kwargs.get("temperature", 0.8)

    @property
    def name(self) -> str:
        return "openai"

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            temperature=kwargs.get("temperature", self._temperature),
        )
        return response.choices[0].message.content

    def generate_batch(
        self, prompts: list[str], system: str | None = None, **kwargs
    ) -> list[str]:
        return [self.generate(p, system=system, **kwargs) for p in prompts]

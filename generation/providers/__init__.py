"""Pluggable LLM provider system for dataset generation."""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract base for LLM providers used in synthetic data generation."""

    @abstractmethod
    def __init__(self, model: str, **kwargs):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """Generate a single completion. Returns the text response."""
        ...

    @abstractmethod
    def generate_batch(
        self, prompts: list[str], system: str | None = None, **kwargs
    ) -> list[str]:
        """Generate completions for multiple prompts. Returns list of text responses."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and config."""
        ...


PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {}


def register_provider(name: str):
    def decorator(cls: type[BaseLLMProvider]):
        PROVIDER_REGISTRY[name] = cls
        return cls
    return decorator


def get_provider(name: str, model: str, **kwargs) -> BaseLLMProvider:
    if name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return PROVIDER_REGISTRY[name](model=model, **kwargs)


from generation.providers.anthropic_provider import AnthropicProvider  # noqa: E402, F401
from generation.providers.openai_provider import OpenAIProvider  # noqa: E402, F401

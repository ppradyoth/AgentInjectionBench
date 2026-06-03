"""Generation configuration — all tunables in one place."""

from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.8
    max_tokens: int = 4096
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    variations_per_seed: int = 20
    output_dir: str = "data"

    # per-category overrides (provider/model can differ by category)
    category_overrides: dict[str, dict] = field(default_factory=dict)

    def get_for_category(self, category: str) -> "GenerationConfig":
        if category not in self.category_overrides:
            return self
        overrides = self.category_overrides[category]
        return GenerationConfig(
            provider=overrides.get("provider", self.provider),
            model=overrides.get("model", self.model),
            temperature=overrides.get("temperature", self.temperature),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
            batch_size=overrides.get("batch_size", self.batch_size),
            max_retries=overrides.get("max_retries", self.max_retries),
            retry_delay=overrides.get("retry_delay", self.retry_delay),
            variations_per_seed=overrides.get("variations_per_seed", self.variations_per_seed),
            output_dir=self.output_dir,
        )

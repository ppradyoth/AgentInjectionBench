# Integrate your agent

The adapter contract is deliberately small. Run every benchmark sample through
your agent or guardrail and write one row containing the sample ID and a binary
verdict.

```python
import json
from pathlib import Path

from evaluation.score import DATA_FILE, load_dataset

def run_guardrail(sample: dict) -> str:
    # Replace this with your system. Pass the system prompt, tool definitions,
    # and conversation to the same entrypoint used in production.
    verdict = my_guardrail(
        system_prompt=sample["system_prompt"],
        tools=sample["tools_available"],
        conversation=sample["conversation"],
    )
    return "unsafe" if verdict else "safe"

with Path("predictions.jsonl").open("w") as f:
    for sample in load_dataset(DATA_FILE):
        f.write(json.dumps({
            "id": sample["id"],
            "prediction": run_guardrail(sample),
        }) + "\n")
```

The same adapter works for LangGraph, CrewAI, LlamaIndex, MCP clients, and
custom agents. Only `run_guardrail` changes.

For a free local model through Ollama:

```bash
ollama pull qwen2.5:7b
ollama serve
AIB_MODEL=qwen2.5:7b aib-run \
  --adapter adapters.ollama:adapter \
  --limit 10 \
  --bundle results/ollama-smoke
```

For any OpenAI-compatible endpoint:

```bash
AIB_BASE_URL=https://your-endpoint.example/v1 \
AIB_API_KEY="$YOUR_API_KEY" \
AIB_MODEL=your-model \
aib-run --adapter adapters.openai_compatible:adapter --bundle results/byo-model
```

Framework wrappers are available in `adapters.frameworks`:

```python
from adapters.frameworks import (
    make_crewai_adapter,
    make_langgraph_adapter,
    make_llamaindex_adapter,
    make_mcp_adapter,
)
```

They accept already-created user-owned framework objects. The benchmark never
hosts those frameworks or receives their credentials.

Score locally:

```bash
aib-score --predictions predictions.jsonl --name "My agent"
```

Gate a pull request:

```yaml
- uses: ppradyoth/AgentInjectionBench@v1
  with:
    predictions: predictions.jsonl
    max-asr: "0.25"
    max-fpr: "0.10"
```

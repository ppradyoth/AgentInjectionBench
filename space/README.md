---
title: AgentInjectionBench
emoji: 🔬
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: "6.29.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# AgentInjectionBench

**Catch unsafe tool calls before you ship.**

Switching models? Changing a prompt? Adding a tool? Bring your current OpenAI or Anthropic model ID and test what poisoned content can make it do.

Run attacks alongside benign controls, inspect native tool-call traces, and rerun the same cases after a change. The dataset contains 182 public synthetic cases, including 40 benign controls.

Tools are simulated. Results are explicit contract matches, with errors and missing evidence shown separately. Task completion requires explicit evidence; this Space does not certify your production agent or verify persisted-state changes.

Your API key is sent to the Space server to call your provider. Provider billing applies. Use the local CLI for private configurations.

## Run the same implementation locally

```bash
git clone https://huggingface.co/spaces/ppradyoth/AgentInjectionBench
cd AgentInjectionBench
python3 -m venv .venv
source .venv/bin/activate
pip install 'gradio==6.29.1' -r requirements.txt
python app.py
```

Or save a CLI run with traces and a report:

```bash
AIB_PROVIDER=OpenAI AIB_MODEL=your-model AIB_API_KEY="$OPENAI_API_KEY" \
python -m runtime.cli --adapter adapters.openai_compatible:adapter --seed 42 --limit 10 --bundle results/model-check
```

Use `AIB_PROVIDER=Anthropic` and your Anthropic key for Claude. To test your production agent, return an `AgentTrace` from a local adapter with generated outputs, tool calls and state writes. Set `task_completed` and `task_blocked` only from a trusted task evaluator; leave them unset when unmeasured.

- [Integrate your agent](https://github.com/ppradyoth/AgentInjectionBench/blob/main/docs/integrations.md)
- [Dataset](https://huggingface.co/datasets/ppradyoth/AgentInjectionBench)
- [GitHub](https://github.com/ppradyoth/AgentInjectionBench)

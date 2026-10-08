#!/usr/bin/env python3
import json
import random
import sys
from collections import Counter
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.openai_compatible import run_agent
from evaluation.graders import grade_trace
from evaluation.metrics import render_run_report, summarize_outcomes
from evaluation.score import dataset_fingerprint
from runtime.cases import Case, load_cases
from runtime.sandbox import FakeToolSandbox
from jsonschema.exceptions import SchemaError

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" if (APP_DIR / "data").exists() else APP_DIR.parent / "data"
DATASET_PATH = DATA_DIR / "agent_injection_bench.jsonl"
TAXONOMY_PATH = DATA_DIR / "taxonomy.json"


def load_dataset() -> list[dict]:
    return [case.payload for case in load_cases(DATASET_PATH)]


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH) as f:
        return json.load(f)


DATASET = load_dataset()
TAXONOMY = load_taxonomy()

# ─── Tab 1: Dataset Explorer ───


def get_filter_options():
    categories = sorted(set(s["attack_category"] for s in DATASET))
    intents = sorted(set(s["attacker_intent"] for s in DATASET))
    surfaces = sorted(set(s["injection_surface"] for s in DATASET))
    complexities = sorted(set(s["complexity"] for s in DATASET))
    severities = sorted(set(s["severity"] for s in DATASET))
    bypasses = sorted(set(s["defense_bypass"] for s in DATASET))
    return categories, intents, surfaces, complexities, severities, bypasses


def filter_samples(category, intent, surface, complexity, severity, bypass, search_text):
    filtered = DATASET
    if category:
        filtered = [s for s in filtered if s["attack_category"] == category]
    if intent:
        filtered = [s for s in filtered if s["attacker_intent"] == intent]
    if surface:
        filtered = [s for s in filtered if s["injection_surface"] == surface]
    if complexity:
        filtered = [s for s in filtered if s["complexity"] == complexity]
    if severity:
        filtered = [s for s in filtered if s["severity"] == severity]
    if bypass:
        filtered = [s for s in filtered if s["defense_bypass"] == bypass]
    if search_text:
        search_lower = search_text.lower()
        filtered = [s for s in filtered if search_lower in json.dumps(s).lower()]
    return filtered


def make_table(samples: list[dict]) -> pd.DataFrame:
    if not samples:
        return pd.DataFrame()
    rows = []
    for s in samples:
        rows.append(
            {
                "ID": s["id"],
                "Category": s["attack_category"],
                "Intent": s["attacker_intent"],
                "Surface": s["injection_surface"],
                "Complexity": s["complexity"],
                "Severity": s["severity"],
                "Bypass": s["defense_bypass"],
                "Notes": s.get("notes", "")[:80],
            }
        )
    return pd.DataFrame(rows)


def explore(category, intent, surface, complexity, severity, bypass, search_text):
    filtered = filter_samples(category, intent, surface, complexity, severity, bypass, search_text)
    df = make_table(filtered)
    count_text = f"**{len(filtered)}** samples found"
    return df, count_text


def view_sample(sample_id: str) -> str:
    for s in DATASET:
        if s["id"] == sample_id:
            return json.dumps(s, indent=2, ensure_ascii=False)
    return "Sample not found"


def make_category_chart():
    if not DATASET:
        return go.Figure()
    counts = Counter(s["attack_category"] for s in DATASET)
    fig = px.bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        labels={"x": "Attack Category", "y": "Count"},
        title="Samples by Attack Category",
        color=list(counts.keys()),
    )
    fig.update_layout(showlegend=False, height=400)
    return fig


def make_intent_chart():
    if not DATASET:
        return go.Figure()
    counts = Counter(s["attacker_intent"] for s in DATASET)
    fig = px.pie(
        names=list(counts.keys()),
        values=list(counts.values()),
        title="Attacker Intent Distribution",
    )
    fig.update_layout(height=400)
    return fig


def make_surface_chart():
    if not DATASET:
        return go.Figure()
    counts = Counter(s["injection_surface"] for s in DATASET)
    fig = px.bar(
        x=list(counts.values()),
        y=list(counts.keys()),
        orientation="h",
        labels={"x": "Count", "y": "Injection Surface"},
        title="Injection Surface Distribution",
        color=list(counts.keys()),
    )
    fig.update_layout(showlegend=False, height=400)
    return fig


def make_heatmap():
    if not DATASET:
        return go.Figure()
    categories = sorted(set(s["attack_category"] for s in DATASET))
    intents = sorted(set(s["attacker_intent"] for s in DATASET))
    matrix = []
    for cat in categories:
        row = []
        for intent in intents:
            count = sum(
                1 for s in DATASET if s["attack_category"] == cat and s["attacker_intent"] == intent
            )
            row.append(count)
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=intents,
            y=categories,
            colorscale="YlOrRd",
            text=matrix,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Category × Intent Heatmap", height=450)
    return fig


# ─── Tab 2: Live Agent Tester ───


def test_agent(api_key, provider, model, system_prompt, tools_json, num_attacks, categories):
    if not isinstance(api_key, str) or not api_key.strip():
        return "Please provide an API key.", "", None
    if provider not in {"OpenAI", "Anthropic"}:
        return "Choose OpenAI or Anthropic.", "", None
    if not isinstance(model, str) or not model.strip():
        return "Enter the model ID from your provider.", "", None
    if (
        isinstance(num_attacks, bool)
        or not isinstance(num_attacks, (int, float))
        or not 1 <= num_attacks <= 100
        or int(num_attacks) != num_attacks
    ):
        return "Choose a whole number of cases between 1 and 100.", "", None
    if system_prompt is not None and not isinstance(system_prompt, str):
        return "System prompt must be text.", "", None
    if tools_json is not None and not isinstance(tools_json, str):
        return "Tool definitions must be JSON text.", "", None
    categories = categories or []
    known = {sample["attack_category"] for sample in DATASET if sample["ground_truth"] == "unsafe"}
    if not isinstance(categories, list) or any(
        not isinstance(category, str) or category not in known for category in categories
    ):
        return "Choose categories from the list.", "", None
    try:
        overrides = json.loads(tools_json) if tools_json and tools_json.strip() else None
        if overrides is not None:
            FakeToolSandbox.for_case(Case("config", {"id": "config", "tools_available": overrides}))
    except (ValueError, TypeError, SchemaError):
        return (
            "Invalid tools: use a JSON array with unique names and object parameter schemas.",
            "",
            None,
        )

    attacks = [
        sample
        for sample in DATASET
        if sample["ground_truth"] == "unsafe"
        and (not categories or sample["attack_category"] in categories)
    ]
    controls = [sample for sample in DATASET if sample["ground_truth"] == "safe"]
    if not attacks:
        return "No attacks match the selected categories.", "", None
    count = min(int(num_attacks), len(attacks) + len(controls))
    control_count = min(len(controls), max(1, round(count * 0.2))) if count > 1 else 0
    attack_count = min(len(attacks), count - control_count)
    control_count = min(len(controls), count - attack_count)
    rng = random.Random(42)
    selected = rng.sample(attacks, attack_count) + rng.sample(controls, control_count)
    rng.shuffle(selected)
    results = []
    for sample in selected:
        payload = dict(sample)
        if system_prompt and system_prompt.strip():
            payload["system_prompt"] = system_prompt.strip()
        if overrides is not None:
            payload["tools_available"] = overrides
        case = Case(sample["id"], payload)
        trace = run_agent(
            case,
            provider=provider,
            model=model.strip(),
            api_key=api_key.strip(),
            max_tool_calls=8,
            timeout=60,
        )
        results.append({"outcome": grade_trace(case, trace).to_dict(), "trace": trace.to_dict()})
    summary = summarize_outcomes(result["outcome"] for result in results)
    manifest = {
        "adapter": f"{provider}: {model.strip()}",
        "dataset_sha256": dataset_fingerprint(DATASET),
        "seed": 42,
        "tools": "simulated",
        "max_tool_calls": 8,
        "timeout_seconds": 60,
        "case_ids": [sample["id"] for sample in selected],
        "categories": categories,
        "system_prompt_override": system_prompt or None,
        "tool_definitions_override": overrides,
    }
    report = render_run_report(manifest, summary)
    report += "\nTools were simulated. A violation is a match against a case's explicit contract, not proof of a real external action. Errors and incomplete runs cannot pass.\n"
    report += "\nCompletion and benign blocking stay n/a without explicit task evidence. Persisted-state checks are unscored in this sandbox. Broad argument patterns can flag legitimate calls; inspect the trace before interpreting a match.\n"
    chart_data = [
        (category, stats)
        for category, stats in summary["by_category"].items()
        if stats["evaluated"]
    ]
    fig = go.Figure()
    if chart_data:
        fig.add_bar(
            x=[category for category, _ in chart_data],
            y=[100 * stats["compromised"] / stats["evaluated"] for _, stats in chart_data],
        )
    fig.update_layout(
        title="Contract violations in evaluated attacks",
        yaxis_title="Violation rate (%)",
        yaxis_range=[0, 100],
        height=400,
    )
    details = json.dumps(
        {"manifest": manifest, "summary": summary, "results": results}, indent=2, ensure_ascii=False
    )
    return report, details, fig


# ─── Build App ───


def build_app():
    categories, intents, surfaces, complexities, severities, bypasses = (
        get_filter_options() if DATASET else ([], [], [], [], [], [])
    )

    with gr.Blocks(
        title="AgentInjectionBench — test before you upgrade",
        analytics_enabled=False,
    ) as app:
        gr.Markdown("""
# 🔬 AgentInjectionBench

**Catch unsafe tool calls before you ship.**

Switching models? Changing your system prompt? Adding a tool? Test what untrusted web pages, files, and tool responses can make your model do.

Bring your model ID. Run attacks alongside benign controls. Inspect the calls and contract matches, then rerun the same cases after a change.
        """)

        with gr.Tab("🧪 Test a model"):
            gr.Markdown("""
### Will your next model follow the wrong instructions?

Test a model and system prompt with native tool calling against poisoned content. Tool calls run in a deterministic sandbox with no external side effects.

**Your key is sent to this Space server and used to call your provider.** Provider billing applies. For private prompts or data, [run locally](https://huggingface.co/spaces/ppradyoth/AgentInjectionBench/blob/main/README.md).

Start with 5 cases. Each case allows up to 8 tool calls and 4,096 output tokens per request. Retries are disabled; the time budget is checked between requests. This tests model behavior in the benchmark sandbox; use your own CLI adapter to evaluate your production agent.
            """)

            with gr.Row():
                with gr.Column():
                    provider_select = gr.Dropdown(
                        choices=["Anthropic", "OpenAI"],
                        value="Anthropic",
                        label="Provider",
                    )
                    model_input = gr.Textbox(
                        label="Model",
                        value="",
                        placeholder="Paste your provider's current model ID",
                    )
                    api_key_input = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="sk-...",
                    )
                with gr.Column():
                    system_prompt_input = gr.Textbox(
                        label="System prompt (blank uses each case's prompt)",
                        lines=4,
                        placeholder="You are a helpful assistant...",
                    )
                    tools_input = gr.Code(
                        label="Tools (blank uses case tools; [] disables tools)",
                        language="json",
                        value="",
                    )

            with gr.Row():
                num_attacks_slider = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=5,
                    step=5,
                    label="Total cases (attacks + benign controls)",
                )
                category_select = gr.CheckboxGroup(
                    choices=sorted(
                        {
                            sample["attack_category"]
                            for sample in DATASET
                            if sample["ground_truth"] == "unsafe"
                        }
                    ),
                    label="Attack categories (controls included automatically)",
                )

            test_btn = gr.Button("Test this configuration", variant="primary")

            test_summary = gr.Markdown(label="Summary")
            test_chart = gr.Plot(label="Results Chart")
            test_details = gr.Code(
                label="Traces, outcomes and run settings (JSON)", language="json"
            )

            test_btn.click(
                test_agent,
                inputs=[
                    api_key_input,
                    provider_select,
                    model_input,
                    system_prompt_input,
                    tools_input,
                    num_attacks_slider,
                    category_select,
                ],
                outputs=[test_summary, test_details, test_chart],
                api_name="test_agent",
                concurrency_limit=1,
            )

        with gr.Tab("📊 Dataset Explorer"):
            with gr.Row():
                with gr.Column(scale=1):
                    cat_filter = gr.Dropdown(
                        choices=[""] + categories, label="Attack Category", value=""
                    )
                    intent_filter = gr.Dropdown(
                        choices=[""] + intents, label="Attacker Intent", value=""
                    )
                    surface_filter = gr.Dropdown(
                        choices=[""] + surfaces, label="Injection Surface", value=""
                    )
                with gr.Column(scale=1):
                    complexity_filter = gr.Dropdown(
                        choices=[""] + complexities, label="Complexity", value=""
                    )
                    severity_filter = gr.Dropdown(
                        choices=[""] + severities, label="Severity", value=""
                    )
                    bypass_filter = gr.Dropdown(
                        choices=[""] + bypasses, label="Defense Bypass", value=""
                    )

            search_box = gr.Textbox(
                label="Search (keyword)", placeholder="e.g., system prompt, exfiltration, MCP"
            )
            search_btn = gr.Button("Search", variant="primary")
            count_label = gr.Markdown(f"**{len(DATASET)}** samples total")

            results_table = gr.Dataframe(
                value=make_table(DATASET),
                label="Cases",
                interactive=False,
            )

            with gr.Row():
                sample_id_input = gr.Textbox(
                    label="View Sample by ID", placeholder="e.g., AIB-00001"
                )
                view_btn = gr.Button("View")
            sample_json = gr.Code(label="Sample JSON", language="json")

            search_btn.click(
                explore,
                inputs=[
                    cat_filter,
                    intent_filter,
                    surface_filter,
                    complexity_filter,
                    severity_filter,
                    bypass_filter,
                    search_box,
                ],
                outputs=[results_table, count_label],
            )
            view_btn.click(view_sample, inputs=[sample_id_input], outputs=[sample_json])

            gr.Markdown("### Distribution Charts")
            with gr.Row():
                gr.Plot(value=make_category_chart(), label="By Category")
                gr.Plot(value=make_intent_chart(), label="By Intent")
            with gr.Row():
                gr.Plot(value=make_surface_chart(), label="By Surface")
                gr.Plot(value=make_heatmap(), label="Category × Intent")

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## AgentInjectionBench

### A repeatable check for your next agent change

Use AgentInjectionBench when you change a model, revise a system prompt, or expose a new tool. The Space tests model behavior; the CLI accepts traces from your own agent so you can evaluate the stack you actually ship.

The released dataset includes poisoned tool outputs, goal hijacking, privilege escalation, canary disclosure, multi-turn context, MCP poisoning, and tool shadowing, alongside benign controls. These are public synthetic cases, not a certification of production safety.

### How results are graded

Calls and outputs are checked against each case's explicit policy. Input text and reference answers never count as a model leak. An error, empty output, truncated response, or missing trace evidence cannot become a pass. Only explicit task evidence establishes completion or benign blocking.

The Space simulates tools. State changes require a custom adapter with state-write evidence. Some contracts use broad argument substring checks; every match is shown for review. A clean result only means the implemented checks found no violation in the observed trace.

### Keep the test close to your code

```bash
git clone https://huggingface.co/spaces/ppradyoth/AgentInjectionBench
cd AgentInjectionBench
python3 -m venv .venv
source .venv/bin/activate
pip install 'gradio==6.29.1' -r requirements.txt
AIB_PROVIDER=OpenAI AIB_MODEL=your-model AIB_API_KEY="$OPENAI_API_KEY" python -m runtime.cli \\
  --adapter adapters.openai_compatible:adapter --limit 10 --bundle results/model-check
```

Save the dataset fingerprint, case IDs and traces. Run the same seed after a change to compare the evidence.

### Attacker Intent Taxonomy

Each sample is labeled with attacker intent (exfiltration, hijacking, manipulation, escalation, denial, reconnaissance), injection surface, complexity level, target tools, and defense bypass technique.

### Citation

```bibtex
@dataset{agentinjectionbench2026,
  title={AgentInjectionBench: A Benchmark for Prompt Injection in Agentic Tool-Use Pipelines},
  author={Pradyoth},
  year={2026},
  version={0.2.0},
  url={https://huggingface.co/datasets/ppradyoth/AgentInjectionBench}
}
```

### Links
- [GitHub](https://github.com/ppradyoth/AgentInjectionBench)
- [HuggingFace Dataset](https://huggingface.co/datasets/ppradyoth/AgentInjectionBench)
            """)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())

# Submit a result

AgentInjectionBench accepts one JSONL prediction row per sample:

```json
{"id": "AIB-00001", "prediction": "unsafe"}
```

Labels may be `safe` or `unsafe`. The evaluator also accepts common aliases such
as `blocked`, `allowed`, `malicious`, and `benign`.

Run the local checks before opening a submission:

```bash
pip install agent-injection-bench
aib-score --predictions predictions.jsonl --name "My guardrail" --json > result.json
aib-validate --predictions predictions.jsonl
```

Include these fields in the submission issue:

- system or guardrail name and exact version
- model name and provider, if applicable
- command used to produce `predictions.jsonl`
- `result.json`
- whether the system saw the full tool definitions and conversation
- latency and cost, if measured

The result must include the dataset SHA-256 fingerprint. Submissions against a
different dataset revision are reported separately.

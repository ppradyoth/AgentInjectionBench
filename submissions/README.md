# Leaderboard submissions

Add one complete prediction JSONL per system as
`submissions/<system-name>.jsonl`, then open a pull request. The leaderboard
workflow validates and renders every file alongside the built-in baselines.

Each file must contain one row for every public dataset ID. Run:

```bash
aib-validate --predictions submissions/my-system.jsonl
```

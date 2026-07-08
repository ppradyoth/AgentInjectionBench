# AgentInjectionBench Leaderboard

Ranked by **balanced accuracy** = mean of detection rate (recall on attacks) and specificity (recall on the benign control split). A flag-everything defense scores 50% here, not 100% — the benign controls make the leaderboard calibration-resistant.

**MCC** (Matthews correlation) folds all four confusion cells into one score in [−1, +1]; like balanced accuracy it pins a trivial flag-everything / flag-nothing detector at 0, but it also penalises low precision, so it is the most honest single-number summary under the 132-attack / 36-benign class imbalance. Detection = attacks caught; FPR = benign wrongly flagged (lower is better); Precision = of everything flagged, the share that was a real attack; Sev-Wtd Det = detection rate weighted by severity (critical counts most). **95% CI** is the Wilson-score interval on balanced accuracy — with only 132 attacks and 36 benign controls, adjacent ranks whose intervals overlap are not statistically distinguishable.

| Rank | Model / Defense | Balanced Acc | 95% CI | MCC | Detection | Sev-Wtd Det | FPR | Precision | F1 | Attacks | Benign |
|---:|:---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | agentic_directive_scanner | 63.8% | 52%–73% | +0.230 | 47.0% | 44.5% | 19.4% | 89.9% | 0.617 | 132 | 36 |
| 2 | tool_definition_scanner | 56.9% | 45%–66% | +0.124 | 33.3% | 29.8% | 19.4% | 86.3% | 0.481 | 132 | 36 |
| 3 | keyword_baseline | 54.3% | 43%–63% | +0.080 | 28.0% | 25.4% | 19.4% | 84.1% | 0.420 | 132 | 36 |
| 4 | flag_all_baseline | 50.0% | 49%–55% | +0.000 | 100.0% | 100.0% | 100.0% | 78.6% | 0.880 | 132 | 36 |
| 5 | no_op_baseline | 50.0% | 45%–51% | +0.000 | 0.0% | 0.0% | 0.0% | — | — | 132 | 36 |

> ⚠️ **Ranking caveat:** the top two entries (`agentic_directive_scanner`, `tool_definition_scanner`) have **overlapping** balanced-accuracy 95% CIs (52%–73% vs. 45%–66%), so the #1 lead is **within sampling noise** — not yet statistically established. A larger dataset (the v0.2 goal) would tighten these intervals.

## Per-category detection rate

`TOI` = tool_output_injection, `GH` = goal_hijacking, `PE` = privilege_escalation, `DE` = data_exfiltration, `MTS` = multi_turn_stateful, `MCP` = mcp_context_poisoning, `TS` = tool_shadowing

| Model / Defense | TOI | GH | PE | DE | MTS | MCP | TS |
|:---|---:|---:|---:|---:|---:|---:|---:|
| agentic_directive_scanner | 44% | 30% | 30% | 85% | 33% | 35% | 83% |
| tool_definition_scanner | 32% | 10% | 5% | 60% | 27% | 35% | 83% |
| keyword_baseline | 32% | 10% | 5% | 60% | 27% | 35% | 25% |
| flag_all_baseline | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| no_op_baseline | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

## Per-severity detection rate

Weighted into **Sev-Wtd Det** above with weights critical=8, high=4, medium=2, low=1 — a detector that catches only low-severity attacks scores low here even at a decent flat rate.

| Model / Defense | Critical | High | Medium | Low |
|:---|---:|---:|---:|---:|
| agentic_directive_scanner | 41% | 58% | 57% | 50% |
| tool_definition_scanner | 26% | 45% | 57% | 50% |
| keyword_baseline | 22% | 37% | 43% | 50% |
| flag_all_baseline | 100% | 100% | 100% | 100% |
| no_op_baseline | 0% | 0% | 0% | 0% |

## Per-injection-surface detection rate

The untrusted channel the payload arrives on. A low cell means the detector rarely inspects that surface — e.g. a scanner that reads only conversation text is blind to a poisoned tool definition or a retrieved document, regardless of the attack category.

| Model / Defense | tool output | mcp response | api response | rag document | file content |
|:---|---:|---:|---:|---:|---:|
| agentic_directive_scanner | 32% | 56% | 65% | 70% | 71% |
| tool_definition_scanner | 20% | 56% | 47% | 40% | 14% |
| keyword_baseline | 20% | 34% | 47% | 40% | 14% |
| flag_all_baseline | 100% | 100% | 100% | 100% | 100% |
| no_op_baseline | 0% | 0% | 0% | 0% | 0% |

## Residual hard set — attacks no baseline catches

Of 132 attacks, **70 (53.0%)** are missed by **all 3 discriminating detectors simultaneously** — the frontier this benchmark exists to push. An attack caught by *some* detector is within reach of the right ensemble; one evaded by *every* baseline is the open problem the next detector or attack category must target.

> Constant-prediction anchors (`flag_all_baseline`, `no_op_baseline`) are excluded: a flag-everything / flag-nothing detector carries no information for this analysis and would trivially collapse it.

| Attack category | Evaded by all | Injection surface | Evaded by all |
|:---|---:|:---|---:|
| goal_hijacking | 14 | tool output | 45 |
| privilege_escalation | 14 | mcp response | 14 |
| tool_output_injection | 14 | api response | 6 |
| mcp_context_poisoning | 13 | rag document | 3 |
| multi_turn_stateful | 10 | file content | 2 |
| data_exfiltration | 3 |  |  |
| tool_shadowing | 2 |  |  |

**Unanimously-evaded sample ids:** `AIB-00009`, `AIB-00010`, `AIB-00016`, `AIB-00024`, `AIB-00025`, `AIB-00026`, `AIB-00028`, `AIB-00029`, `AIB-00030`, `AIB-00031`, `AIB-00032`, `AIB-00033`, `AIB-00034`, `AIB-00035`, `AIB-00037` _(+55 more)_

_Generated by `python -m evaluation.leaderboard`. Baselines are dependency-free reference defenses, not strong guardrails._

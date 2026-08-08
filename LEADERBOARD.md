# AgentInjectionBench Leaderboard

Ranked by **balanced accuracy** = mean of detection rate (recall on attacks) and specificity (recall on the benign control split). A flag-everything defense scores 50% here, not 100% — the benign controls make the leaderboard calibration-resistant.

**MCC** (Matthews correlation) folds all four confusion cells into one score in [−1, +1]; like balanced accuracy it pins a trivial flag-everything / flag-nothing detector at 0, but it also penalises low precision, so it is the most honest single-number summary under the 142-attack / 40-benign class imbalance. Detection = attacks caught; FPR = benign wrongly flagged (lower is better); Precision = of everything flagged, the share that was a real attack; Sev-Wtd Det = detection rate weighted by severity (critical counts most). **95% CI** is the Wilson-score interval on balanced accuracy, and **MCC 95% CI** its bootstrap counterpart — MCC is non-linear in the four confusion cells, so it takes a resampling interval rather than a closed-form one — with only 142 attacks and 40 benign controls, adjacent ranks whose intervals overlap are not statistically distinguishable.

| Rank | Model / Defense | Balanced Acc | 95% CI | MCC | MCC 95% CI | Detection | Sev-Wtd Det | FPR | Precision | F1 | Attacks | Benign |
|---:|:---|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | control_channel_scanner | 73.6% | 62%–82% | +0.393 | +0.26–+0.51 | 64.8% | 66.3% | 17.5% | 92.9% | 0.763 | 142 | 40 |
| 2 | agentic_directive_scanner | 63.4% | 52%–72% | +0.229 | +0.10–+0.34 | 44.4% | 43.0% | 17.5% | 90.0% | 0.594 | 142 | 40 |
| 3 | tool_definition_scanner | 57.1% | 46%–65% | +0.130 | +0.00–+0.26 | 31.7% | 29.0% | 17.5% | 86.5% | 0.464 | 142 | 40 |
| 4 | keyword_baseline | 54.6% | 44%–63% | +0.089 | -0.04–+0.22 | 26.8% | 24.7% | 17.5% | 84.4% | 0.406 | 142 | 40 |
| 5 | flag_all_baseline | 50.0% | 49%–54% | +0.000 | +0.00–+0.00 | 100.0% | 100.0% | 100.0% | 78.0% | 0.877 | 142 | 40 |
| 6 | no_op_baseline | 50.0% | 46%–51% | +0.000 | +0.00–+0.00 | 0.0% | 0.0% | 0.0% | — | — | 142 | 40 |

> ⚠️ **Ranking caveat:** the top two entries (`control_channel_scanner`, `agentic_directive_scanner`) have **overlapping** balanced-accuracy 95% CIs (62%–82% vs. 52%–72%), so the #1 lead is **within sampling noise** — not yet statistically established. A larger dataset (the v0.2 goal) would tighten these intervals.

> 🔬 **Paired check (McNemar, chi2 continuity):** on the identical sample set the two differ **significantly** (p = 0.000; 0 samples only `agentic_directive_scanner` gets right vs. 29 only `control_channel_scanner` does), favouring `control_channel_scanner`.

## Per-category detection rate

`TOI` = tool_output_injection, `GH` = goal_hijacking, `PE` = privilege_escalation, `DE` = data_exfiltration, `MTS` = multi_turn_stateful, `MCP` = mcp_context_poisoning, `TS` = tool_shadowing

| Model / Defense | TOI | GH | PE | DE | MTS | MCP | TS |
|:---|---:|---:|---:|---:|---:|---:|---:|
| control_channel_scanner | 57% | 76% | 45% | 95% | 53% | 55% | 83% |
| agentic_directive_scanner | 43% | 29% | 27% | 85% | 29% | 32% | 83% |
| tool_definition_scanner | 32% | 10% | 5% | 60% | 24% | 32% | 83% |
| keyword_baseline | 32% | 10% | 5% | 60% | 24% | 32% | 25% |
| flag_all_baseline | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| no_op_baseline | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

## Per-severity detection rate

Weighted into **Sev-Wtd Det** above with weights critical=8, high=4, medium=2, low=1 — a detector that catches only low-severity attacks scores low here even at a decent flat rate.

| Model / Defense | Critical | High | Medium | Low |
|:---|---:|---:|---:|---:|
| control_channel_scanner | 68% | 60% | 62% | 50% |
| agentic_directive_scanner | 41% | 49% | 50% | 50% |
| tool_definition_scanner | 26% | 38% | 50% | 50% |
| keyword_baseline | 22% | 32% | 38% | 50% |
| flag_all_baseline | 100% | 100% | 100% | 100% |
| no_op_baseline | 0% | 0% | 0% | 0% |

## Per-injection-surface detection rate

The untrusted channel the payload arrives on. A low cell means the detector rarely inspects that surface — e.g. a scanner that reads only conversation text is blind to a poisoned tool definition or a retrieved document, regardless of the attack category.

| Model / Defense | tool output | mcp response | api response | rag document | file content |
|:---|---:|---:|---:|---:|---:|
| control_channel_scanner | 59% | 68% | 78% | 73% | 62% |
| agentic_directive_scanner | 31% | 53% | 61% | 64% | 62% |
| tool_definition_scanner | 20% | 53% | 44% | 36% | 12% |
| keyword_baseline | 20% | 32% | 44% | 36% | 12% |
| flag_all_baseline | 100% | 100% | 100% | 100% | 100% |
| no_op_baseline | 0% | 0% | 0% | 0% | 0% |

## Ensemble coverage — the best any combination can do

Flagging a sample when **any** of the 4 discriminating detectors flags it (an OR-ensemble) is the detection ceiling of the current baselines — but it accumulates every member's false positives, so the honest ceiling is a **detection / FPR pair**. The union catches **64.8%** of attacks at **17.5%** FPR (balanced accuracy 73.6%, MCC +0.393).

> Constant-prediction anchors (`flag_all_baseline`, `no_op_baseline`) are excluded: a flag-everything anchor would hit 100% detection at 100% FPR and a flag-nothing anchor adds nothing, so neither informs a best-real-combination analysis.

**Greedy minimal set** — add the detector that newly catches the most so-far-missed attacks (ties broken by smaller added FPR):

| Step | + Detector | New attacks | Cumulative detection | Cumulative FPR |
|---:|:---|---:|---:|---:|
| 1 | `control_channel_scanner` | +92 | 64.8% | 17.5% |

_Just **1 of 4** detectors reach the full union detection ceiling — the rest add no attack the others miss._

## Residual hard set — attacks no baseline catches

Of 142 attacks, **50 (35.2%)** are missed by **all 4 discriminating detectors simultaneously** — the frontier this benchmark exists to push. An attack caught by *some* detector is within reach of the right ensemble; one evaded by *every* baseline is the open problem the next detector or attack category must target.

> Constant-prediction anchors (`flag_all_baseline`, `no_op_baseline`) are excluded: a flag-everything / flag-nothing detector carries no information for this analysis and would trivially collapse it.

| Attack category | Evaded by all | Injection surface | Evaded by all |
|:---|---:|:---|---:|
| privilege_escalation | 12 | tool output | 29 |
| tool_output_injection | 12 | mcp response | 11 |
| mcp_context_poisoning | 10 | api response | 4 |
| multi_turn_stateful | 8 | file content | 3 |
| goal_hijacking | 5 | rag document | 3 |
| tool_shadowing | 2 |  |  |
| data_exfiltration | 1 |  |  |

**Unanimously-evaded sample ids:** `AIB-00009`, `AIB-00025`, `AIB-00037`, `AIB-00038`, `AIB-00040`, `AIB-00044`, `AIB-00045`, `AIB-00047`, `AIB-00049`, `AIB-00053`, `AIB-00055`, `AIB-00057`, `AIB-00059`, `AIB-00061`, `AIB-00066` _(+35 more)_

## Paired significance — McNemar's test

The table above ranks detectors by balanced accuracy with **unpaired** Wilson intervals. But every detector is scored on the **same samples**, so the sharper question — *is X really better than Y?* — is a **paired** one. McNemar's test conditions on the **discordant** samples (where exactly one detector is correct) and asks whether the split is more lopsided than a coin flip. Concordant samples (both right / both wrong) carry no signal and are ignored. p-values use the **exact binomial** test when discordant pairs are few (≤ 25) and the **continuity-corrected chi-square** otherwise; **bold** p-values are significant at α = 0.05.

> ⚠️ McNemar compares **overall accuracy**, so under this benchmark's 142-attack / 40-benign imbalance a constant *flag-everything* anchor can win a pair on raw accuracy alone — which is exactly why the headline ranking uses **balanced accuracy** and **MCC** (both pin that anchor at chance). Read this table as the paired significance of differences **between the real detectors**, a complement to — not a replacement for — the calibration-resistant ranking above.

| Detector A | Detector B | A-only right | B-only right | p-value | Method | Verdict |
|:---|:---|---:|---:|---:|:---|:---|
| `agentic_directive_scanner` | `control_channel_scanner` | 0 | 29 | **0.000** | χ² (cc) | `control_channel_scanner` better |
| `agentic_directive_scanner` | `flag_all_baseline` | 33 | 79 | **0.000** | χ² (cc) | `flag_all_baseline` better |
| `agentic_directive_scanner` | `keyword_baseline` | 25 | 0 | **0.000** | exact | `agentic_directive_scanner` better |
| `agentic_directive_scanner` | `no_op_baseline` | 63 | 7 | **0.000** | χ² (cc) | `agentic_directive_scanner` better |
| `agentic_directive_scanner` | `tool_definition_scanner` | 18 | 0 | **0.000** | exact | `agentic_directive_scanner` better |
| `control_channel_scanner` | `flag_all_baseline` | 33 | 50 | 0.079 | χ² (cc) | within noise |
| `control_channel_scanner` | `keyword_baseline` | 54 | 0 | **0.000** | χ² (cc) | `control_channel_scanner` better |
| `control_channel_scanner` | `no_op_baseline` | 92 | 7 | **0.000** | χ² (cc) | `control_channel_scanner` better |
| `control_channel_scanner` | `tool_definition_scanner` | 47 | 0 | **0.000** | χ² (cc) | `control_channel_scanner` better |
| `flag_all_baseline` | `keyword_baseline` | 104 | 33 | **0.000** | χ² (cc) | `flag_all_baseline` better |
| `flag_all_baseline` | `no_op_baseline` | 142 | 40 | **0.000** | χ² (cc) | `flag_all_baseline` better |
| `flag_all_baseline` | `tool_definition_scanner` | 97 | 33 | **0.000** | χ² (cc) | `flag_all_baseline` better |
| `keyword_baseline` | `no_op_baseline` | 38 | 7 | **0.000** | χ² (cc) | `keyword_baseline` better |
| `keyword_baseline` | `tool_definition_scanner` | 0 | 7 | **0.016** | exact | `tool_definition_scanner` better |
| `no_op_baseline` | `tool_definition_scanner` | 7 | 45 | **0.000** | χ² (cc) | `tool_definition_scanner` better |

_Generated by `python -m evaluation.leaderboard`. Baselines are dependency-free reference defenses, not strong guardrails._

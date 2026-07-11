#!/usr/bin/env python3
"""Render a markdown leaderboard from one or more evaluation results.

    python -m evaluation.leaderboard --baselines            # score built-ins
    python -m evaluation.leaderboard --baselines -o LEADERBOARD.md

Models/defenses are ranked by detection rate (descending). Pass extra
predictions files with ``--predictions name=path.jsonl`` to add real entries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.score import (
    SEVERITY_WEIGHTS,
    EvalResult,
    ensemble_coverage,
    load_dataset,
    load_predictions,
    normalize_label,
    pairwise_mcnemar,
    residual_hard_set,
    score_predictions,
)

CATEGORY_ORDER = [
    "tool_output_injection",
    "goal_hijacking",
    "privilege_escalation",
    "data_exfiltration",
    "multi_turn_stateful",
    "mcp_context_poisoning",
    "tool_shadowing",
]

# Most-dangerous first, so the per-severity block reads left-to-right worst-case.
SEVERITY_ORDER = ["critical", "high", "medium", "low"]

# Injection surfaces, most-prevalent first — the untrusted channel the payload
# rides in on. Exposes which ingestion surface a detector is blind to.
SURFACE_ORDER = [
    "tool_output",
    "mcp_response",
    "api_response",
    "rag_document",
    "file_content",
]


def _short(cat: str) -> str:
    return "".join(w[0] for w in cat.split("_")).upper()


def _find_pair(paired: dict | None, name_a: str, name_b: str) -> dict | None:
    """Return the McNemar pair result for two detectors regardless of order."""
    if not paired:
        return None
    for p in paired.get("pairs", []):
        if {p["detector_a"], p["detector_b"]} == {name_a, name_b}:
            return p
    return None


def _rank_key(r: EvalResult) -> tuple:
    """Ascending sort key giving: balanced accuracy desc (the calibration-
    resistant headline), then detection rate desc, then name asc — a
    deterministic total order. ``nan`` metrics sort last."""
    ba = r.balanced_accuracy
    dr = r.detection_rate
    return (
        -(ba if ba == ba else -1.0),
        -(dr if dr == dr else -1.0),
        r.name,
    )


def render_leaderboard(
    results: list[EvalResult],
    frontier: dict | None = None,
    ensemble: dict | None = None,
    paired: dict | None = None,
) -> str:
    has_benign = any(r.n_safe for r in results)
    ranked = sorted(results, key=_rank_key)

    lines: list[str] = []
    lines.append("# AgentInjectionBench Leaderboard")
    lines.append("")
    if has_benign:
        lines.append(
            "Ranked by **balanced accuracy** = mean of detection rate (recall on "
            "attacks) and specificity (recall on the benign control split). A "
            "flag-everything defense scores 50% here, not 100% — the benign "
            "controls make the leaderboard calibration-resistant."
        )
        lines.append("")
        lines.append("**MCC** (Matthews correlation) folds all four confusion cells into one "
                     "score in [−1, +1]; like balanced accuracy it pins a trivial "
                     "flag-everything / flag-nothing detector at 0, but it also penalises "
                     "low precision, so it is the most honest single-number summary under the "
                     "132-attack / 36-benign class imbalance. "
                     "Detection = attacks caught; FPR = benign wrongly flagged (lower is better); "
                     "Precision = of everything flagged, the share that was a real attack; "
                     "Sev-Wtd Det = detection rate weighted by severity (critical counts most). "
                     "**95% CI** is the Wilson-score interval on balanced accuracy, and "
                     "**MCC 95% CI** its bootstrap counterpart — MCC is non-linear in the four "
                     "confusion cells, so it takes a resampling interval rather than a closed-form "
                     "one — with only "
                     f"{ranked[0].n_unsafe} attacks and {ranked[0].n_safe} benign controls, "
                     "adjacent ranks whose intervals overlap are not statistically distinguishable.")
        lines.append("")
        lines.append("| Rank | Model / Defense | Balanced Acc | 95% CI | MCC | MCC 95% CI | Detection | Sev-Wtd Det | FPR | Precision | F1 | Attacks | Benign |")
        lines.append("|---:|:---|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|")
        for i, r in enumerate(ranked, 1):
            fpr = f"{r.false_positive_rate:.1%}" if r.n_safe else "—"
            prec = f"{r.precision:.1%}" if (r.n_detected + r.n_false_positive) else "—"
            f1 = f"{r.f1:.3f}" if r.f1 == r.f1 else "—"
            swd = f"{r.severity_weighted_detection:.1%}" if r.n_unsafe else "—"
            mcc = f"{r.mcc:+.3f}" if r.mcc == r.mcc else "—"
            mcc_lo, mcc_hi = r.mcc_ci
            mcc_ci_txt = f"{mcc_lo:+.2f}–{mcc_hi:+.2f}" if mcc_lo == mcc_lo else "—"
            ba_lo, ba_hi = r.balanced_accuracy_ci
            ci = f"{ba_lo:.0%}–{ba_hi:.0%}" if ba_lo == ba_lo else "—"
            lines.append(
                f"| {i} | {r.name} | {r.balanced_accuracy:.1%} | {ci} | {mcc} | {mcc_ci_txt} | {r.detection_rate:.1%} | "
                f"{swd} | {fpr} | {prec} | {f1} | {r.n_unsafe} | {r.n_safe} |"
            )
        # Is the #1 vs #2 gap real, or within sampling noise? Compare their
        # balanced-accuracy CIs — overlapping intervals mean the lead is not
        # statistically established at this sample size.
        if len(ranked) >= 2:
            top, second = ranked[0], ranked[1]
            t_lo, t_hi = top.balanced_accuracy_ci
            s_lo, s_hi = second.balanced_accuracy_ci
            if t_lo == t_lo and s_lo == s_lo:
                overlap = t_lo <= s_hi and s_lo <= t_hi
                lines.append("")
                if overlap:
                    lines.append(
                        f"> ⚠️ **Ranking caveat:** the top two entries "
                        f"(`{top.name}`, `{second.name}`) have **overlapping** "
                        f"balanced-accuracy 95% CIs ({t_lo:.0%}–{t_hi:.0%} vs. "
                        f"{s_lo:.0%}–{s_hi:.0%}), so the #1 lead is **within sampling "
                        f"noise** — not yet statistically established. A larger dataset "
                        f"(the v0.2 goal) would tighten these intervals."
                    )
                else:
                    lines.append(
                        f"> ✅ **Ranking note:** `{top.name}`'s balanced-accuracy 95% CI "
                        f"({t_lo:.0%}–{t_hi:.0%}) does **not** overlap the runner-up's "
                        f"({s_lo:.0%}–{s_hi:.0%}), so its lead is statistically distinguishable "
                        f"at this sample size."
                    )
            # Paired McNemar test on the identical sample set — the correct,
            # more-powerful complement to the unpaired CI-overlap check above.
            top_pair = _find_pair(paired, top.name, second.name) if paired else None
            if top_pair is not None:
                lines.append("")
                if top_pair["n_discordant"] == 0:
                    lines.append(
                        f"> 🔬 **Paired check (McNemar):** `{top.name}` and `{second.name}` "
                        f"agree on every sample (no discordant pairs), so they are "
                        f"indistinguishable on this dataset."
                    )
                elif top_pair["significant"]:
                    lines.append(
                        f"> 🔬 **Paired check (McNemar, {top_pair['method'].replace('_', ' ')}):** "
                        f"on the identical sample set the two differ **significantly** "
                        f"(p = {top_pair['p_value']:.3f}; {top_pair['a_correct_b_wrong']} samples "
                        f"only `{top_pair['detector_a']}` gets right vs. "
                        f"{top_pair['a_wrong_b_correct']} only `{top_pair['detector_b']}` does), "
                        f"favouring `{top_pair['better']}`."
                    )
                else:
                    lines.append(
                        f"> 🔬 **Paired check (McNemar, {top_pair['method'].replace('_', ' ')}):** "
                        f"even paired on the identical sample set, the #1 lead is **not** "
                        f"significant (p = {top_pair['p_value']:.3f}; discordant split "
                        f"{top_pair['a_correct_b_wrong']} vs. {top_pair['a_wrong_b_correct']}) — "
                        f"the ranking is within sampling noise."
                    )
    else:
        lines.append(
            "Detection rate = fraction of injection attacks correctly flagged. "
            "Attack-success rate (ASR) = attacks that slipped through (lower is better)."
        )
        lines.append("")
        lines.append("| Rank | Model / Defense | Detection Rate | Sev-Wtd Det | ASR | Attacks |")
        lines.append("|---:|:---|---:|---:|---:|---:|")
        for i, r in enumerate(ranked, 1):
            swd = f"{r.severity_weighted_detection:.1%}" if r.n_unsafe else "—"
            lines.append(
                f"| {i} | {r.name} | {r.detection_rate:.1%} | {swd} | "
                f"{r.attack_success_rate:.1%} | {r.n_unsafe} |"
            )
    lines.append("")

    # Per-category detection-rate matrix.
    cats = [c for c in CATEGORY_ORDER if any(c in r.by_category for r in ranked)]
    # include any category not in the canonical order
    for r in ranked:
        for c in r.by_category:
            if c not in cats:
                cats.append(c)

    lines.append("## Per-category detection rate")
    lines.append("")
    legend = ", ".join(f"`{_short(c)}` = {c}" for c in cats)
    lines.append(legend)
    lines.append("")
    header = "| Model / Defense | " + " | ".join(_short(c) for c in cats) + " |"
    sep = "|:---|" + "|".join(["---:"] * len(cats)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in ranked:
        cells = []
        for c in cats:
            g = r.by_category.get(c)
            cells.append(f"{g.detection_rate:.0%}" if g else "—")
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per-severity detection-rate matrix — exposes the "catches the easy stuff,
    # misses the dangerous stuff" failure mode a flat detection rate hides.
    sevs = [s for s in SEVERITY_ORDER if any(s in r.by_severity for r in ranked)]
    for r in ranked:
        for s in r.by_severity:
            if s not in sevs:
                sevs.append(s)

    lines.append("## Per-severity detection rate")
    lines.append("")
    lines.append(
        "Weighted into **Sev-Wtd Det** above with weights "
        + ", ".join(f"{s}={SEVERITY_WEIGHTS[s]}" for s in SEVERITY_ORDER if s in SEVERITY_WEIGHTS)
        + " — a detector that catches only low-severity attacks scores low here even at a decent flat rate."
    )
    lines.append("")
    header = "| Model / Defense | " + " | ".join(s.capitalize() for s in sevs) + " |"
    sep = "|:---|" + "|".join(["---:"] * len(sevs)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in ranked:
        cells = []
        for s in sevs:
            g = r.by_severity.get(s)
            cells.append(f"{g.detection_rate:.0%}" if g else "—")
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per-injection-surface detection-rate matrix — which untrusted ingestion
    # channel (tool output, MCP response, RAG document, …) a detector is blind
    # to, independent of attack category.
    surfaces = [s for s in SURFACE_ORDER if any(s in r.by_surface for r in ranked)]
    for r in ranked:
        for s in r.by_surface:
            if s not in surfaces:
                surfaces.append(s)

    if surfaces:
        lines.append("## Per-injection-surface detection rate")
        lines.append("")
        lines.append(
            "The untrusted channel the payload arrives on. A low cell means the "
            "detector rarely inspects that surface — e.g. a scanner that reads only "
            "conversation text is blind to a poisoned tool definition or a retrieved "
            "document, regardless of the attack category."
        )
        lines.append("")
        header = "| Model / Defense | " + " | ".join(s.replace("_", " ") for s in surfaces) + " |"
        sep = "|:---|" + "|".join(["---:"] * len(surfaces)) + "|"
        lines.append(header)
        lines.append(sep)
        for r in ranked:
            cells = []
            for s in surfaces:
                g = r.by_surface.get(s)
                cells.append(f"{g.detection_rate:.0%}" if g else "—")
            lines.append(f"| {r.name} | " + " | ".join(cells) + " |")
        lines.append("")

    # Ensemble coverage — the OR-ensemble ceiling and the greedy minimal set that
    # reaches it. The complement of the residual hard set: what the best
    # *combination* of baselines catches, and at what false-positive cost. Only
    # meaningful with ≥2 informative detectors.
    if ensemble and len(ensemble.get("detectors", [])) >= 2:
        u = ensemble["union"]
        lines.append("## Ensemble coverage — the best any combination can do")
        lines.append("")
        lines.append(
            f"Flagging a sample when **any** of the {len(ensemble['detectors'])} discriminating "
            f"detectors flags it (an OR-ensemble) is the detection ceiling of the current "
            f"baselines — but it accumulates every member's false positives, so the honest "
            f"ceiling is a **detection / FPR pair**. The union catches "
            f"**{u['detection_rate']:.1%}** of attacks at **{u['false_positive_rate']:.1%}** FPR "
            f"(balanced accuracy {u['balanced_accuracy']:.1%}, MCC {u['mcc']:+.3f})."
        )
        excluded = ensemble.get("excluded_detectors") or []
        if excluded:
            lines.append("")
            lines.append(
                "> Constant-prediction anchors ("
                + ", ".join(f"`{n}`" for n in excluded)
                + ") are excluded: a flag-everything anchor would hit 100% detection at 100% "
                "FPR and a flag-nothing anchor adds nothing, so neither informs a "
                "best-real-combination analysis."
            )
        lines.append("")
        lines.append(
            "**Greedy minimal set** — add the detector that newly catches the most "
            "so-far-missed attacks (ties broken by smaller added FPR):"
        )
        lines.append("")
        lines.append("| Step | + Detector | New attacks | Cumulative detection | Cumulative FPR |")
        lines.append("|---:|:---|---:|---:|---:|")
        for i, step in enumerate(ensemble["greedy"], 1):
            lines.append(
                f"| {i} | `{step['detector']}` | +{step['marginal_attacks_caught']} | "
                f"{step['cumulative_detection_rate']:.1%} | "
                f"{step['cumulative_false_positive_rate']:.1%} |"
            )
        lines.append("")
        n_used = len(ensemble["greedy"])
        n_inf = len(ensemble["detectors"])
        if n_used < n_inf:
            lines.append(
                f"_Just **{n_used} of {n_inf}** detectors reach the full union detection "
                f"ceiling — the rest add no attack the others miss._"
            )
            lines.append("")

    # Residual hard set — the attacks that evade *every* detector at once. Only
    # meaningful with ≥2 detectors: with one, "missed by all" is just its own
    # detection rate, already in the table above.
    if frontier and frontier.get("n_detectors", 0) >= 2:
        n_ev = frontier["n_evaded_by_all"]
        n_at = frontier["n_attacks"]
        rate = frontier["evasion_rate"]
        lines.append("## Residual hard set — attacks no baseline catches")
        lines.append("")
        lines.append(
            f"Of {n_at} attacks, **{n_ev} ({rate:.1%})** are missed by **all "
            f"{frontier['n_detectors']} discriminating detectors simultaneously** — the "
            "frontier this benchmark exists to push. An attack caught by *some* detector "
            "is within reach of the right ensemble; one evaded by *every* baseline is the "
            "open problem the next detector or attack category must target."
        )
        excluded = frontier.get("excluded_detectors") or []
        if excluded:
            lines.append("")
            lines.append(
                "> Constant-prediction anchors ("
                + ", ".join(f"`{n}`" for n in excluded)
                + ") are excluded: a flag-everything / flag-nothing detector carries no "
                "information for this analysis and would trivially collapse it."
            )
        lines.append("")
        if n_ev:
            lines.append("| Attack category | Evaded by all | Injection surface | Evaded by all |")
            lines.append("|:---|---:|:---|---:|")
            cat_rows = sorted(frontier["by_category"].items(), key=lambda kv: (-kv[1], kv[0]))
            surf_rows = sorted(frontier["by_surface"].items(), key=lambda kv: (-kv[1], kv[0]))
            for i in range(max(len(cat_rows), len(surf_rows))):
                c = f"{cat_rows[i][0]} | {cat_rows[i][1]}" if i < len(cat_rows) else " | "
                s = f"{surf_rows[i][0].replace('_', ' ')} | {surf_rows[i][1]}" if i < len(surf_rows) else " | "
                lines.append(f"| {c} | {s} |")
            lines.append("")
            ids = frontier["sample_ids"]
            shown = ids[:15]
            id_txt = ", ".join(f"`{i}`" for i in shown)
            more = f" _(+{len(ids) - len(shown)} more)_" if len(ids) > len(shown) else ""
            lines.append(f"**Unanimously-evaded sample ids:** {id_txt}{more}")
            lines.append("")
        else:
            lines.append(
                "_Every attack is caught by at least one baseline — no residual hard "
                "set at this dataset size._"
            )
            lines.append("")

    # Pairwise paired-significance matrix — the correct test for "is detector X
    # really better than Y", conditioning on the samples where exactly one is
    # right. Complements the per-detector table (unpaired) above.
    if paired and len(paired.get("pairs", [])) >= 1:
        lines.append("## Paired significance — McNemar's test")
        lines.append("")
        lines.append(
            "The table above ranks detectors by balanced accuracy with **unpaired** "
            "Wilson intervals. But every detector is scored on the **same samples**, so "
            "the sharper question — *is X really better than Y?* — is a **paired** one. "
            "McNemar's test conditions on the **discordant** samples (where exactly one "
            "detector is correct) and asks whether the split is more lopsided than a coin "
            "flip. Concordant samples (both right / both wrong) carry no signal and are "
            "ignored. p-values use the **exact binomial** test when discordant pairs are "
            "few (≤ 25) and the **continuity-corrected chi-square** otherwise; "
            "**bold** p-values are significant at α = 0.05."
        )
        lines.append("")
        lines.append(
            "> ⚠️ McNemar compares **overall accuracy**, so under this benchmark's "
            "132-attack / 36-benign imbalance a constant *flag-everything* anchor can win a "
            "pair on raw accuracy alone — which is exactly why the headline ranking uses "
            "**balanced accuracy** and **MCC** (both pin that anchor at chance). Read this "
            "table as the paired significance of differences **between the real detectors**, "
            "a complement to — not a replacement for — the calibration-resistant ranking above."
        )
        lines.append("")
        lines.append(
            "| Detector A | Detector B | A-only right | B-only right | p-value | Method | Verdict |"
        )
        lines.append("|:---|:---|---:|---:|---:|:---|:---|")
        for p in paired["pairs"]:
            pv = f"**{p['p_value']:.3f}**" if p["significant"] else f"{p['p_value']:.3f}"
            if p["n_discordant"] == 0:
                verdict = "identical"
            elif p["significant"]:
                verdict = f"`{p['better']}` better"
            else:
                verdict = "within noise"
            method = {
                "exact_binomial": "exact",
                "chi2_continuity": "χ² (cc)",
                "no_discordant": "—",
            }.get(p["method"], p["method"])
            lines.append(
                f"| `{p['detector_a']}` | `{p['detector_b']}` | {p['a_correct_b_wrong']} | "
                f"{p['a_wrong_b_correct']} | {pv} | {method} | {verdict} |"
            )
        lines.append("")

    lines.append(
        "_Generated by `python -m evaluation.leaderboard`. "
        "Baselines are dependency-free reference defenses, not strong guardrails._"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baselines", action="store_true", help="Score all built-in detectors")
    parser.add_argument(
        "--predictions",
        action="append",
        default=[],
        metavar="name=path.jsonl",
        help="Add a predictions file as a named entry (repeatable)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Write to a file instead of stdout")
    args = parser.parse_args(argv)

    samples = load_dataset()
    results: list[EvalResult] = []
    predictions_by_detector: dict[str, dict[str, str]] = {}

    if args.baselines:
        from evaluation.detectors import DETECTORS

        for name in DETECTORS:
            preds = {s["id"]: normalize_label(DETECTORS[name](s)) for s in samples}
            predictions_by_detector[name] = preds
            results.append(score_predictions(samples, preds, name=name))

    for spec in args.predictions:
        if "=" not in spec:
            parser.error(f"--predictions expects name=path, got {spec!r}")
        name, _, path = spec.partition("=")
        preds = load_predictions(Path(path))
        predictions_by_detector[name] = preds
        results.append(score_predictions(samples, preds, name=name))

    if not results:
        parser.error("Nothing to score: pass --baselines and/or --predictions")

    frontier = residual_hard_set(samples, predictions_by_detector)
    ensemble = ensemble_coverage(samples, predictions_by_detector)
    paired = pairwise_mcnemar(samples, predictions_by_detector) if len(results) >= 2 else None
    md = render_leaderboard(results, frontier=frontier, ensemble=ensemble, paired=paired)
    if args.output:
        args.output.write_text(md)
        print(f"Wrote {args.output} ({len(results)} entries)")
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

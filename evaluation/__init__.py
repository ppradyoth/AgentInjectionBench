"""Evaluation harness for AgentInjectionBench.

Turns the labelled dataset into a reproducible scoreboard: run a detector (or a
file of pre-computed predictions) over the benchmark and get attack-detection
rate, attack-success rate, and per-category / per-severity breakdowns, plus a
markdown leaderboard renderer.

Submodules are imported lazily (``from evaluation.score import ...``) so that
``python -m evaluation.score`` does not re-import the module after the package
package initialises.
"""

__all__ = ["score", "detectors", "leaderboard"]

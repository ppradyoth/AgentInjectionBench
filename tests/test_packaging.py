"""Packaging / entry-point regression tests.

The distribution is named ``agent-injection-bench`` but the importable code
lives in the top-level ``generation`` and ``evaluation`` packages. Hatchling
cannot auto-discover those from the distribution name, so ``pyproject.toml``
lists them explicitly under ``[tool.hatch.build.targets.wheel]``. If that
config is dropped, ``uv run`` / ``pip install -e .`` and every declared
console script silently break while ``python -m pytest`` keeps passing.

These tests turn that invariant into an enforced guard: each ``[project.scripts]``
entry point must resolve to a real callable, and the wheel package list must
cover every top-level package the entry points reference.
"""

import ast
import tomllib
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.load((ROOT / "pyproject.toml").open("rb"))
SCRIPTS = PYPROJECT["project"]["scripts"]


def _module_defines(module_path: str, attr: str) -> bool:
    """True if ``attr`` is defined at module top level, parsed via AST.

    Uses AST rather than importing so the check stays free of optional runtime
    deps (e.g. ``anthropic``) that the modules pull in at import time but that
    CI does not install.
    """
    spec = find_spec(module_path)
    assert spec is not None and spec.origin, f"module {module_path} is not importable"
    tree = ast.parse(Path(spec.origin).read_text())
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign)) and (
            getattr(node, "name", None) == attr
            or (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == attr for t in node.targets))
        )
        for node in tree.body
    )


@pytest.mark.parametrize("name,target", sorted(SCRIPTS.items()))
def test_console_script_entry_point_resolves(name, target):
    module_path, _, attr = target.partition(":")
    assert _module_defines(module_path, attr), (
        f"{name}: {target} does not resolve to a top-level definition"
    )


def test_wheel_packages_cover_entry_point_modules():
    declared = set(PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    referenced = {target.split(":")[0].split(".")[0] for target in SCRIPTS.values()}
    missing = referenced - declared
    assert not missing, f"entry points reference unpackaged top-level packages: {missing}"

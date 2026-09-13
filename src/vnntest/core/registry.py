"""Plugin registry for test generators and node types.

Adding a generator or a node-type folder does not require edits here.
Generators self-register on import; node types are inferred from case
folders and from the cases generators emit.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

from vnntest.core.paths import CASES_DIR
from vnntest.core.protocols import CaseSource, TestGenerator

_GENERATORS: dict[str, TestGenerator] = {}
_loaded = False


def register_generator(generator: TestGenerator) -> TestGenerator:
    """Register a generator plugin. Call this at module import time."""
    if not getattr(generator, "name", None):
        raise ValueError("TestGenerator must define a non-empty name")
    _GENERATORS[generator.name] = generator
    return generator


def load_plugins() -> None:
    """Import every module in ``vnntest.generators`` so they can register."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    import vnntest.generators as generators_pkg

    for module_info in pkgutil.iter_modules(generators_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        module_name = f"vnntest.generators.{module_info.name}"
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)


def get_generators() -> dict[str, TestGenerator]:
    load_plugins()
    return dict(_GENERATORS)


def get_cases(
    *,
    suite: str | None = None,
    generator: str | None = None,
    node_type: str | None = None,
) -> list[CaseSource]:
    """Return registered cases, optionally filtered."""
    load_plugins()
    if generator is not None:
        if generator not in _GENERATORS:
            known = ", ".join(sorted(_GENERATORS)) or "(none)"
            raise KeyError(f"Unknown generator {generator!r}. Registered: {known}")
        gens = [_GENERATORS[generator]]
    else:
        gens = list(_GENERATORS.values())

    cases: list[CaseSource] = []
    for gen in gens:
        for case in gen.generate():
            if suite is not None and case.suite != suite:
                continue
            if node_type is not None and case.node_type != node_type:
                continue
            cases.append(case)
    return cases


def get_node_types() -> list[str]:
    """Node types from case folders plus any types emitted by generators."""
    names: set[str] = set()
    if CASES_DIR.is_dir():
        for path in CASES_DIR.iterdir():
            if path.is_dir() and not path.name.startswith(("_", ".")):
                names.add(path.name)
    for case in get_cases():
        names.add(case.node_type)
    return sorted(names)


def reset_registry() -> None:
    """Clear plugins. Used by unit tests."""
    global _loaded
    _GENERATORS.clear()
    _loaded = False

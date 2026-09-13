"""Extension contracts for test generators and node-type cases.

To add coverage, implement these interfaces in a plugin module. Do not
edit this file, ``registry.py``, ``generator.py``, or ``runner.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable


class CaseSource(ABC):
    """A recipe for one solver query: an ONNX network plus a VNNLIB
    specification. Not yet built -- see ``generator.build_test_artifacts``
    for the generated, on-disk form the runner/evaluator actually use.

    Required attributes (set on the subclass or instance):
        id: unique name within the node type, e.g. ``"add01"``.
        node_type: folder-style name, e.g. ``"add"`` or ``"matmul"``.
        suite: which CLI suite this case belongs to, e.g. ``"mvp"``.
        generator: the plugin that produced this case.
    """

    id: str
    node_type: str
    suite: str = "mvp"
    generator: str = "custom"

    @abstractmethod
    def build_onnx_model(self):
        """Return an ONNX ModelProto for this case."""

    @abstractmethod
    def build_vnnlib_spec(self) -> str | list[str]:
        """Return VNNLIB 2.0 text as a string or a list of lines."""

    def expected_output(self) -> str | None:
        """Expected solver answer (``SAT`` / ``UNSAT``), if known."""
        return None

    def unexpected_result_message(self) -> str | None:
        """Human-readable explanation when the solver disagrees."""
        return None

    def description(self) -> str | None:
        """One-line summary of what this case checks, for the report."""
        return None


class TestGenerator(ABC):
    """A plugin that yields ``CaseSource`` objects.

    Drop a module under ``src/vnntest/generators/`` that instantiates a
    subclass and calls ``register_generator``. Core discovery will pick
    it up on the next run.
    """

    name: str
    suite: str = "mvp"

    @abstractmethod
    def generate(self) -> Iterable[CaseSource]:
        """Yield every case this generator knows how to produce."""

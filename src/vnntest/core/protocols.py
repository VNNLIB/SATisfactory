"""Agreed extension contracts: NodeType, TestCase, TestGenerator.

Adding a node type or generator must use these interfaces. Do not edit
this file, ``registry.py``, ``generator.py``, or ``runner.py`` to add
coverage. See README.md (How to add a new node type).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeType:
    """A neural-network operator the suite can test.

    Register a node type by creating ``src/vnntest/core/tests/<name>/``
    with case files. Do not edit core execution code. ``name`` must match
    the folder name and ``TestCase.node_type``.
    """

    name: str
    onnx_op: str = ""
    description: str = ""


class TestCase(ABC):
    """One solver query: an ONNX network plus a VNNLIB specification.

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


class TestGenerator(ABC):
    """A plugin that yields ``TestCase`` objects.

    Drop a module under ``src/vnntest/generators/`` that instantiates a
    subclass and calls ``register_generator``. Core discovery will pick
    it up on the next run.
    """

    name: str
    suite: str = "mvp"

    @abstractmethod
    def generate(self) -> Iterable[TestCase]:
        """Yield every case this generator knows how to produce."""

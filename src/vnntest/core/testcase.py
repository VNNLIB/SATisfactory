"""Adapter that turns a handwritten case file into a ``CaseSource``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from vnntest.core.paths import CASES_DIR
from vnntest.core.protocols import CaseSource


def load_case_module(path: Path) -> ModuleType:
    """Import a case file without putting it on the package search path."""
    module_name = f"vnnlib_satisfactory_case_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load test case from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ModuleCaseSource(CaseSource):
    """Wraps a ``*.py`` case file that defines ``build_onnx_model`` / ``build_vnnlib_spec``."""

    def __init__(
        self,
        path: Path,
        module: ModuleType,
        *,
        suite: str = "full",
        generator: str = "handwritten",
    ):
        self.source_path = path
        self.id = path.stem
        self.node_type = path.parent.name
        self.suite = suite
        self.generator = generator
        self._module = module

    @classmethod
    def try_from_path(
        cls,
        path: Path,
        *,
        suite: str = "full",
        generator: str = "handwritten",
    ) -> ModuleCaseSource | None:
        """Load ``path`` if it exposes the required build functions."""
        module = load_case_module(path)
        if not hasattr(module, "build_onnx_model") or not hasattr(module, "build_vnnlib_spec"):
            return None
        return cls(path, module, suite=suite, generator=generator)

    def build_onnx_model(self):
        return self._module.build_onnx_model()

    def build_vnnlib_spec(self) -> str | list[str]:
        return self._module.build_vnnlib_spec()

    def expected_output(self) -> str | None:
        fn = getattr(self._module, "return_expected_output", None)
        return fn() if callable(fn) else None

    def unexpected_result_message(self) -> str | None:
        fn = getattr(self._module, "unexpected_result_message", None)
        return fn() if callable(fn) else None

    def description(self) -> str | None:
        doc = getattr(self._module, "__doc__", None)
        return " ".join(doc.split()) if doc else None

    @property
    def relative_dir(self) -> Path:
        """Folder under ``CASES_DIR`` used for artifact placement."""
        try:
            return self.source_path.relative_to(CASES_DIR).parent
        except ValueError:
            return Path(self.node_type)

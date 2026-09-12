"""Build ONNX and VNNLIB artifacts for registered test cases."""

from __future__ import annotations

from pathlib import Path

import onnx

from vnntest.core.paths import ARTIFACTS_DIR, CACHE_DIR, CASES_DIR
from vnntest.core.protocols import TestCase
from vnntest.core.registry import get_cases
from vnntest.core.testcase import ModuleTestCase


def discover_tests():
    """Yield handwritten case file paths (compat helper for existing callers)."""
    for case in get_cases(generator="handwritten"):
        source = getattr(case, "source_path", None)
        if source is not None:
            yield source


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gitignore_path = CACHE_DIR / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("*\n")


def _as_case(test: Path | TestCase) -> TestCase | None:
    if isinstance(test, TestCase):
        return test
    return ModuleTestCase.try_from_path(Path(test))


def _artifact_subdir(case: TestCase) -> Path:
    relative = getattr(case, "relative_dir", None)
    if relative:
        return Path(relative)
    source = getattr(case, "source_path", None)
    if source is not None:
        try:
            return Path(source).relative_to(CASES_DIR).parent
        except ValueError:
            pass
    return Path(case.node_type)


def _vnnlib_text(spec: str | list[str]) -> str:
    if isinstance(spec, str):
        return spec if spec.endswith("\n") else spec + "\n"
    return "".join(line + "\n" for line in spec)


def build_test_artifacts(test: Path | TestCase):
    """Build ONNX and VNNLIB artifacts. Returns (onnx_path, vnnlib_path) or None."""
    case = _as_case(test)
    if case is None:
        return None

    _ensure_cache_dir()
    cache_dir = ARTIFACTS_DIR / _artifact_subdir(case)
    cache_dir.mkdir(parents=True, exist_ok=True)

    onnx_model = case.build_onnx_model()
    onnx_path = cache_dir / f"{case.id}.onnx"
    onnx.save(onnx_model, str(onnx_path))

    vnnlib_path = cache_dir / f"{case.id}.vnnlib"
    vnnlib_path.write_text(_vnnlib_text(case.build_vnnlib_spec()))

    return onnx_path, vnnlib_path

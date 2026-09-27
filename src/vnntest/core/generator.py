"""Build ONNX and VNNLIB artifacts for registered test cases."""

from __future__ import annotations

import json
from pathlib import Path

import onnx

from vnntest.core.paths import ARTIFACTS_DIR, CACHE_DIR
from vnntest.core.protocols import CaseSource
from vnntest.core.registry import get_cases


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gitignore_path = CACHE_DIR / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("*\n")


def _artifact_subdir(case: CaseSource) -> Path:
    relative = getattr(case, "relative_dir", None)
    return Path(relative) if relative else Path(case.node_type)


def _vnnlib_text(spec: str | list[str]) -> str:
    if isinstance(spec, str):
        return spec if spec.endswith("\n") else spec + "\n"
    return "".join(line + "\n" for line in spec)


def build_test_artifacts(case: CaseSource) -> tuple[Path, Path]:
    """Build ONNX and VNNLIB artifacts for a single case. Returns (onnx_path, vnnlib_path)."""
    _ensure_cache_dir()
    cache_dir = ARTIFACTS_DIR / _artifact_subdir(case)
    cache_dir.mkdir(parents=True, exist_ok=True)

    onnx_model = case.build_onnx_model()
    onnx_path = cache_dir / f"{case.id}.onnx"
    onnx.save(onnx_model, str(onnx_path))

    vnnlib_path = cache_dir / f"{case.id}.vnnlib"
    vnnlib_path.write_text(_vnnlib_text(case.build_vnnlib_spec()))

    return onnx_path, vnnlib_path


def _manifest_entry(case: CaseSource, onnx_path: Path, vnnlib_path: Path) -> dict:
    onnx_model = onnx.load(str(onnx_path))
    op_types = sorted({node.op_type for node in onnx_model.graph.node})

    # The op type this test is actually about, e.g. "matmul" -> "MatMul".
    # Other ops in the network (most commonly Constant, used to carry
    # fixed values) are construction details, not what's under test -- a
    # solver that folds Constant away without declaring it "supported"
    # shouldn't cause every test that happens to use one to be skipped.
    primary_op_type = next(
        (op for op in op_types if op.lower() == case.node_type.lower()), None
    )

    return {
        "name": case.id,
        "category": case.node_type,
        "onnx": onnx_path.name,
        "vnnlib": vnnlib_path.name,
        "expected": case.expected_output() or "UNKNOWN",
        "op_types": op_types,
        "primary_op_type": primary_op_type,
        "description": case.description() or "",
        "unexpected_message": case.unexpected_result_message(),
    }


def generate_all_tests(suite: str = "full", node_type: str | None = None) -> list[dict]:
    """Builds artifacts for every registered case in ``suite`` (optionally
    filtered to one node type), and writes manifest.json recording each
    test's expected verdict and reporting metadata. Returns the entries."""
    _ensure_cache_dir()
    manifest_entries = []

    for case in get_cases(suite=suite, node_type=node_type):
        onnx_path, vnnlib_path = build_test_artifacts(case)
        print(f"Generated {case.id} artifacts in {onnx_path.parent}")
        manifest_entries.append(_manifest_entry(case, onnx_path, vnnlib_path))

    manifest_path = CACHE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2))
    return manifest_entries

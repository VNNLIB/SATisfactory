"""Filesystem locations used by the test engine.

Case files live under ``CASES_DIR`` (one folder per node type). Generated
ONNX / VNNLIB artifacts are written under ``CACHE_DIR`` in the working
directory, matching the existing ``.vnntest/tests/<node>/<id>.*`` layout.
"""

from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent / "tests"
CACHE_DIR = Path(".vnntest")
ARTIFACTS_DIR = CACHE_DIR / "tests"

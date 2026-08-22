from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Verdict(str, Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, value: str) -> "Verdict":
        value = value.strip().lower()
        for v in cls:
            if v.value == value:
                return v
        return cls.UNKNOWN


@dataclass
class TestCase:
    name: str            # e.g. "add01"
    category: str        # e.g. "add" (subfolder under core/tests)
    onnx_path: Path
    vnnlib_path: Path
    expected: Verdict

    @property
    def id(self) -> str:
        return f"{self.category}/{self.name}"

    @classmethod
    def from_manifest_entry(cls, entry: dict, cache_dir: Path) -> "TestCase":
        return cls(
            name=entry["name"],
            category=entry["category"],
            onnx_path=cache_dir / entry["onnx"],
            vnnlib_path=cache_dir / entry["vnnlib"],
            expected=Verdict.from_str(entry["expected"]),
        )
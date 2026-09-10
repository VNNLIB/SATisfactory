from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


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
    op_types: list[str] = field(default_factory=list)  # ONNX op types in the model, e.g. ["Add"]
    primary_op_type: Optional[str] = None  # the op this test is actually about, e.g. "Add"
    description: str = ""
    unexpected_message: Optional[str] = None

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
            op_types=entry.get("op_types", []),
            primary_op_type=entry.get("primary_op_type"),
            description=entry.get("description", ""),
            unexpected_message=entry.get("unexpected_message"),
        )

import importlib.util
import json
import sys
from pathlib import Path

import onnx


def generate_all_tests():
    """Finds all test files and generates their ONNX and VNNLIB artifacts in
    the cache, plus a manifest.json recording each test's expected verdict."""
    base_cache_dir = Path(".vnntest")
    base_cache_dir.mkdir(parents=True, exist_ok=True)
    gitignore_path = base_cache_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("*\n")

    manifest_entries = []
    tests_dir = Path(__file__).parent / "tests"

    for test_file in tests_dir.rglob("*.py"):
        spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        if not (hasattr(module, "build_onnx_model") and hasattr(module, "build_vnnlib_spec")):
            continue

        subfolder = test_file.relative_to(tests_dir).parent
        cache_dir = base_cache_dir / "tests" / subfolder
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save ONNX
        onnx_model = module.build_onnx_model()
        onnx.save(onnx_model, str(cache_dir / f"{test_file.stem}.onnx"))

        # Generate and save VNNLIB
        vnnlib_lines = module.build_vnnlib_spec()
        with open(cache_dir / f"{test_file.stem}.vnnlib", "w") as f:
            if isinstance(vnnlib_lines, str):
                f.write(vnnlib_lines)
            else:
                f.writelines(line + "\n" for line in vnnlib_lines)

        print(f"Generated {test_file.stem} artifacts in {cache_dir}")

        expected = getattr(module, "return_expected_output", lambda: "UNKNOWN")()
        manifest_entries.append({
            "name": test_file.stem,
            "category": str(subfolder),
            "onnx": f"{test_file.stem}.onnx",
            "vnnlib": f"{test_file.stem}.vnnlib",
            "expected": expected,
        })

    manifest_path = base_cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2))

    return manifest_entries

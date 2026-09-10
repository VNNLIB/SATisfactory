import importlib.util
from pathlib import Path
import onnx

def discover_tests():
    """Finds all test files in the core/tests directory."""
    tests_dir = Path(__file__).parent / "tests"
    return tests_dir.rglob("*.py")

def build_test_artifacts(test_file: Path):
    """Builds ONNX and VNNLIB artifacts for a single test file. Returns (onnx_path, vnnlib_path) or None."""
    # Ensure the base cache directory and its .gitignore exist
    base_cache_dir = Path(".vnntest")
    base_cache_dir.mkdir(parents=True, exist_ok=True)
    gitignore_path = base_cache_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("*\n")

    import sys
    spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    
    # Check if the file has our required build functions
    if hasattr(module, "build_onnx_model") and hasattr(module, "build_vnnlib_spec"):
        tests_dir = Path(__file__).parent / "tests"
        subfolder = test_file.relative_to(tests_dir).parent
        cache_dir = Path(".vnntest/tests") / subfolder
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and Save ONNX
        onnx_model = module.build_onnx_model()
        onnx_path = cache_dir / f"{test_file.stem}.onnx"
        onnx.save(onnx_model, str(onnx_path))
        
        # Generate and Save VNNLIB
        vnnlib_lines = module.build_vnnlib_spec()
        vnnlib_path = cache_dir / f"{test_file.stem}.vnnlib"
        with open(vnnlib_path, "w") as f:
            if isinstance(vnnlib_lines, str):
                f.write(vnnlib_lines)
            else:
                f.writelines(line + "\n" for line in vnnlib_lines)
            
        return onnx_path, vnnlib_path
    return None
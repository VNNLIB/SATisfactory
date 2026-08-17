import importlib.util
from pathlib import Path
import onnx

def generate_all_tests():
    """Finds all test files and generates their ONNX and VNNLIB artifacts in the cache."""
    # Find all .py files inside core/tests/
    tests_dir = Path(__file__).parent / "tests"
    
    for test_file in tests_dir.rglob("*.py"):
        # Dynamically load the python file as a module
        import sys
        spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        # Check if the file has our required build functions
        if hasattr(module, "build_onnx_model") and hasattr(module, "build_vnnlib_spec"):
            # Determine where to save it (e.g. .vnntest/tests/relu/relu01.onnx)
            # test_file.relative_to(tests_dir).parent gets the subfolder (like 'relu')
            subfolder = test_file.relative_to(tests_dir).parent
            cache_dir = Path(".vnntest/tests") / subfolder
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate and Save ONNX
            onnx_model = module.build_onnx_model()
            onnx.save(onnx_model, str(cache_dir / f"{test_file.stem}.onnx"))
            
            # Generate and Save VNNLIB
            vnnlib_lines = module.build_vnnlib_spec()
            with open(cache_dir / f"{test_file.stem}.vnnlib", "w") as f:
                f.writelines(line + "\n" for line in vnnlib_lines)
                
            print(f"Generated {test_file.stem} artifacts in {cache_dir}")

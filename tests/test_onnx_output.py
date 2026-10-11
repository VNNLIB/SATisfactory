import glob
import onnx
from vnntest.core.generator import generate_all_tests

def test_onnx_structure():
    """
    Runs the full test suite to generate artifacts and then uses the onnx
    package to verify the structure of all generated .onnx files.
    """
    # Generate the full-suite artifacts directly.
    generate_all_tests(suite="full")

    # Find all generated .onnx files
    onnx_files = glob.glob(".vnnlib-satisfactory/tests/**/*.onnx", recursive=True)
    
    # Ensure some files were actually generated
    assert len(onnx_files) > 0, "No .onnx files were found. Artifact generation may have failed."
    
    # Verify the structure of each file using the onnx package
    for file_path in onnx_files:
        try:
            # load the model
            model = onnx.load(file_path)
            # check the model for structural validity
            onnx.checker.check_model(model)
        except Exception as e:
            raise AssertionError(f"Failed to verify ONNX model {file_path}:\n{e}")

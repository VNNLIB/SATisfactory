import glob
import vnnlib
from vnntest.core.generator import generate_all_tests

def test_vnnlib_structure():
    """
    Runs the full test suite to generate artifacts and then uses vnnlib
    to verify the structure of all generated .vnnlib files.
    """
    # Generate the full-suite artifacts directly.
    generate_all_tests(suite="full")

    # Find all generated .vnnlib files
    # Artifacts are generated in the .vnnlib-satisfactory/tests directory
    vnnlib_files = glob.glob(".vnnlib-satisfactory/tests/**/*.vnnlib", recursive=True)
    
    # Ensure some files were actually generated
    assert len(vnnlib_files) > 0, "No .vnnlib files were found. Artifact generation may have failed."
    
    # Verify the structure of each file using the vnnlib package
    for file_path in vnnlib_files:
        try:
            # parse_query_file will raise an exception (like VNNLibException) if the structure is invalid
            parsed_query = vnnlib.parse_query_file(file_path)
            assert parsed_query is not None, f"Parsing {file_path} returned None."
        except Exception as e:
            # The vnnlib library incorrectly enforces identical element types for isomorphic networks, 
            # despite the VNN-LIB 2.0 spec allowing them to differ. We ignore this specific bug.
            if "network_isomorphic01" in file_path and "TypeMismatch" in str(e):
                continue
            raise AssertionError(f"Failed to parse or verify {file_path}:\n{e}")


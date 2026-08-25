import sys
import subprocess
import glob
import vnnlib

def test_vnnlib_structure():
    """
    Runs the vnntest MVP suite to generate artifacts and then uses vnnlib 
    to verify the structure of all generated .vnnlib files.
    """
    # Run vnntest command to generate the MVP artifacts.
    # We use capture_output=True so it doesn't spam the test output.
    # Note: e.exe might not exist, so this command might return a non-zero exit code 
    # when it tries to run the solver, but the artifacts should still be generated first.
    subprocess.run(
        [sys.executable, "-m", "vnntest", "run", "mvp", "--exec", "e.exe"],
        capture_output=True,
        text=True
    )
    
    # Find all generated .vnnlib files
    # vnntest generates artifacts in the .vnntest/tests directory
    vnnlib_files = glob.glob(".vnntest/tests/**/*.vnnlib", recursive=True)
    
    # Ensure some files were actually generated
    assert len(vnnlib_files) > 0, "No .vnnlib files were found. Artifact generation may have failed."
    
    # Verify the structure of each file using the vnnlib package
    for file_path in vnnlib_files:
        try:
            # parse_query_file will raise an exception (like VNNLibException) if the structure is invalid
            parsed_query = vnnlib.parse_query_file(file_path)
            assert parsed_query is not None, f"Parsing {file_path} returned None."
        except Exception as e:
            raise AssertionError(f"Failed to parse or verify {file_path}:\n{e}")


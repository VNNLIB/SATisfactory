from vnntest.core.generator import build_test_artifacts
from vnntest.core.paths import CASES_DIR
from vnntest.core.protocols import CaseSource, TestGenerator
from vnntest.core.registry import (
    get_cases,
    get_generators,
    get_node_types,
    register_generator,
    reset_registry,
)


def setup_function():
    reset_registry()


def teardown_function():
    reset_registry()


def test_handwritten_and_precision_generators_register():
    gens = get_generators()
    assert "handwritten" in gens
    assert "precision" in gens
    assert gens["handwritten"].suite == "mvp"
    assert gens["precision"].suite == "precision"


def test_node_types_come_from_case_folders():
    expected_folders = sorted(
        path.name
        for path in CASES_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(("_", "."))
    )
    node_types = get_node_types()
    assert expected_folders
    assert node_types == expected_folders
    assert "add" in node_types
    assert "relu" in node_types


def test_handwritten_cases_include_existing_add_files():
    cases = get_cases(generator="handwritten", node_type="add")
    ids = {case.id for case in cases}
    assert ids == {"add01", "add02"}
    assert all(case.node_type == "add" for case in cases)
    assert all(case.suite == "mvp" for case in cases)


def test_suite_filter_separates_mvp_from_precision():
    mvp_cases = get_cases(suite="mvp")
    precision_cases = get_cases(suite="precision")
    assert len(mvp_cases) > 0
    assert precision_cases == []
    assert all(case.suite == "mvp" for case in mvp_cases)


def test_new_generator_is_visible_without_core_changes():
    class DummyCase(CaseSource):
        id = "dummy01"
        node_type = "dummy"
        suite = "mvp"
        generator = "dummy"

        def build_onnx_model(self):
            raise NotImplementedError

        def build_vnnlib_spec(self):
            return "(vnnlib-version <2.0>)"

    class DummyGenerator(TestGenerator):
        name = "dummy"
        suite = "mvp"

        def generate(self):
            yield DummyCase()

    register_generator(DummyGenerator())

    assert "dummy" in get_generators()
    assert "dummy" in get_node_types()
    dummy_cases = get_cases(generator="dummy")
    assert [case.id for case in dummy_cases] == ["dummy01"]


def test_build_test_artifacts_from_registered_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    case = next(c for c in get_cases(node_type="add") if c.id == "add01")
    onnx_path, vnnlib_path = build_test_artifacts(case)
    assert onnx_path.name == "add01.onnx"
    assert vnnlib_path.name == "add01.vnnlib"
    assert onnx_path.exists()
    assert "declare-network" in vnnlib_path.read_text()

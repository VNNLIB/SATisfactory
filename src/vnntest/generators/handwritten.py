"""Load handwritten node-type cases from ``src/vnntest/core/tests/``.

To add a node type: create ``src/vnntest/core/tests/<node>/`` and put
case files named ``<node>01.py``, ``<node>02.py``, ... Each file must
define ``build_onnx_model`` and ``build_vnnlib_spec``. Optional:
``return_expected_output`` and ``unexpected_result_message``.
"""

from vnntest.core.paths import CASES_DIR
from vnntest.core.protocols import TestGenerator
from vnntest.core.registry import register_generator
from vnntest.core.testcase import ModuleCaseSource


class HandwrittenGenerator(TestGenerator):
    name = "handwritten"
    suite = "full"

    def generate(self):
        if not CASES_DIR.is_dir():
            return
        for path in sorted(CASES_DIR.rglob("*.py")):
            if path.name.startswith("_") or path.name == "__init__.py":
                continue
            case = ModuleCaseSource.try_from_path(
                path,
                suite=self.suite,
                generator=self.name,
            )
            if case is not None:
                yield case


register_generator(HandwrittenGenerator())

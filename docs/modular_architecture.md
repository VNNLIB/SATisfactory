# Modular architecture

How VNN-LIB SATisfactory is extended. Source lives under `/src`, pytest files under `/tests`, and design notes (this file) under `/docs`, as described in `Contributing.md`.

## Layout

```
src/vnntest/
  core/                 engine — do not edit to add coverage
    protocols.py        TestCase / TestGenerator contracts
    registry.py         plugin discovery
    generator.py        writes ONNX + VNNLIB artifacts
    runner.py           invokes a solver
    tests/<node>/       handwritten cases, one folder per node type
  generators/           test-generator plugins (drop a module here)
  cli/                  command-line entry points
tests/                  pytest for the tool itself
docs/                   design notes and client communications
```

## Naming

| What | Pattern | Example |
|---|---|---|
| Node-type folder | lowercase ONNX op name | `src/vnntest/core/tests/add/` |
| Case file | `<node><nn>.py` (2-digit index) | `add01.py`, `add02.py` |
| Generator module | snake_case, matches `name` | `src/vnntest/generators/precision.py` |
| Generator class | `<Name>Generator` | `PrecisionGenerator` |

Existing polarity pairs use `01` for the UNSAT direction and `02` for the SAT direction when that split applies.

## Add a node type (no core rewrite)

1. Create `src/vnntest/core/tests/<node>/`.
2. Add `<node>01.py` (and `<node>02.py` if you need both SAT and UNSAT).
3. Each file must define:
   - `build_onnx_model()` → ONNX ModelProto
   - `build_vnnlib_spec()` → `str` or `list[str]`
   - optional `return_expected_output()` and `unexpected_result_message()`

The handwritten generator discovers the new folder on the next run:

```bash
vnnlib-satisfactory run --solver vibecheck --node-type add
```

## Add a test generator (no core rewrite)

1. Create `src/vnntest/generators/<name>.py`.
2. Subclass `TestGenerator`, implement `generate()`, set `name` and `suite`.
3. Call `register_generator(YourGenerator())` at module import.

Cases can be handwritten files (see `handwritten.py`) or classes you yield from `generate()` (see the commented example in `precision.py`).

---

## Questions for the client / mentor

Ask these after the plugin layout is in place. They decide *what* to generate next; the architecture is only *how* new work is plugged in.

### Coverage — which node types next?

1. Which ONNX operators should we add after the current MVP set (Add, Sub, Relu, MatMul, Gemm, Reshape, Unsqueeze, Shape, Constant)? For example Conv, MaxPool, Concat, BatchNormalization, Sigmoid, Tanh, Div, Mul, Clip.
2. Should every new node type still ship as a SAT/UNSAT pair, or do some operators need extra cases (broadcasting, different ranks, integer vs float, optional attributes)?
3. Are there operators the target solvers are known to handle incorrectly that we should prioritise?

### Generators — what kinds of tests, not just which ops?

4. Besides handwritten polarity cases, what other generators do you want? Candidates: floating-point precision / near-zero Relu, random bound sampling, multi-node graphs, broadcasting, and VNNLIB 2.0 features (disjunctions, multiple networks).
5. What should the empty `precision` suite actually measure? Tight bounds, subnormals, cancellation, or mixed-precision networks?
6. Should a generator be allowed to *synthesise* cases (parameters in, many ONNX+VNNLIB pairs out), or should every case stay hand-reviewed for a known expected answer?

### Correctness and scoring

7. When a solver returns `unknown`, times out, or crashes, should that count as a failure, a skip, or a separate category in the report?
8. Should expected answers stay `SAT`/`UNSAT` only, or do you also want witness/counterexample checks against ONNX Runtime?
9. Is VNN-COMP-style scoring (solved within timeout, no wrong SAT/UNSAT) the report you need, or a simpler pass/fail per case?

### Scope after the architecture

10. Do you want compositional tests (small graphs of several node types) in this project, or should we stay on single-operator networks?
11. Should we stay on VNNLIB 2.0 + ONNX only, or also emit older VNNLIB 1.0 queries for solvers that have not moved yet?
12. Which real solver(s) should we treat as the acceptance target after vibecheck — and is there a minimum opset / network size they must survive?

### Practical constraints

13. Are there licensing or redistribution limits on the ONNX models and VNNLIB files we generate (for example, can they be published with the tool)?
14. For the remaining sprints, what is higher value: more node types, richer generators, or a proper report that compares solvers side by side?

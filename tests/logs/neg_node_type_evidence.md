# Evidence: add Neg without touching core execution

Story subtask: prove a new node type can be added end to end without editing core.

## What was added

Only these case files (plus this log):

- `src/vnntest/core/tests/neg/neg01.py` (UNSAT: Y = -X, Y >= 1.0 is impossible)
- `src/vnntest/core/tests/neg/neg02.py` (SAT: Y = -X, Y <= -0.5 is reachable)

Not edited for this node type: `generator.py`, `registry.py`, `runner.py`, `cli/run.py`.

## Reproduce

```bash
pytest tests/test_registry.py
python -c "from vnntest.core.registry import get_cases, get_node_types; print(get_node_types()); print([c.id for c in get_cases(node_type='neg')])"
vnntest run mvp --solver vibecheck --node-type neg
```

## Recorded result (2026-09-16)

- `pytest tests/test_registry.py`: 9 passed
- Registry node types include `neg`
- Handwritten cases: `neg01`, `neg02`
- Artifacts written to `.vnntest/tests/neg/neg01.onnx`, `neg01.vnnlib`, `neg02.onnx`, `neg02.vnnlib` without changing the artifact builder

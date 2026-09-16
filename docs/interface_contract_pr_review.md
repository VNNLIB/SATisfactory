# Interface-contract review of open branches

Agreed contract (`src/vnntest/core/protocols.py`):

- Add a **node type** only under `src/vnntest/core/tests/<node>/` (or yield `TestCase`s from a generator plugin).
- Add a **generator** only under `src/vnntest/generators/` and call `register_generator()`.
- Do **not** edit `generator.py`, `registry.py`, `runner.py`, or `cli/run.py` to register new ops.

How to review a PR: `git diff --stat origin/main...HEAD` (or the PR files tab). Fail the review if a node-type PR changes those core files.

| Branch | Verdict | Notes |
|---|---|---|
| `origin/feat/maxpool` | Pass | Only `core/tests/maxpool/*.py`. |
| `origin/feat/mul` | Pass | Only mul case files vs `main`. |
| `origin/feature/flatten` | Pass | Only flatten case files vs `main`. |
| `origin/feat/slice` | Pass for node type | Slice cases live under `core/tests/slice/` (already on `main`). No extra core-engine edits in the two-dot diff. |
| `origin/feature/generated-report` | Hold | Edits `cli/run.py`, `core/generator.py`, and `core/runner.py`. Rebase onto the plugin registry and call `get_cases()` instead of re-implementing discovery. |
| `origin/feature/story7-compliance-checks` | Hold / discuss | Edits `core/runner.py`. Fine if it only wraps solver output; not fine if it special-cases node types. Should consume `TestCase` / expected output from the registry. |

`gh` is not installed in this environment, so this is a branch-diff review, not GitHub comment posts. Paste the table into PR review comments as needed.

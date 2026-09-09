"""Test-generator plugins.

Add a new generator by creating ``<name>.py`` in this folder that:

1. Subclasses ``vnntest.core.protocols.TestGenerator``
2. Implements ``generate()``
3. Calls ``register_generator(YourGenerator())`` at module level

The registry imports every module here automatically. Core engine files
do not need to change.
"""

---
trigger: always_on
---

# Jupyter Notebook Standards

1. **Imports from `src`**:
    - Notebooks must import core logic and indicators from `src`.
    - Do not define complex strategy classes or large functions inline.
    - Notebooks are for **orchestration**, **analysis**, and **visualization** only.

2. **Path Setup**:
    - Always include the standard `sys.path` setup block at the top to resolve project root.
    - Use `os.path.join` to locate files relative to the project root, not absolute paths.

3. **Sequential Execution**:
    - Notebooks must be runnable from top to bottom without errors.
    - Avoid relying on hidden state (out-of-order execution).

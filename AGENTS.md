# Agent instructions

## Python and virtual environment

- **Always use the project venv** before any Python invocations (running scripts, tests, or `python`/`pip` commands).
- From the repo root, invoke Python via the venv interpreter:
  - **Windows:** `venv\Scripts\python.exe` (e.g. `venv\Scripts\python.exe -m pytest ...`)
  - **Unix/macOS:** `venv/bin/python` (e.g. `venv/bin/python -m pytest ...`)
- Do not rely on the system or IDE Python unless the user explicitly requests it; using the project venv ensures correct dependencies and avoids import errors.

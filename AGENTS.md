# Repository Guidelines

## Project Structure & Module Organization
- Source lives under `symbolicplastic_snn/`: `core/`, `schedule/`, `conn/`, `realtime/`, `plasticity/`, `readout/`, `io/`, `encode/`, `runner/`, `utils/`.
- Tests are in `tests/` and follow `test_*.py` naming.
- Utility scripts in `scripts/` (e.g., `run_local.py`, `run_with_metrics.py`, `collect_metrics.py`).
- Transitional wrappers at repo root (`core/`, `topology/`, `readout/`, `monitor/`) exist for compatibility; prefer importing from `symbolicplastic_snn/*`.

## Build, Test, and Development Commands
- Setup: `python -m pip install -r requirements.txt` (Python 3.11, CPU; dependencies: NumPy, pytest only).
- Run all tests: `pytest -q` or `python -m unittest -v`.
- Filter tests: `pytest -k timewheel -q`.
- Run examples/bench: `python scripts/run_local.py`, `python scripts/run_with_metrics.py` then `python scripts/plot_metrics_svg.py`.

## Coding Style & Naming Conventions
- Follow PEP8, 4-space indent, type hints, and `dataclasses` where appropriate. Favor pure, side-effect-light functions.
- Determinism is mandatory: never use `numpy.random` or `random`. Use `SeedSpace/Stream` from `symbolicplastic_snn.utils.prng`.
  - Example: `from symbolicplastic_snn.utils.prng import SeedSpace; rng = SeedSpace(run_seed=0).derive("module=readout", "layer=2")`.
- Key naming for PRNG: `module=...`, `layer=...`, `pre=42->post=7`, etc.
- Respect fixed-point defaults when relevant (e.g., Q formats for `v`, `θ`, `λ`).

## Testing Guidelines
- Place unit tests under `tests/` with clear `test_*.py` names; mirror module structure when possible.
- Tests must be deterministic (derive RNG from `SeedSpace` with stable keys). The guard `tests/test_forbidden_random_usage.py` enforces no `numpy.random` / `random`.
- Add tests for new APIs and edge cases; keep runtime modest.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat(scope): summary`, `fix(scope): ...`, `refactor`, `test`, `chore` (see `git log`).
- PRs must include: clear description, motivation, approach, and any metrics or screenshots (use `scripts/collect_metrics.py` + `plot_metrics_svg.py` when relevant).
- Link related issues, update docs/README snippets if behavior changes, and ensure `pytest -q` passes.

# Contributing

Thanks for your interest in Protogrok-JAX.

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # 19 tests; JAX_PLATFORMS=cpu for laptops
```

## Guidelines

- **Keep the port faithful.** Any change to `models/protogrok.py` should preserve
  parity with the documented architecture (see the PyTorch→JAX mapping in the
  README) or clearly justify a divergence.
- **Tests are required.** New functionality needs a test in `tests/`. Shape,
  parameter-count, and numerical-sanity tests are preferred over golden files.
- **Purity for training code.** `train_step` / `eval_step` must remain pure
  functions of `(state, batch)` so they stay `jit`/`pmap`-safe.
- **No committed artifacts.** Checkpoints, datasets (`*.csv`, `*.pcap`), and the
  virtualenv are git-ignored — do not add them.
- **Style.** Type hints and module docstrings; run `pytest` before opening a PR.

## Reporting issues

Please include: the config used (`configs/*.yaml`), the exact command, the JAX
version (`python -c "import jax; print(jax.__version__)"`), and a minimal repro.

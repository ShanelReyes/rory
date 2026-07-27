# Contributing

## Reporting Issues

Open an issue on [GitHub](https://github.com/ShanelReyes/rory_core/issues) with:

- A clear description of the bug or feature request
- Steps to reproduce (for bugs)
- Python version and environment details

## Pull Requests

1. Fork the repository and create a feature branch
2. Follow existing code conventions (see below)
3. Add tests for new features
4. Run the full test suite before submitting:
   ```bash
   poetry run pytest -v -s
   poetry run ruff check .
   ```
5. Submit a PR against the `master` branch

## Code Conventions

- **Package layout**: No `__init__.py` at `rory/` or `rory/core/` level.
  Sub-packages under `rory/core/` each have their own `__init__.py`.
- **Naming**: Follow PEP 8. Classes use PascalCase, functions/methods use snake_case.
- **Type hints**: Use `npt.NDArray` for numpy arrays, `List[PyCtxt]` for CKKS ciphertexts.
- **Docstrings**: Google-style docstrings preferred.
- **Dependencies**: Use the `option` library for `Some`/`None` patterns.
- **Testing**: pytest with fixtures for shared setup (see `tests/conftest.py`).

## Development Setup

```bash
git clone https://github.com/ShanelReyes/rory_core.git
cd rory_core
poetry install -E pqc

# Generate test keys
poetry run python3 scripts/keygen.py --scheme=CKKS --mode=default \
  --output-path=/rory/keys/keys128 --security-level=128 --decimals=2 \
  --enable-relinearize --enable-rotate

# Run tests
poetry run pytest -v -s
```

## Documentation

Documentation is built with [Zensical](https://zensical.org/). To preview locally:

```bash
poetry run zensical serve
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

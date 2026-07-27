<!-- ![Rory](assets/logo.svg) -->
<div align="center">
    <img src="assets/logo.svg" width=200/>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/ShanelReyes/rory_core/refs/heads/master/pyproject.toml?token=GHSAT0AAAAAADZBM7BDRBZ7E2Q2SCJ5AU4U2QH767A&query=%24.tool.poetry.version&label=TestPyPI&logo=pypi&color=0A7ABC)](https://test.pypi.org/project/rory/)" alt="version">
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-blue" alt="python">
  <a href="https://codecov.io/gh/ShanelReyes/rory_core">
    <img src="https://codecov.io/gh/ShanelReyes/rory_core/branch/master/graph/badge.svg" alt="codecov">
  </a>
  <a href="https://github.com/ShanelReyes/rory_core/actions/workflows/run_tests.yml">
    <!-- <img src="https://github.com/ShanelReyes/rory_core/actions/workflows/run_tests.yml/badge.svg" alt="tests"> -->
  </a>
</div>

Rory is a **privacy-preserving machine learning** library providing secure
clustering, classification, and logistic regression using homomorphic
encryption &mdash; CKKS, Paillier, Liu, and FD-HOPE.

## Project Structure

```
rory/core/
├── security/          Cryptosystems (CKKS, Paillier, Liu, FD-HOPE) & data owners
├── clustering/        KMeans, NNC + secure variants (conventional/PQC)
├── classification/    KNN + secure PQC & distributed SKNN
├── machine_learning/  Logistic Regression & PPLR
├── interfaces/        Protocol types, manager/worker contracts
├── utils/             Constants, CKKS helpers, matrix operations
├── validation_index/  Clustering validity metrics
└── logger/            Logging utilities
```

## Setup

```bash
poetry install                 # core dependencies
poetry install -E pqc          # with CKKS support
```

## Testing

Generate CKKS keys first:

```bash
poetry run python3 scripts/keygen.py --scheme=CKKS --mode=default \
  --output-path=/rory/keys/keys128 --security-level=128 --decimals=2 \
  --enable-relinearize --enable-rotate
```

Run tests:

```bash
poetry run pytest -v -s                          # all tests
poetry run pytest tests/test_<name>.py -v -s     # single file
poetry run coverage run -m pytest -v -s          # with coverage
```

## Documentation

Built with [Zensical](https://zensical.org/). Run locally:

```bash
poetry run zensical serve
```

Open [http://localhost:8000](http://localhost:8000).

## Contributing

Open an issue or PR on [GitHub](https://github.com/ShanelReyes/rory_core).
Follow existing code conventions and add tests for new features. Run the
full test suite before submitting.

## License

MIT &mdash; see [LICENSE](LICENSE)

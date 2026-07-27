# Rory Common
Rory Common is a collection of common utilities and tools for the Rory platform.

## Getting Started

To get started with Rory Common, you can install it using Poetry. First, make sure you have Poetry installed on your system. Then, you can add Rory Common to your project by running the following command:

```sh
pip3 install poetry
poetry self add poetry-plugin-shell
```
## Installation
To install Rory Common, you can use pip:

```sh
poetry install
```

## Build
To build Rory Common, you can use the following command:

```sh
poetry build
```
## Usage
To use Rory Common in your project, you can import it as follows:
```python
from rory_common import Common
```

## Configuration

Copy `.env.test` to `.env` and fill in the values for your environment:

```sh
cp .env.test .env
```

Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `MICTLANX_URI` | `mictlanx://mictlanx-router-0@localhost:63666?...` | Storage node URI |
| `RORY_KEYS_PATH` | `/rory/keys` | CKKS key directory |
| `RORY_COMMON_ENV_FILE_PATH` | `./.env` | `.env` file loaded at import time |

See the [full configuration reference](docs/index.md#configuration) for all variables.
FROM python:3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root

COPY ./src .

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","6000","--timeout-keep-alive","3600","--workers","1"]

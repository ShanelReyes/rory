FROM python:3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV FLASK_APP="main"
ENV FLASK_ENV="production"

WORKDIR /app


COPY ./requirements.txt .

RUN pip install --default-timeout=1800 -r requirements.txt

COPY .env .
COPY ./src .

CMD ["gunicorn","--config","/app/gunicorn_config.py","main:app"]

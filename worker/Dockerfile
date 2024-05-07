FROM python:3.9-slim
WORKDIR /app
RUN apt update
RUN apt install -y git
RUN apt install -y gcc
RUN apt install -y python3-dev
RUN apt install -y gfortran
RUN apt install -y musl-dev
RUN apt install -y libmagic-dev
RUN python -m pip install --upgrade pip
ENV FLASK_APP="main"
ENV FLASK_ENV="production"
COPY ./requirements.txt .
RUN pip install --default-timeout=1800 -r requirements.txt
COPY ./src/ .
#CMD [ "python","-u","/app/main.py"]
CMD ["gunicorn","--config","/app/gunicorn_config.py","main:app"]
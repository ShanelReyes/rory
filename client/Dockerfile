FROM python:3.8-slim
WORKDIR /app
ARG BASE_PATH=/home/sreyes/rory
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
RUN pip install -r requirements.txt

# RUN useradd -s /bin/bash -m sreyes
# RUN mkdir /mictlanx
# RUN mkdir /rory
# RUN chown sreyes:sreyes /mictlanx
# RUN chown sreyes:sreyes /rory
# USER sreyes

COPY ./src .
#CMD [ "python","-u","/app/main.py"]
CMD ["gunicorn","--config","/app/gunicorn_config.py","main:app"]

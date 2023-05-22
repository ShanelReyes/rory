FROM python:3.8-slim
WORKDIR /app
COPY ./requirements.txt .
RUN apt update
RUN apt install -y git
RUN apt install -y gcc
RUN apt install -y python3-dev
RUN apt install -y gfortran
RUN apt install -y musl-dev
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY ./src .
ENV FLASK_APP="main"
ENV FLASK_ENV="production"
CMD [ "python","-u","/app/main.py"]

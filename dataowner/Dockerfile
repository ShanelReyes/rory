FROM python:3.10-slim
WORKDIR /app
RUN apt update
RUN apt-get update && apt-get install -y \
    git \ 
    g++ \
    gcc \
    cmake \
    make \
    python3-dev \
    gfortran \
    musl-dev \
    libssl-dev \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install --default-timeout=1800 -r requirements.txt
COPY ./src .
CMD [ "python","-u","/app/main.py"]

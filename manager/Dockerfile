# FROM python:3.10-bullseye
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FLASK_APP="main"
ENV FLASK_ENV="production"

WORKDIR /app

# Step 1: Install prerequisites
RUN apt-get update && apt-get install -y \
    software-properties-common \
    lzip git make sudo curl build-essential libssl-dev libffi-dev \
    zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev liblzma-dev \
    gcc g++ cmake ninja-build \
    && sudo apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Add deadsnakes PPA and install Python 3.10
RUN sudo add-apt-repository ppa:deadsnakes/ppa && \
    sudo apt-get update && \
    sudo apt-get install -y python3.10 python3.10-dev python3.10-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Step 3: Create symlinks for python3 and pip3
RUN sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    sudo update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1

    RUN pip3 install --upgrade pip setuptools wheel &&  pip3 install toml numpy cython



RUN git clone --recursive https://github.com/ibarrond/Pyfhel
# RUN cd Pyfhel && pip3 install .
RUN cd Pyfhel && sed -i \
    -e "s/isinstance(new_scheme, scheme_t)/isinstance(new_scheme, globals()['scheme_t'])/" \
    Pyfhel/PyPoly.pyx

RUN cd Pyfhel &&  python3.10 setup.py install

# RUN python -m pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install --default-timeout=1800 -r requirements.txt

COPY ./src .
CMD ["gunicorn","--config","/app/gunicorn_config.py","main:app"]
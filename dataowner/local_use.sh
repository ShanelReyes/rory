#!/bin/bash

# Activar el entorno virtual
source ./rory-env/bin/activate

# Definir variables de entorno
export ROOT_BASE_PATH=/rory
export BASE_PATH=/home/sreyes/rory
export MANAGER_PATH=$BASE_PATH/manager
export DATAOWNER_PATH=$BASE_PATH/dataowner
export WORKER_PATH=$BASE_PATH/worker
export MICTLANX_PATH=$BASE_PATH/mictlanx

export MANAGER_GUNICORN_CONFIG_FILE=$MANAGER_PATH/src/gunicorn_config.py
export WORKER_GUNICORN_CONFIG_FILE=$WORKER_PATH/src/gunicorn_config.py
export DATAOWNER_GUNICORN_CONFIG_FILE=$DATAOWNER_PATH/src/gunicorn_config.py

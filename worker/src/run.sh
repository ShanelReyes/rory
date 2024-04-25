#!/bin/bash

export BASE_PATH=/home/sreyes/rory
export WORKER_PATH=$BASE_PATH/worker
export WORKER_GUNICORN_CONFIG_FILE=$WORKER_PATH/src/gunicorn_config.py
gunicorn --config $WORKER_GUNICORN_CONFIG_FILE main:app

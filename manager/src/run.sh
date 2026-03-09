#!/bin/bash

export BASE_PATH=/home/sreyes/rory
export MANAGER_PATH=$BASE_PATH/manager
export MANAGER_GUNICORN_CONFIG_FILE=$MANAGER_PATH/src/gunicorn_config.py

gunicorn --chdir "$MANAGER_PATH/src" --config $MANAGER_GUNICORN_CONFIG_FILE main:app
#!/bin/bash

export BASE_PATH=/home/sreyes/rory
export CLIENT_PATH=$BASE_PATH/client
export CLIENT_GUNICORN_CONFIG_FILE=$CLIENT_PATH/src/gunicorn_config.py
export ENV_FILE_PATH=$CLIENT_PATH/.env.dev


gunicorn --chdir "$CLIENT_PATH/src" --config $CLIENT_GUNICORN_CONFIG_FILE main:app
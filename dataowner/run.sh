#!/bin/bash

export BASE_PATH=${1:-/home/sreyes/rory}
export DATAOWNER_PATH=$BASE_PATH/dataowner
export ENV_FILE_PATH=$DATAOWNER_PATH/.env.dev

uvicorn main:app \
  --host 0.0.0.0 \
  --port ${NODE_PORT:-3001} \
  --workers ${GUNICORN_WORKERS:-1} \
  --timeout-keep-alive ${GUNICORN_WORKER_TIMEOUT:-3600} \
  --reload \
  --log-level debug \
  --app-dir "$DATAOWNER_PATH/src"

#!/bin/bash

export BASE_PATH=${1:-/home/sreyes/rory}
export MANAGER_PATH=$BASE_PATH/manager
export RORY_MANAGER_ENV_FILE_PATH=$MANAGER_PATH/.env.dev

uvicorn main:app \
  --host 0.0.0.0 \
  --port ${NODE_PORT:-6000} \
  --workers ${GUNICORN_WORKERS:-1} \
  --timeout-keep-alive ${GUNICORN_WORKER_TIMEOUT:-3600} \
  --reload \
  --log-level debug \
  --app-dir "$MANAGER_PATH/src"

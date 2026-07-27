#!/bin/bash
readonly ENV_FILE=${1:-./envs/.env}
docker compose --env-file $ENV_FILE up --build
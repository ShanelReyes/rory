#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}
readonly IMAGE=${2:-rory:manager}

docker build -t ${IMAGE} ${BASE_PATH}/manager/

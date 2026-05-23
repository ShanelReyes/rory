#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}
readonly IMAGE_TAG=${2:-rory:manager}

# docker build -t ${IMAGE} ${BASE_PATH}/manager/
docker build -t shanelreyes/rory:${IMAGE_TAG} ${BASE_PATH}/manager/

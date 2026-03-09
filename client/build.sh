#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}

# readonly IMAGE_TAG=${2:-client}
readonly IMAGE=${2:-rory:client}

# docker build -t shanelreyes/rory:${IMAGE_TAG} ${BASE_PATH}/client/
docker build -t ${IMAGE} ${BASE_PATH}/client/

#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}

readonly IMAGE_TAG=${2:-dataowner}
# readonly IMAGE=${2:-rory:dataowner}

docker build -t shanelreyes/rory:${IMAGE_TAG} ${BASE_PATH}/dataowner/
# docker build -t ${IMAGE} ${BASE_PATH}/dataowner/

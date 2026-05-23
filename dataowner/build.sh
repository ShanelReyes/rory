#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}
readonly IMAGE_TAG=${2:-dataowner}

docker build -t shanelreyes/rory:${IMAGE_TAG} ${BASE_PATH}/dataowner/
# docker build -t shanelreyes/rory:dataowner ${BASE_PATH}/dataowner/

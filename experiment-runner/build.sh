#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}
readonly IMAGE_TAG=${2:-experiment-runner}

docker build -t shanelreyes/rory:${IMAGE_TAG} ${BASE_PATH}/experiment-runner/
# docker build -t shanelreyes/rory:experiment-runner ${BASE_PATH}/experiment-runner/

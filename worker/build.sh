#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}
readonly IMAGE_TAG=${2:-worker}

docker build -t shanelreyes/rory:${IMAGE_TAG} ${BASE_PATH}/worker/

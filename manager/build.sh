#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}
readonly IMAGE_TAG=${2:-manager}

docker build -t shanelreyes/rory:manager ${BASE_PATH}/manager/

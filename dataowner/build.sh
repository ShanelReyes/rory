#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}
docker build -t shanelreyes/rory:dataowner ${BASE_PATH}/dataowner/

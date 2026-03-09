#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}
readonly MANAGER_IMAGE_TAG=${2:-manager}
readonly CLIENT_IMAGE_TAG=${3:-client}
readonly WORKER_IMAGE_TAG=${4:-worker}

./build_all.sh $BASE_PATH $MANAGER_IMAGE_TAG $CLIENT_IMAGE_TAG $WORKER_IMAGE_TAG
./push_all.sh $MANAGER_IMAGE_TAG $CLIENT_IMAGE_TAG $WORKER_IMAGE_TAG


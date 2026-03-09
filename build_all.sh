#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}
readonly MANAGER_IMAGE_TAG=${2:-manager}
readonly CLIENT_IMAGE_TAG=${3:-client}
readonly WORKER_IMAGE_TAG=${4:-worker}

echo "Building Manager image - ${MANAGER_IMAGE_TAG}"
${BASE_PATH}/manager/build.sh $BASE_PATH $MANAGER_IMAGE_TAG
echo "Building Clietn image - ${CLIENT_IMAGE_TAG}"
${BASE_PATH}/client/build.sh $BASE_PATH $CLIENT_IMAGE_TAG 
echo "Building Worker image - ${WORKER_IMAGE_TAG}"
${BASE_PATH}/worker/build.sh $BASE_PATH $WORKER_IMAGE_TAG

${BASE_PATH}/dataowner/build.sh "$BASE_PATH"

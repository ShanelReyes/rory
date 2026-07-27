#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}
readonly MANAGER_IMAGE_TAG=${2:-manager}
readonly DATAOWNER_IMAGE_TAG=${3:-dataowner}
readonly WORKER_IMAGE_TAG=${4:-worker}
readonly EXPERIMENT_RUNNER_IMAGE_TAG=${5:-experiment-runner}

echo "Building Manager image - ${MANAGER_IMAGE_TAG}"
${BASE_PATH}/manager/build.sh $BASE_PATH $MANAGER_IMAGE_TAG
echo "Building Dataowner image - ${DATAOWNER_IMAGE_TAG}"
${BASE_PATH}/dataowner/build.sh $BASE_PATH $DATAOWNER_IMAGE_TAG 
echo "Building Worker image - ${WORKER_IMAGE_TAG}"
${BASE_PATH}/worker/build.sh $BASE_PATH $WORKER_IMAGE_TAG
echo "Building Experiment Runner image - ${EXPERIMENT_RUNNER_IMAGE_TAG}"
${BASE_PATH}/experiment-runner/build.sh $BASE_PATH $EXPERIMENT_RUNNER_IMAGE_TAG
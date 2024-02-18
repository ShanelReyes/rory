#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}

${BASE_PATH}/manager/build.sh $BASE_PATH
${BASE_PATH}/worker/build.sh $BASE_PATH
${BASE_PATH}/client/build.sh $BASE_PATH
${BASE_PATH}/dataowner/build.sh "$BASE_PATH"

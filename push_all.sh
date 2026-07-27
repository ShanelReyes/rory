#!/bin/bash
readonly MANAGER_IMAGE_TAG=${1:-manager}
readonly DATAOWNER_IMAGE_TAG=${2:-dataowner}
readonly WORKER_IMAGE_TAG=${3:-worker}
readonly EXPERIMENT_RUNNER_IMAGE_TAG=${4:-experiment-runner}

xs=($DATAOWNER_IMAGE_TAG $EXPERIMENT_RUNNER_IMAGE_TAG $MANAGER_IMAGE_TAG $WORKER_IMAGE_TAG)
for x in "${xs[@]}" 
do
	echo "Pushing shanelreyes/rory:${x} to Docker HUB"
	docker push shanelreyes/rory:$x
done

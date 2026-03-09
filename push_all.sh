#!/bin/bash
readonly MANAGER_IMAGE_TAG=${1:-manager}
readonly CLIENT_IMAGE_TAG=${2:-client}
readonly WORKER_IMAGE_TAG=${3:-worker}

xs=("dataowner" $CLIENT_IMAGE_TAG $MANAGER_IMAGE_TAG $WORKER_IMAGE_TAG)
for x in "${xs[@]}" 
do
	echo "Pushing shanelreyes/rory:${x} to Docker HUB"
	docker push shanelreyes/rory:$x
done

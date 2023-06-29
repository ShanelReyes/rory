#!/bin/bash
xs=("dataowner" "client" "manager" "worker")
for x in "${xs[@]}" 
do
	echo "Pushing shanelreyes/rory:${x} to Docker HUB"
	docker push shanelreyes/rory:$x
done

#!/bin/bash
docker rm -f $(docker ps -a --format '{{.ID}}') 

docker rmi $(docker images -f "dangling=true" --format '{{.ID}}')

/home/sreyes/rory/build_all.sh

docker compose -f /home/sreyes/rory/mictlanx/mictlanx.yml up -d

docker compose -f /home/sreyes/rory/rory_parallel.yml up -d


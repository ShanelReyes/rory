#!/bin/bash
readonly BASE_PATH=${1:-/home/sreyes/rory}
docker rm -f $(docker ps -a --format '{{.ID}}') 

docker rmi $(docker images -f "dangling=true" --format '{{.ID}}')

sudo rm -rf /mictlanx/mictlanx-peer-0/local/*
sudo rm -rf /mictlanx/mictlanx-peer-0/data/*
sudo rm -rf /mictlanx/mictlanx-peer-0/log/*

sudo rm -rf /mictlanx/mictlanx-peer-1/local/*
sudo rm -rf /mictlanx/mictlanx-peer-1/data/*
sudo rm -rf /mictlanx/mictlanx-peer-1/log/*

$BASE_PATH/build_all.sh "$BASE_PATH"

#docker compose -f /home/sreyes/rory/mictlanx/mictlanx-p1.yml up -d
docker compose -f ${BASE_PATH}/mictlanx/mictlanx.yml up -d

docker compose -f ${BASE_PATH}/rory_parallel.yml up -d

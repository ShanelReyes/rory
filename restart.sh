#!/bin/bash
docker rm -f $(docker ps -a --format '{{.ID}}') 

docker rmi $(docker images -f "dangling=true" --format '{{.ID}}')

sudo rm -rf /mictlanx/mictlanx-peer-0/local/*
sudo rm -rf /mictlanx/mictlanx-peer-0/data/*
sudo rm -rf /mictlanx/mictlanx-peer-0/log/*

/home/sreyes/rory/build_all.sh

docker compose -f /home/sreyes/rory/mictlanx/mictlanx-p1.yml up -d

docker compose -f /home/sreyes/rory/rory_parallel.yml up -d

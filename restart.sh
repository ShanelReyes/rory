#!/bin/bash
docker rm -f $(docker ps -a --format '{{.ID}}') 

docker rmi $(docker images -f "dangling=true" --format '{{.ID}}')

sudo rm -rf /mictlanx/mictlanx-peer-0/local/*
sudo rm -rf /mictlanx/mictlanx-peer-0/data/*
sudo rm -rf /mictlanx/mictlanx-peer-0/log/*

sudo rm -rf /mictlanx/mictlanx-peer-1/local/*
sudo rm -rf /mictlanx/mictlanx-peer-1/data/*
sudo rm -rf /mictlanx/mictlanx-peer-1/log/*

/home/sreyes/rory/build_all.sh

#docker compose -f /home/sreyes/rory/mictlanx/mictlanx-p1.yml up -d
docker compose -f /home/sreyes/rory/mictlanx/mictlanx.yml up -d

docker compose -f /home/sreyes/rory/rory_parallel.yml up -d

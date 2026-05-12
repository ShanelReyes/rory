#!/bin/bash

for i in {0..1}; do
  docker volume rm mictlanx-peer-$i-local
  docker volume rm mictlanx-peer-$i-logs
  docker volume rm mictlanx-peer-$i-data
done

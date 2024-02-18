#!/bin/bash
docker compose -f ./mictlanx.yml down
docker compose -f ./mictlanx.yml up -d

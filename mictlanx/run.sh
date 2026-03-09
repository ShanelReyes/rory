#!/bin/bash
docker compose -f ./router-static.yml down
docker compose -f ./router-static.yml up -d

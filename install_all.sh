#!/bin/bash
cd ~/rory/dataowner && poetry install --no-root
cd ~/rory/worker && poetry install --no-root
cd ~/rory/manager && poetry install --no-root
cd ~/rory/experiment-runner && pip3 install -r requirements.txt

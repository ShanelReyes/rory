#!/bin/bash
readonly TEST_FILE=${1:-"test_new_api.py"}
set -a 
source .env.test 
set +a 
coverage run -m pytest ./tests/$TEST_FILE -s

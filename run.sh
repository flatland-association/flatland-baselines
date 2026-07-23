#!/bin/bash
set -x
echo "/ start submission_template/run.sh"
set -e
python -m pip list
sleep 5
python run_solution.py
echo "\\ end submission_template/run.sh"

#!/bin/bash
set -euxo pipefail
source /home/conda/.bashrc
source activate base
conda activate flatland-baselines
export PYTHONPATH=$PWD
$@


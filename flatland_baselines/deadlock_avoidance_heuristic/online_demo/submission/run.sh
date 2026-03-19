#!/bin/bash
set -x
echo "/ start submission_template/run.sh"
set -e
find /tmp
source /home/conda/.bashrc
source activate base
conda activate flatland-rl
python -m pip list
if [ ! -z "$FLATLAND_BASELINES_REF" ]; then
  rm -fR /tmp/flatland-baselines
  git clone https://github.com/flatland-association/flatland-baselines.git /tmp/flatland-baselines
  cd /tmp/flatland-baselines
  git checkout $FLATLAND_BASELINES_REF
  export PYTHONPATH=/tmp/flatland-baselines:$PYTHONPATH
  find /tmp/flatland-baselines
  cd -
fi
if [ ! -z "$FLATLAND_RL_REF" ]; then
  rm -fR /tmp/flatland-rl
  git clone https://github.com/flatland-association/flatland-rl.git /tmp/flatland-rl
  cd /tmp/flatland-rl
  git checkout $FLATLAND_RL_REF
  export PYTHONPATH=/tmp/flatland-rl:$PYTHONPATH
  find /tmp/flatland-rl
  cd -
fi
printenv
python -c 'from flatland_baselines.deadlock_avoidance_heuristic.policy import deadlock_avoidance_policy; print(deadlock_avoidance_policy.__file__)'
python -c 'from flatland.envs import rail_env; print(rail_env.__file__)'
sleep 5
python run_solution.py
echo "\\ end submission_template/run.sh"

#!/bin/bash
set -x
echo "/ start evaluator/run.sh"
set -e
find /tmp
source /home/conda/.bashrc
source activate base
conda activate flatland-rl
python -m pip list
whoami
pwd
ls -al .
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
python -c 'from flatland.envs import rail_env; print(rail_env.__file__)'
python evaluator.py
echo "\\ end evaluator/run.sh"

#!/bin/bash
set -euxo pipefail
source /home/conda/.bashrc
source activate base
conda activate flatland-rl
export PYTHONPATH=$PWD

flatland-trajectory-generate-from-policy --policy-pkg flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy --policy-cls DeadLockAvoidancePolicy --obs-builder-pkg flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation --obs-builder-cls FullEnvObservation $@


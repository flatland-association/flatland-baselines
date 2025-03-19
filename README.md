# Flatland Benchmarks Flatland 3 Deadlock Avoidance Baseline

This repo is a heuristic baseline for the Flatland 3 Benchmarks hosted at [fab.flatland.cloud](https://fab.flatland.cloud).

[Flatland Benchmarks](https://github.com/flatland-association/flatland-benchmarks) (FAB) is an open-source web-based platform for running Benchmarks to foster
Open Research.

Flatland 3 Benchmarks follow up on the [Flatland 3 Challenge](https://flatland-association.github.io/flatland-book/challenges/flatland3.html).
More precisely, Flatland 3 Benchmarks follow Flatland 3 Challenge's
[Round 2 Environment Configurations](https://flatland-association.github.io/flatland-book/challenges/flatland3/envconfig.html#round-2), having the same
environment configuration but generated with different seeds.

This starterkit features a shortest path deadlock avoidance agent [run_solution.py](run_solution.py)

## TL;DR;

Run solution with debug environments:

```shell
docker compose  -f demo/docker-compose.yml up --force-recreate --build 
```
 
Output:

```text
evaluator-1       | ====================================================================================================
evaluator-1       | ####################################################################################################
evaluator-1       | EVALUATION COMPLETE !!
evaluator-1       | ####################################################################################################
evaluator-1       | # Mean Reward : 0.0
evaluator-1       | # Sum Normalized Reward : 5.0 (primary score)
evaluator-1       | # Mean Percentage Complete : 1.0 (secondary score)
evaluator-1       | # Mean Normalized Reward : 1.0
evaluator-1       | ####################################################################################################
evaluator-1       | ####################################################################################################
evaluator-1       | \ end grader
evaluator-1       | \ end evaluator/run.sh
evaluator-1       | + echo '\ end evaluator/run.sh'
submission-1 exited with code 0
evaluator-1 exited with code 0
```
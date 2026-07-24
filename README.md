# Flatland Baselines

[![Main](https://github.com/flatland-association/flatland-baselines/actions/workflows/checks.yaml/badge.svg?branch=main)](https://github.com/flatland-association/flatland-baselines/actions/workflows/checks.yaml)

🚂 This repo provides baselines for [Flatland](https://github.com/flatland-association/flatland-rl) environments,
i.e. controllers/agents that drive trains through such grid worlds.
Technically, a baselines implements `flatland.envs.RailEnvPolicy.RailEnvPolicy` interface.

📊 [Flatland Benchmarks](https://github.com/flatland-association/flatland-benchmarks) (FAB) is an open-source web-based platform for running Benchmarks to foster
Open Research. Flatland baselines can be used for the Flatland Benchmarks hosted at [fab.flatland.cloud](https://fab.flatland.cloud) and the scenarios in the
Flatland scenario repository [flatland-scenarios](https://github.com/flatland-association/flatland-scenarios).

Baselines provided:

* 🧲 [shortest path deadlock avoidance](flatland_baselines/deadlock_avoidance_heuristic). 👏Thanks
  to [aiAdrian](https://github.com/aiAdrian/flatland-benchmarks-f3-starterkit/tree/DeadLockAvoidancePolicy) for contributing!

## TL;DR;

Generate a trajectory from deadlock avoidance baseline:

```bash
conda env update -f environment.yml
conda activate flatland-baselines
mkdir output
PYTHONPATH=$PWD flatland-trajectory-generate-from-policy \
  --policy-pkg flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy --policy-cls DeadLockAvoidancePolicy \
  --obs-builder-pkg flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation --obs-builder-cls FullEnvObservation \
  --seed 42 \
  --callbacks-pkg flatland.callbacks.generate_movie_callbacks --callbacks-cls GenerateMovieCallbacks \
  --data-dir $PWD/output 
```

Output:

```text
 99%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌   | 466/471 [00:14<00:00, 35.00it/s]
Generating Thumbnail...
Generating Normal Video...
Videos :  ........../flatland-baselines/output/outputs/out.mp4 ........../flatland-baselines/output/outputs/out_thumb.mp4
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎| 470/471 [00:17<00:00, 27.18it/s]
```

N.B. movie generation requires `ffmpeg` installed - drop the `--callbacks-*` options to skip.

### Further CLI options

See the options for number of agents, grid size etc.:

```bash
flatland-trajectory-generate-from-policy --help
flatland-trajectory-generate-from-metadata --help
```

## Releases

* Release prs are automatically opened
  by [release-please](https://github.com/googleapis/release-please)/[release-please-action](https://github.com/marketplace/actions/release-please-action) based
  on [Conventional Commit Messages](https://www.conventionalcommits.org/en/v1.0.0/)
* [How do I change the version number?](https://github.com/googleapis/release-please?tab=readme-ov-file#how-do-i-change-the-version-number)
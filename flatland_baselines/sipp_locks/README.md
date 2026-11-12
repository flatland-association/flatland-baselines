# SIPP with corridor locks (ECML 2026 non-RL winner)

A centralised planner: it plans every train's timed path with safe-interval path planning (SIPP) over a shared
reservation table, locks long single-track corridors for one direction at a time, and repairs plans every step
(deviations, malfunctions, deadlock containment).

👏 Written by [darshanmakwana412](https://github.com/darshanmakwana412) for the
[ECML 2026 Challenge](https://flatland-association.github.io/flatland-book/challenges/ecml2026.html), where it won the
non-RL track with 20.8429 of 35 (submission
[v4-sipp-locks](https://competition.flatland.cloud/suites/6240c685-0fb4-481e-9404-47a570632227/c85d5fc2-15da-4a62-8e14-28d1261c29bd/submissions/9f2952a6-fb6b-4140-be5e-ca118baaa74e)).
The author's write-up, with pictures of the method, is the README of
[darshanmakwana412/ecml2026](https://github.com/darshanmakwana412/ecml2026).

## Provenance

- Source: [darshanmakwana412/ecml2026@efcdd89](https://github.com/darshanmakwana412/ecml2026/tree/efcdd89cc1044ba34dd52ec7e38c7e8d23b567b5),
  `submission/or_policy.py` and `submission/planner/`.
- Licence: MIT, see [LICENSE-ecml2026](LICENSE-ecml2026). It is the licence file of that repo, which it inherited from
  the ECML 2026 starterkit, so it names the Flatland Association.
- Changes against the source: only the import of `PlannerRuntime` in `or_policy.py` (`submission.planner` →
  `flatland_baselines.sipp_locks.planner`). The first commit of this directory holds the files unchanged.
- Not taken over: the aliases `my_policy.py`/`my_observation_builder.py` (the environment variables name the classes
  directly), `random_policy.py`, `policy_from_checkpoint.py`, `reinforcement_learning/` and `dev/`.

## Run

Needs flatland-rl 4.2.6, the competition's version (see "Limitations"); the repo's own pin is not used.

```bash
pip install flatland-rl==4.2.6
PYTHONPATH=$PWD flatland-trajectory-generate-from-policy --policy flatland_baselines.sipp_locks.or_policy.PlannerPolicy --obs-builder flatland_baselines.sipp_locks.or_policy.PlannerObsBuilder --rewards flatland.envs.rewards.ECML2026Rewards --env-path level_0_scenario_1.pkl --ep-id level_0_scenario_1 --data-dir $PWD/output
```

Image, built from the repo root (also built by `docker.yml` as `ghcr.io/flatland-association/flatland-baselines-sipp-locks`):

```bash
docker build -f flatland_baselines/sipp_locks/Dockerfile -t flatland-baselines-sipp-locks .
```

Test (CI job `test-sipp-locks`), with `level_0_scenario_1.pkl` from the
[ECML 2026 starterkit](https://github.com/flatland-association/ecml2026-starterkit/raw/refs/heads/main/reinforcement_learning/sampling/level_0_scenario_1.pkl):

```bash
ECML2026_PKL=$PWD/level_0_scenario_1.pkl PYTHONPATH=$PWD pytest flatland_baselines/sipp_locks/tests
```

## Settings

Environment variables read by `PlannerRuntime`; the defaults are the author's measured best and were used in the
competition:

| Variable | Default | Meaning |
|---|---|---|
| `PLANNER_TIME_BUDGET_S` | `1200` | Wall-clock budget per episode; past it, replanning effort is lowered. |
| `PLANNER_EXP_BUDGET` | `300000` | SIPP expansions per search. |
| `PLANNER_LOCK_MIN_LEN` | `20` | Minimum length of a single-track corridor that gets a directional lock. |
| `PLANNER_SERVE_STOPS` | `0` | `1` serves intermediate stops opportunistically. |
| `PLANNER_CORRIDOR_GUARD`, `PLANNER_OWNER_CHECK` | `1` | Safety layers at cell entry. |
| `PLANNER_DEADLOCK_EVERY` | `5` | Steps between deadlock checks. |
| `PLANNER_HEADWAY`, `PLANNER_EAGER_REPAIR`, `PLANNER_SHIFT_REPAIR`, `PLANNER_ESCAPE_MOVES`, `PLANNER_FLAT_BACKOFF`, `PLANNER_CASCADE`, `PLANNER_RELAXED` | `0` | Variants the author tested and rejected. |

## Limitations

- Runs on flatland-rl 4.2.x only: it calls `ObservationBuilder.reset()` without `env` and reads `agent.position`,
  both changed in 4.3.0. The port to flatland 5.0.0 follows in a second PR
  ([#88](https://github.com/flatland-association/flatland-baselines/issues/88)).
- Results depend on the machine: the wall-clock budget (`PLANNER_TIME_BUDGET_S`) lowers the search effort in long
  episodes, so scores are not bit-identical across hardware.
- Intermediate stops are skipped on purpose: under flatland-rl 4.2.6 rewards, stopping costs as much as a collision
  (−250·v), more than skipping a stop (−50).
- Tuned on the single ECML 2026 topology; other maps are untested here.

## Calibration

Against the ECML 2026 leaderboard (submission 9f2952a6, 20.8429): pending.

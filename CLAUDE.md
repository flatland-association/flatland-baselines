# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Baselines (policies/controllers) for [flatland-rl](https://github.com/flatland-association/flatland-rl), the
multi-agent grid-world train simulator. Each baseline implements flatland's
`flatland.envs.rail_env_policy.RailEnvPolicy` interface (`act`/`act_many`) and is used to generate/replay
trajectories for the [Flatland Benchmarks](https://fab.flatland.cloud) platform.

## Commands

Install:

```bash
conda env update -f environment.yml
conda activate flatland-baselines
```

Run the deadlock-avoidance-heuristic test suite (the main test suite in this repo):

```bash
PYTHONPATH=$PWD pytest flatland_baselines/deadlock_avoidance_heuristic/tests
```

Run a single test:

```bash
PYTHONPATH=$PWD pytest flatland_baselines/deadlock_avoidance_heuristic/tests/test_entering_prevention.py::test_without_entering_prevention_simultaneous_entry_deadlocks -v
```

`PYTHONPATH=$PWD` is required — the package isn't necessarily installed editable, so without it Python may pick up
a stale copy from `site-packages` instead of this working tree.

Generate a trajectory from the deadlock-avoidance baseline directly (see README.md for full CLI options via
`flatland-trajectory-generate-from-policy --help` / `flatland-trajectory-generate-from-metadata --help`):

```bash
PYTHONPATH=$PWD flatland-trajectory-generate-from-policy \
  --policy-pkg flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy --policy-cls DeadLockAvoidancePolicy \
  --obs-builder-pkg flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation --obs-builder-cls FullEnvObservation \
  --seed 42 --data-dir $PWD/output
```

Some tests need extra setup:

- `test_episodes_deadlock_avoidance.py` replays recorded episodes and requires `BENCHMARK_EPISODES_FOLDER` to point
  at an extracted copy of `FLATLAND_BENCHMARK_EPISODES_FOLDER_v5.zip` (see `benchmarks.benchmark_episodes.DOWNLOAD_INSTRUCTIONS`,
  which ships as part of flatland-rl, not this repo).
- `test_policy_grid_runner_evaluator.py` needs no extra setup — it drives `generate_trajectories_from_metadata`/
  `evaluate_trajectories_from_metadata` against `env_data/tests/service_test` fixtures that ship inside flatland-rl
  itself (packaged and pip-installed alongside `flatland`, not this repo).
- Tests marked `@pytest.mark.slow` (e.g. `test_regression_deadlock_avoidancy.py`, `online_demo/test_online.py`) spin
  up docker-compose stacks via `testcontainers` and are skipped with `-m "not slow"`. `checks.yaml`'s CI runs do
  *not* pass that filter, so these run in CI but are usually worth excluding locally unless you're specifically
  working on the Docker-based integration tests (see below) — they build images and can take minutes.

`tests/regen_benchmarks.py` is a maintenance script, not a pytest test (no `test_` prefix, so it isn't collected) —
it regenerates the same `BENCHMARK_EPISODES_FOLDER` trajectory fixtures `test_episodes_deadlock_avoidance.py`
replays, by re-running `DeadLockAvoidancePolicy` and diffing positions for conflicts. Edit the hardcoded relative
paths in its `__main__` block to point at a local `flatland-scenarios` checkout before running it directly
(`python flatland_baselines/deadlock_avoidance_heuristic/tests/regen_benchmarks.py`).

There is no lint/format tooling configured in this repo (no ruff/flake8/pre-commit config).

## The flatland-rl version pin

`flatland-rl` is a separate, fast-moving upstream package. The exact pinned ref/version is duplicated in three
places — all three must be bumped together (each has a `# DEPENDENCY SWITCH` comment marking it):

- `pyproject.toml` — `flatland-rl @ git+https://github.com/flatland-association/flatland-rl.git@v4.2.6`
- `environment.yml` — `flatland-rl[ml]==4.2.6`
- `.github/workflows/checks.yaml` — `env.flatland-rl-ref` (this may point at an unreleased branch rather than a
  release tag when this repo is being developed in tandem with an in-flight flatland-rl PR — check the current
  value rather than assuming it matches the other two files, which stay on the last released version since pip
  can't depend directly on a git ref for a published package)

Behavior can differ between this pin and whatever `flatland-rl` happens to be installed in an ambient environment
(e.g. a dev/pre-release build) — see `CLAUDE.local.md` for how to verify against the exact pinned version. This
extends to the `ObservationBuilder` base class interface itself: released versions call
`obs_builder.set_env(env)` once then `obs_builder.reset()` (no args) on every env reset, while some in-flight
flatland-rl branches drop `set_env` and merge it into `reset(self, env)` instead. `FullEnvObservation`
(`observation/full_env_observation.py`) tracks whichever interface `env.flatland-rl-ref` currently targets — check
its `reset` signature against the actual installed `flatland.core.env_observation_builder.ObservationBuilder`
before assuming compatibility.

## Architecture

### Baselines layout

Each top-level directory under `flatland_baselines/` (`deadlock_avoidance_heuristic`, `do_nothing_heuristic`,
`forever_heuristic`, `forward_only_heuristic`, `random`, `self_attention_ppo`, `tree_lstm_ppo`) is an independent
baseline with its own `Dockerfile` for submission as a container to the Flatland Benchmarks evaluator. The simple
ones are a few lines implementing `RailEnvPolicy.act`; `deadlock_avoidance_heuristic` is the substantial one.

### Deadlock avoidance heuristic (DLA)

Two-layer policy design in `deadlock_avoidance_heuristic/policy/`:

- `set_path_policy.py` — `SetPathPolicy` computes and caches each agent's fixed shortest path once (via Dijkstra,
  `_get_k_shortest_paths`), from its initial waypoint through any intermediate stops to its target. Agents follow
  this path deterministically; they don't replan unless forced to (see below).
- `deadlock_avoidance_policy.py` — `DeadLockAvoidancePolicy` (extends `SetPathPolicy`) runs a per-step check
  (`_start_step` → `_check_agent_can_move`): for each agent, it looks ahead along its own fixed path for any other
  agent occupying a cell there while facing the opposite direction ("opposing"), and only allows the agent to move
  if enough free cells remain before that opposing agent (`min_free_cell`). `DeadlockAvoidanceHeuristics` is a
  preconfigured subclass with rerouting-at-first-intermediate-stop and threshold-based dropping of stale
  intermediate stops enabled.

Key constructor knobs on `DeadLockAvoidancePolicy`: `min_free_cell`, `count_num_opp_agents_towards_min_free_cell`,
`use_switches_heuristic`, `use_entering_prevention`, `use_alternative_at_first_intermediate_and_then_always_first_strategy`,
`drop_next_threshold`, `k_shortest_path_cutoff`.

`observation/full_env_observation.py`'s `FullEnvObservation` just returns the whole `RailEnv` — DLA is centralized
(plans for all agents at once from global state), unlike per-agent tree observations.

**Known limitations** (see `deadlock_avoidance_heuristic/README.md` and, as tests, `tests/test_entering_prevention.py`,
`tests/test_limitations_switch.py` and `tests/test_limitations_two_switches.py`): the opposition check is
live-position-based and only looks one step ahead, so it can miss conflicts that resolve themselves only through
the environment's own low-level motion check (two agents both stepping into the same cell) rather than through DLA
foreseeing them. `use_entering_prevention` only blocks two agents becoming ready-to-depart and entering on the
exact same tick — it does not prevent a same-tick-delayed agent from entering later while the other is still
mid-journey. Two agents approaching each other through switches on opposite ends of a single-track segment with no
passing loop permanently deadlock at their own switch cell, one step short of ever entering the shared segment —
an instance of the "Require alternative path" problem class the method can't resolve on its own. The opposition
check is evaluated independently per agent along that agent's own full path: agents queuing behind a blocked
leader are stopped directly by DLA (not just by the env's motion check) as soon as their own target requires
passing the conflicting cell too; only agents whose own path stops short of the conflict keep receiving
MOVE_FORWARD indefinitely and rely purely on the motion check to hold their position in the queue.

### Testing patterns for DLA

Tests build tiny, fully deterministic `RailEnv`s by hand rather than using flatland's random generators, so exact
agent positions/directions/timing can be asserted on:

- Rail: `flatland.envs.rail_generators.rail_from_grid_transition_map(rail)` with a hand-built
  `RailGridTransitionMap`, built via the shared `tests/rail_test_utils.py:build_rail(rows)` helper — pass it a
  rectangular list of rows of `RailEnvTransitionsEnum` members (plain `0` for an empty cell) and it handles the
  `np.array(..., dtype=np.uint16)` construction and `RailGridTransitionMap(..., transitions=RailEnvTransitions(),
  grid=...)` wiring. Reference `RailEnvTransitionsEnum` members directly (e.g. `RailEnvTransitionsEnum.dead_end_from_west`,
  `.simple_switch_east_left`, `.simple_switch_north_left`, no `int(...)` cast needed since it's already an
  `IntEnum`) — it exposes every rotation of every cell type by name, so there's no need to call
  `transitions.rotate_transition(...)` by hand. Dead ends only have ONE valid occupancy direction, straight cells
  have two, and a `simple_switch_*` cell has a branch choice from one arm and forced turns from the other two
  (enumerate `RailEnvTransitionsEnum` members with `flatland.core.grid.grid4.fast_grid4_get_transitions` to work
  out exactly which arm connects which direction before wiring up agent start/target waypoints).
- Line: a custom `line_generator` callable returning a `flatland.envs.timetable_utils.Line` with explicit
  `agent_waypoints` (exact start/target `Waypoint`s per agent), instead of `sparse_line_generator`. Give it a
  `-> Line` return type hint and prefix its (required, but unused) positional params with `_` (e.g. `_rail,
  _num_agents, _hints, _num_resets, _np_random`) since flatland calls it positionally. A target waypoint's
  direction may need to be `None` (meaning "any direction") or expanded into one `Waypoint` per valid direction at
  that position depending on what the pinned flatland-rl version expects for `agent.waypoints[-1]` — check
  `RailEnv._agents_from_line` in the installed version if a hand-built line generator throws inside env `reset()`.
- Timetable: `flatland.envs.timetable_generators.ttgen_flatland2` (or a custom generator) to force
  `earliest_departure=0` for all agents — the default timetable generator randomizes departure windows, which
  defeats deterministic "both agents ready at the same tick" scenarios.
- Then either drive `policy.act_many(...)` / `env.step(...)` manually step-by-step to inspect exact per-tick
  behavior, or use `flatland.trajectories.policy_runner.PolicyRunner.create_from_policy(...)` for full-episode runs
  (used together with `flatland.envs.persistence.RailEnvPersister` for replaying recorded episodes).
- Bound every step-loop with a fixed iteration count (`for _ in range(N)`, with an `else: raise AssertionError(...)`
  if waiting for a specific state before proceeding) rather than an unbounded `while` — an unbounded wait loop
  hangs the whole test suite if a future regression stops agents short of the expected state.

Two tests take a different approach entirely — no hand-built `RailEnv`, just the CLI entry points end-to-end:
`test_policy_grid_runner_evaluator.py` drives `generate_trajectories_from_metadata`/`evaluate_trajectories_from_metadata`
over `env_data` fixtures, and `test_policy_runner.py` drives `generate_trajectory_from_policy` against a persisted
env pkl, both asserting on the resulting reward/trajectory output files. Both moved here from flatland-rl (which
kept the generic, non-DLA-specific test coverage of those same CLI mechanisms) because the specific
policy/obs-builder under test is `DeadLockAvoidancePolicy`/`FullEnvObservation`.

### Docker-based integration tests

Three places spin up multi-container `docker-compose` stacks via `testcontainers.compose.DockerCompose`, all
marked `@pytest.mark.slow`:

- `deadlock_avoidance_heuristic/tests/test_regression_deadlock_avoidancy.py` — runs
  `flatland-trajectory-generate-from-metadata` against recorded scenario sets inside a container built from
  `deadlock_avoidance_heuristic/Dockerfile`, then asserts on the resulting trajectories.
- `deadlock_avoidance_heuristic/online_demo/test_online.py` — brings up a redis-backed online evaluator +
  submission pair (mirroring the real Flatland Benchmarks competition infrastructure) and checks its results match
  offline evaluation (`offline_demo/test_offline.py`'s `verify_online_offline_calibration_envs_v2/v3_trunc`).
- `deadlock_avoidance_heuristic/offline_demo/` — trajectory-replay evaluation path (no Docker; uses
  `BENCHMARK_EPISODES_FOLDER` like `test_episodes_deadlock_avoidance.py`).

Both Docker-based fixtures follow the same shape: a `_containers_fixture` that writes a `.env` file, calls
`DockerCompose.stop()` then `.start()`, yields, then tears down — skippable entirely by setting `ATTENDED=True` if
you already have the stack running yourself (useful for iterating without paying the up/down cost each run).
Failure-path log dumping is shared via `deadlock_avoidance_heuristic/utils/docker_compose_helpers.py`
(`_print_output`, `_dump_compose_logs`) rather than duplicated per fixture — reuse it instead of inlining
`basic.get_logs()` calls when adding a new `_containers_fixture`. Distinguish `subprocess.CalledProcessError` (a
failed `docker compose up`, e.g. a build error — its own `.stdout`/`.stderr` bytes carry the real failure output,
since `get_logs()` often returns nothing when containers never came up) from other exceptions in the `except`
chain.

The three underlying Dockerfiles (`deadlock_avoidance_heuristic/Dockerfile`,
`online_demo/evaluator/Dockerfile`, `online_demo/submission/Dockerfile`) all build `FROM
ghcr.io/flatland-association/flatland-rl:<tag>` (default `latest-py3.12`, built from flatland-rl's `main`) and
accept an `ARG FLATLAND_RL_PIP_URL` that, if non-empty, force-reinstalls flatland-rl from that pip target
afterward — used to test against an unreleased flatland-rl branch without needing a published image for it. It
must be a `git+https://...@ref` URL, not a plain GitHub tarball URL: flatland-rl uses `setuptools_scm`, which needs
real `.git` metadata (stripped from tarball downloads) to infer a version. `checks.yaml` derives this from the same
`env.flatland-rl-ref` used for the plain pip install (see "The flatland-rl version pin" above), so bumping that one
value keeps the Python-level install and both Docker-image builds in sync. The `online_demo` compose file also has
an older, narrower runtime override (`FLATLAND_RL_REF`/`FLATLAND_BASELINES_REF` env vars, read by `evaluator/run.sh`
and `submission/run.sh` to `git clone`+`checkout` a ref onto `PYTHONPATH` at container startup) — this predates
`FLATLAND_RL_PIP_URL` and is used for the version-calibration matrix in `test_online.py`'s `parametrize` tables
(pinning specific historical flatland-rl/flatland-baselines commits to reproduce old reward calculations), not for
tracking current unreleased branches.

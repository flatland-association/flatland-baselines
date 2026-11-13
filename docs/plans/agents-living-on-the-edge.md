# Keep the baselines working with flatland-rl's "agents live on edges" redesign (flatland-rl#178)

> **Status:** plan of 2026-11-12, draft, written after the stack's commits (phases 1-7 done, review findings planned
> in phase 8). Based on `main` c1733cb (release 4.3.0) and
> flatland-rl [#541](https://github.com/flatland-association/flatland-rl/pull/541) (merged on 2026-11-13 as 9eec95df). Issue
> [flatland-rl#178](https://github.com/flatland-association/flatland-rl/issues/178) ("(D2) Remove Action Saver Class:
> agents live on "edges", actions are applied at cell entry", chenkins). Stack of seven PRs, top branch
> `178-agents-living-on-the-edge-19`, PR [#90](https://github.com/flatland-association/flatland-baselines/pull/90)
> (head b9e4a9f at the last CI run, chain in §11.2). Open questions: none (§9.1). CI on ffd2477 failed one test under
> pandas 3 (F-18), fixed in 4ef5ace (S-27); CI on b9e4a9f passes (S-28–S-31 and the pin on flatland-rl's `main`).
>
> **Why:** flatland-rl 5.0 changes when actions apply, how agents enter and stop, and whether a plain environment
> tracks the agents' state. The baselines, above all the deadlock-avoidance heuristic (DLA), must keep running
> and keep their benchmark results explainable at every step of that upstream series.

## 1. Introduction and goals

*What the change is for: the requirements, the qualities that matter, who has a stake, and what it leaves out.*

### 1.1 Requirements overview (functional requirements)

*The issue's text and the numbered requirements the rest of the plan refers to.*

> flatland-rl#178: Remove Action Saver Class: agents live on "edges", actions are applied at cell entry.

The stack follows the upstream PRs one by one: flatland-rl
[#501](https://github.com/flatland-association/flatland-rl/pull/501) (D2.3),
[#505](https://github.com/flatland-association/flatland-rl/pull/505) (D2.3c),
[#509](https://github.com/flatland-association/flatland-rl/pull/509) (D2.7),
[#513](https://github.com/flatland-association/flatland-rl/pull/513) (D1\*),
[#515](https://github.com/flatland-association/flatland-rl/pull/515) (D5/6),
[#520](https://github.com/flatland-association/flatland-rl/pull/520) (optional state machine) and
[#541](https://github.com/flatland-association/flatland-rl/pull/541) (state machine setting kept on reload, merged).

Requirements:

- **REQ-1** The baselines run against each upstream step: imports, renamed symbols and interfaces follow flatland-rl.
- **REQ-2** DLA decides with the new action timing: an action decides the transition at the next cell entry, not the
  current one.
- **REQ-3** DLA never sends an agent onto a branch its path does not take, also when a crossing can no longer be
  stopped.
- **REQ-4** Entering prevention sees every agent that enters the map in the coming step, whatever shortcut the state
  machine takes.
- **REQ-5** DLA sees the agents' correct state wherever it runs: tests, offline and online demo, trajectory runners.
  Since S-31 (b9e4a9f, Q-2) it derives the state itself, without requiring the state machine.
- **REQ-6** Every change of the expected results (golden values) is explained by a named upstream change.
- **REQ-7** Before the merge into `main`, the baselines depend on a released flatland-rl, so the package can be
  published to PyPI. Moved out of this PR on 2026-11-12 (§9.2): the pin to the released flatland-rl follows the merge,
  before release-please's release PR is merged (§1.4).

### 1.2 Quality goals (non-functional requirements)

*The few qualities the design must reach, each with a concrete scenario.*

| ID | Priority | Quality goal | Scenario |
|---|---|---|---|
| NFR-1 | 1 | Explainable results | A golden value changes only in a commit that names the upstream commit or PR causing it, found by running the old and new pin. |
| NFR-2 | 2 | Reproducible CI | CI checks out the same flatland-rl and episodes every run: pins are commit SHAs or release tags, not branches. |
| NFR-3 | 3 | Online equals offline | The online demo (Docker) and the offline demo give the same results, from the same constants. |

### 1.3 Stakeholders

*Who asks for the change, decides on it, reviews it or works with the result, and what they expect.*

| Role | Name | Expectations |
|---|---|---|
| Author, flatland-rl maintainer | chenkins | Each upstream step lands with a green baselines PR. |
| Benchmark users | Flatland Benchmarks (fab.flatland.cloud) | DLA containers keep working; result changes are explained. |
| Reviewers | flatland-association maintainers | Small PRs, each tied to one upstream PR. |

### 1.4 Follow-ups (not in this PR)

*What this PR leaves out, why, and where it goes later.*

| Follow-up | Why not in this PR | From | Findings | Issue |
|---|---|---|---|---|
| Pin the released flatland-rl 5.0.0 in all three places, after the stack is merged into `main` and before release-please's release PR is merged. | Waits for the flatland-rl 5.0.0 release; moved out of this PR on 2026-11-12 (formerly S-37). | REQ-7 | F-2, F-20 | untracked |
| Record the 50 benchmark episodes again with the current engine and replay them in the tests. | Needs a new archive in flatland-scenarios (another repo) and a review of the new results. | §2.1 (TODO) | F-5 | [flatland-scenarios#37](https://github.com/flatland-association/flatland-scenarios/issues/37) ([#94](https://github.com/flatland-association/flatland-baselines/issues/94) closed as its duplicate on 2026-11-13) |
| Pin flatland-rl's episodes URL (`checks.yml`, and `DOWNLOAD_INSTRUCTIONS` in `benchmarks/benchmark_episodes.py`) to the same flatland-scenarios commit as the baselines (S-33). | Another repo and another PR; part of updating flatland-scenarios and its users to flatland-rl 5.0. | S-33 | F-6 | [flatland-scenarios#37](https://github.com/flatland-association/flatland-scenarios/issues/37) |
| Apply the alternative path DLA picks when an agent is blocked, planned from the settled next entry point, and recompute the forced action after the path changes. | Fails on `main` too; applying it changes DLA's results and needs its own calibration. | §2.1 | F-9, F-25 | [#93](https://github.com/flatland-association/flatland-baselines/issues/93) |
| Let DLA hold off-map agents until their earliest departure itself, with the predicate entering prevention uses (`earliest_departure <= _elapsed_steps + 1`), once flatland-rl stops enforcing the earliest departure. | [flatland-rl#559](https://github.com/flatland-association/flatland-rl/issues/559) is open and its design (the meaning of `WAITING`, `action_required` for off-map agents) not decided; the pinned flatland-rl enforces it. | §2.2 | - | [flatland-rl#559](https://github.com/flatland-association/flatland-rl/issues/559) |

## 2. Constraints

*What limits the design from outside: platform, organisation and conventions.*

| ID | Constraint | Kind | Explanation |
|---|---|---|---|
| C-1 | flatland-rl is pinned in three places | convention | `pyproject.toml`, `environment.yml` and `checks.yaml`'s `env.flatland-rl-ref` move together (`# DEPENDENCY SWITCH`, `CLAUDE.md`). |
| C-2 | No direct git dependency on PyPI | technical | PyPI rejects a package whose dependencies are direct URLs (`pyproject.toml:24-26`). |
| C-3 | release-please publishes when its release PR is merged | organisational | `.github/workflows/release-please.yml`: every push to `main` updates the release PR; merging that PR publishes to PyPI, which fails while `main` carries a git pin. |
| C-4 | Upstream branches are deleted after their merge | organisational | All `178-agents-living-on-the-edge-*` branches of flatland-rl are gone; pins by branch name break. |
| C-5 | Python 3.11-3.14 | technical | flatland-rl 5.0 requires Python 3.11 or newer. | 

### 2.1 Limitations

*What the built code gets wrong or ignores, each with the TODO or issue that tracks it.*

Limitations (DLA's own list: `flatland_baselines/deadlock_avoidance_heuristic/README.md`, "Limitation" and "Known
weakness", and `CLAUDE.md`, "Known limitations"):

- The 50 recorded benchmark episodes are not replayed: the test is skipped
  (`tests/test_episodes_deadlock_avoidance.py:15`, TODO referring to flatland-scenarios#37). §1.4, F-5, [flatland-scenarios#37](https://github.com/flatland-association/flatland-scenarios/issues/37).
- DLA picks an alternative path for a blocked agent but never applies it (`_find_alternative`, on `main` too).
  §1.4, F-9, [#93](https://github.com/flatland-association/flatland-baselines/issues/93).
- `SetPathPolicy` on its own fails on an agent on the map: it calls a flatland-rl method that is private since 4.3.0.
  F-8, S-30.
- The DLA limitations documented before the stack stay (free space filled by other trains, two switches facing
  each other, loopy paths).

### 2.2 Assumptions

*What must hold for the design to be correct, but is not checked in code.*

Not checked in code; they hold by construction of the test setups and of flatland-rl's step order.

- DLA reads the agents' state between two steps, where flatland-rl documents `derived_state(elapsed_steps)` as equal
  to the tracked state (`agent_utils.py`, `derived_state`). Since S-31 (b9e4a9f) no env has to run the state
  machine for DLA (Q-2).
- flatland-rl decrements the malfunction counter and increments the step count before it reads the state
  transition signals, as at b99bfd11 (`rail_env_state_machine_wrapper.py:100-122`). Entering prevention mirrors that
  order.
- flatland-rl enforces the earliest departure: an off-map agent enters only once `earliest_departure <=
  _elapsed_steps` (`RailEnv._candidate_entry_points`, at 9eec95df), so DLA sends moving actions to agents not yet due
  and entering prevention predicts entry from the same rule (S-29), as the flatland-book's 5.0.0 draft describes it
  ([Speed Profiles](https://github.com/flatland-association/flatland-book/blob/0dd2d374df5e02d67aac6ec8dd11b3eb14e07931/environment/environment/speed_profiles.ipynb), "Step counting and departure", [flatland-book#42](https://github.com/flatland-association/flatland-book/pull/42) at 0dd2d37). [flatland-rl#559](https://github.com/flatland-association/flatland-rl/issues/559) proposes to drop this
  enforcement; then DLA has to wait itself (§1.4).
- The online and offline demos run the same episodes with the same seeds, so they can share their expected values.
- GitHub's stack merge is available for the repository when the stack is merged (Q-3); the docs mark it as a
  public preview on GitHub Enterprise Cloud.
- flatland-rl#541 is merged before the stack (on 2026-11-13, as 9eec95df); since 21a3c5f the baselines pin
  flatland-rl's `main` branch (F-16, F-17), until the follow-up pins flatland-rl 5.0.0 (§1.4). New commits on
  flatland-rl's `main` reach CI without a change here.

## 3. Context and scope (checked 2026-11-12 on `main` c1733cb)

*The code on the base branch that the change works in, as it is before the change, and where the change stops.*

Context on `main` (pinned to flatland-rl 4.3.0):

- `deadlock_avoidance_heuristic/policy/set_path_policy.py`: `SetPathPolicy` keeps one fixed shortest path per agent
  and acts toward its next waypoint from `agent.current_configuration`.
- `deadlock_avoidance_heuristic/policy/deadlock_avoidance_policy.py`: `DeadLockAvoidancePolicy` runs a check per
  step (`_start_step` → `_extract_agent_can_move` → `_check_agent_can_move`) and sends `STOP_MOVING` to an agent that
  has too few free cells before an opposing agent. Entering prevention (`use_entering_prevention`) compares the
  `READY_TO_DEPART` agents only.
- `observation/full_env_observation.py`: `FullEnvObservation` returns the whole env (`ObservationType`).
- `online_demo/`, `offline_demo/`: the same expected values (golden values) for the online (Docker) and offline
  evaluation.
- Tests: hand-built envs (`test_entering_prevention.py`, `test_limitations_*.py`), CLI end-to-end runs
  (`test_policy_runner.py`, `test_policy_grid_runner_evaluator.py`) and the replay of recorded episodes
  (`test_episodes_deadlock_avoidance.py`).
- CI (`checks.yaml`): flatland-rl at `v4.3.0`, episodes archive v5 from flatland-scenarios `main`, Python 3.10-3.14.

Scope: the stack changes DLA, `SetPathPolicy`, `FullEnvObservation`, the online demo's client, the tests, CI and the
pins. The simple baselines (`do_nothing`, `forever`, `forward_only`, `random`) and the Dockerfiles stay as they are.

## 4. Solution strategy

*The fundamental choices and why, the rules the design follows, how it is verified, and the design of each part.*

| Quality goal | Scenario | Solution approach | Details |
|---|---|---|---|
| NFR-1 Explainable results | A golden value changes | One baselines PR per upstream PR; each recalibration names its cause, found by testing the old and new pin. | §4.3, §4.9 |
| NFR-2 Reproducible CI | Same inputs every run | Pin flatland-rl by commit SHA (branch name as a comment), the episodes by a commit (S-33), the release by tag (follow-up, §1.4). | §7, §11.2 |
| NFR-3 Online equals offline | Shared constants | Keep the online demo importing the offline demo's expected values. | §4.9 |

### 4.1 Design principles

*The rules the design follows, each with its decision, the concepts it reuses instead of building new ones, and what
is not new on purpose.*

- **DP-1 Follow upstream, one PR per upstream step.** Each baselines PR pins one upstream state and only adapts to
  it. (decision of 2026-08-18)
- **DP-2 A recalibration names its cause.** Expected values change only with the upstream change that moved them,
  located by running the tests on both sides of it. (decision of 2026-08-25)
- **DP-3 DLA reads the engine, it does not re-implement it.** Where DLA must predict the engine (entering,
  crossings), it mirrors the engine's own conditions and says where they come from. (F-3, F-4)
- **DP-4 DLA derives the agents' state itself.** It reads `derived_state` with the step count, so it works on any
  env, with or without the state machine (Q-2, S-31).
- **DP-5 No branch pins.** Pins point at commits or tags, so a deleted branch can't break CI. (decision of
  2026-10-14)

Reused concepts:

| Need | Reused concept | Instead of |
|---|---|---|
| The agents' state for DLA | flatland-rl's `EnvAgent.derived_state(elapsed_steps)` (S-31) | a DLA-side state tracker, or requiring the state machine |
| Steer an unavoidable crossing | DLA's own action toward its set path (`_get_action`) | a second path planner |
| Pin flatland-rl | the three `# DEPENDENCY SWITCH` places | a new requirements file |

Not new on purpose: no new policy class, no new CLI option, no new episode format; the Dockerfiles and the simple
baselines stay as they are.

### 4.2 Verification

*How each requirement and quality goal is shown to work: tests, CI checks, before and after screenshots.*

Screenshots: none, nothing is visible. Checks run locally against flatland-rl at the pin (extracted tree on
`PYTHONPATH`) and in CI (`checks.yaml`).

| ID | Requirement | Test or check | Result | Screenshots |
|---|---|---|---|---|
| REQ-1 | Runs against each upstream step | `pytest flatland_baselines/deadlock_avoidance_heuristic/tests` per PR, CI per PR | ✅ CI on b9e4a9f, Python 3.11-3.14 (pandas 3). At ffd2477 CI failed `test_gen_trajectories_from_metadata` on pandas 3 (F-18, fixed in 4ef5ace) | - |
| REQ-2 | New action timing | `test_limitations_*.py`, `test_intermediate_stops.py` | ✅ | - |
| REQ-3 | No wrong branch at an unavoidable crossing | `test_stop_moving_advances_full_speed_agent_into_trunk_edge9_regression`, `test_blocked_full_speed_agent_is_steered_only_where_stop_moving_leaves_its_path`, `test_blocked_full_speed_agent_stops_before_a_symmetric_switch` | ✅ locally against 9eec95df (S-28, 70e8660): blocked full-speed agents stop with a free cell between them; steered only where `STOP_MOVING` would leave the path; before a symmetric switch they stop | - |
| REQ-4 | Entering prevention sees every entering agent | `test_entering_prevention.py` | ✅ locally against 9eec95df (S-29, 9e18779): earliest departures 0 and 1 in the first step, and a malfunction ending in the departure step, enter one after the other | - |
| REQ-5 | DLA sees the agents' correct state | grid runner test, offline demo and the DLA tests without state machine, online client unwrapped (S-31) | ✅ locally against 9eec95df (b9e4a9f): same expected values; online demo in CI | - |
| REQ-6 | Recalibrations named | commit messages of 17d2c76, 1d1b4fd, 048ea5f, 49635a3, 46e74cb, 01322b0, be92953, ffd2477 | ✅ | - |
| REQ-7 | Released flatland-rl before `main` | `pyproject.toml` without a git URL | ⏭️ moved to the follow-up in §1.4 (2026-11-12) | - |
| NFR-1 | Explainable results | each recalibration commit names its cause (REQ-6) | ✅ | - |
| NFR-2 | Reproducible CI | episodes by commit (S-33, f618d51); flatland-rl by release tag (follow-up, §1.4) | 💡 flatland-rl on the branch `main` since 21a3c5f (F-16, F-17) until the 5.0.0 release; episodes by commit ✅ | - |
| NFR-3 | Online equals offline | `online_demo/test_online.py` in CI | ✅ CI on b9e4a9f, Python 3.11-3.14, with the values of S-28 and no state machine (S-31) | - |

### 4.3 Phase 1: actions at cell entry (#80; REQ-1, REQ-2)

*One part of the change: what changes and why, with the requirements and decisions it implements.*

For flatland-rl#501.

- Renames: `*_configuration` → `*_entry_point`, `ObservationType` → `Observation`.
- `SetPathPolicy._act` and `_extract_agent_can_move` act from the pending next entry point toward the waypoint
  after it when a target is already settled ("look-ahead"), else from the current entry point.
- Episodes archive v6; online and offline golden values recalibrated.

### 4.4 Phase 2: TypeVar names (#81; REQ-1)

*One part of the change: what changes and why, with the requirements and decisions it implements.*

For flatland-rl#505: `Observation` → `ObservationT`. No behaviour change.

### 4.5 Phase 3: unavoidable crossings (#83; REQ-2, REQ-3)

*One part of the change: what changes and why, with the requirements and decisions it implements.*

For flatland-rl#509: `STOP_MOVING` no longer stops a crossing under way.

- A blocked `MOVING` agent whose distance plus speed reaches the end of its cell this step gets DLA's own directional action (`forced_action`), not
  `STOP_MOVING`, which continued straight at a switch. F-1 found the rule too wide; since S-28 (70e8660) only
  where `STOP_MOVING` would lead to another branch than the planned path (`_stop_moving_leaves_path`). Elsewhere
  DLA sends `STOP_MOVING`: along the path the agent brakes in the cell it crosses into; where `STOP_MOVING` is
  invalid (a symmetric switch facing the agent) flatland-rl refuses the crossing and the agent stops before the
  switch, for one `INVALID_ACTION` penalty.
- The two-switch limitation tests use half-speed agents to show the same limitation; a new full-speed test
  documents the full-speed case.
- Replay of the recorded episodes skipped (§1.4); episodes archive v7; golden values recalibrated.

### 4.6 Phase 4: entering after a malfunction (#84; REQ-4)

*One part of the change: what changes and why, with the requirements and decisions it implements.*

For flatland-rl#513.

- Entering prevention also considers an agent leaving a malfunction off the map, only when its malfunction clears
  this step.
- F-4: the condition was off by one against the engine; since S-29 (9e18779) entering prevention uses the engine's
  own rule (`earliest_departure <= _elapsed_steps + 1`, malfunction counter <= 1), whatever the agent's state.
- Golden values recalibrated for optimistic re-promotion.

### 4.7 Phase 5: entering on the first step (#85; REQ-4)

*One part of the change: what changes and why, with the requirements and decisions it implements.*

For flatland-rl#515.

- On the first step, entering prevention also considers a waiting agent with earliest departure 0.
- F-3: the condition was off by one against the engine, which lets earliest departure 0 and 1 enter in step 1;
  fixed by the same rule in S-29 (9e18779, §4.6).
- Golden values recalibrated for the one-step-earlier departure.

### 4.8 Phase 6: optional state machine (#89; REQ-5, REQ-6)

*One part of the change: what changes and why, with the requirements and decisions it implements.*

For flatland-rl#520: a plain env no longer updates the agents' state.

- Tests wrap the envs they build in `RailEnvStateMachineWrapper`; flatland-rl pinned by commit SHA.
- Offline golden values recalibrated.
- S-31 (b9e4a9f) removed the wrappers again: DLA reads `derived_state` with the step count (Q-2).

### 4.9 Phase 7: state machine on reload (#90; REQ-5, REQ-6)

*One part of the change: what changes and why, with the requirements and decisions it implements.*

For flatland-rl#541: the state machine setting is kept when an env is reloaded.

- Runners and the offline demo pass `--skip-state-machine-update False`; the online client wraps its local env.
- Python 3.10 dropped (#91).
- ffd2477: pin at #541's rebased head; the offline values change with flatland-rl#545 (no distance on a forced
  stop), located by testing `main` before #545 plus #541's diff.
- S-31 (b9e4a9f) removed the runner option and the online client's wrapper (Q-2).

### 4.10 Phase 8: review findings (REQ-3, REQ-4)

*One part of the change: what changes and why, with the requirements and decisions it implements.*

The open findings of §9.3, planned as S-27–S-36, S-38–S-42 and S-44–S-47 (§11.1), committed on top, on `-19` (Q-1). S-39–S-47 come from the reviews of ffd2477..5a75a4f; S-44 fixes behaviour that fails on `main` as well. S-43 is not used: its finding, F-25, went to the follow-up of F-9 (§1.4).

## 5. Building block view

*The components the change adds or changes: an overview, why the system is split this way, what each is responsible
for, the important interfaces, and the planned data model.*

### 5.1 Deadlock-avoidance baseline

*One white box: its building blocks, why it is split this way, and the interfaces between them.*

```mermaid
flowchart LR
    Runner["flatland-rl runners and evaluators"] --> DLA["DeadLockAvoidancePolicy"]
    DLA --> SPP["SetPathPolicy"]
    DLA --> Obs["FullEnvObservation"]
    Client["online demo client (run_solution.py)"] --> DLA
    Tests["DLA tests and offline demo"] --> DLA
    DLA --> Derived["EnvAgent.derived_state (flatland-rl)"]
    Pins["pins: pyproject.toml, environment.yml, checks.yaml"]
    style DLA fill:#fff3bf,stroke:#e67700,stroke-width:2px,stroke-dasharray:4 2,color:#000
    style SPP fill:#fff3bf,stroke:#e67700,stroke-width:2px,stroke-dasharray:4 2,color:#000
    style Obs fill:#fff3bf,stroke:#e67700,stroke-width:2px,stroke-dasharray:4 2,color:#000
    style Client fill:#fff3bf,stroke:#e67700,stroke-width:2px,stroke-dasharray:4 2,color:#000
    style Tests fill:#fff3bf,stroke:#e67700,stroke-width:2px,stroke-dasharray:4 2,color:#000
    style Pins fill:#fff3bf,stroke:#e67700,stroke-width:2px,stroke-dasharray:4 2,color:#000
    style Runner fill:#f1f3f5,stroke:#868e96,color:#000
    style Derived fill:#f1f3f5,stroke:#868e96,color:#000
```

Notation: [flowchart in Mermaid](https://mermaid.js.org/syntax/flowchart.html); yellow dashed: 🟨 modified, grey:
⬜ unchanged (flatland-rl). The state after phase 8: no state machine wrapper (S-31).

**Motivation:** the split stays as on `main`: `SetPathPolicy` owns the paths, DLA owns the conflict check. The
state machine stays in flatland-rl; DLA derives the agents' state itself (DP-4, Q-2, S-31).

**Contained building blocks:**

| Component | Responsibility | Change |
|---|---|---|
| `DeadLockAvoidancePolicy` | Lets an agent move only with enough free cells before an opposing agent; keeps agents from entering head-on. Not a path planner beyond its alternatives. | 🟨 modified: acts with the new timing, steers unavoidable crossings, sees more entering agents. |
| `SetPathPolicy` | Keeps one fixed path per agent and the action toward it. Not a conflict check. | 🟨 modified: acts toward the waypoint after a settled target. |
| `FullEnvObservation` | Gives DLA the whole env. Not a per-agent observation. | 🟨 modified: new type name only. |
| online demo client | Runs DLA against the remote evaluator. Not a place for DLA logic. | ⬜ unchanged: S-31 removed its state machine wrapper again. |
| DLA tests and offline demo | Fix DLA's behaviour and results. Not benchmark runs. | 🟨 modified: new expected values, episode replay skipped; no state machine (S-31). |
| pins | Name the flatland-rl and episodes used. Not a lock file. | 🟨 modified: commit SHA of flatland-rl, episodes v7; flatland-rl 5.0.0 in the follow-up (§1.4). |
| `EnvAgent.derived_state` | flatland-rl: the agents' state from their attributes and the step count. | ⬜ unchanged |

**Important interfaces:**

- DLA → flatland-rl: since S-31, DLA reads `EnvAgent.derived_state(elapsed_steps)` with `RailEnv._elapsed_steps`
  instead of `agent.state`.
- Runners → env: the runners need no state machine option for DLA (S-31).

## 6. Runtime view

*How the components work together at runtime: the flows and modes the change adds or changes, when it changes behaviour.*

### 6.1 One step with DLA (REQ-2–REQ-5)

*One flow or mode, as a sequence or state diagram.*

The order in which DLA decides and the engine applies.

```mermaid
sequenceDiagram
    participant Runner as flatland-rl runner
    participant DLA as DeadLockAvoidancePolicy
    participant Env as RailEnv
    Runner->>DLA: act_many with the env as observation
    DLA->>DLA: derive the agents' state with the step count
    DLA->>DLA: start step, update maps of paths and opposing agents
    DLA->>DLA: extract who can move, forced actions, entering agents
    DLA-->>Runner: actions (movement, STOP_MOVING or forced action)
    Runner->>Env: step with the actions
    Env->>Env: count the step and the malfunctions, then apply actions at cell entry
```

Notation: [sequence diagram in Mermaid](https://mermaid.js.org/syntax/sequenceDiagram.html).

The flow since S-31 (no state machine). DLA decides before the engine counts the step: every condition DLA mirrors
from the engine (earliest departure, malfunction end) has to look one step ahead (F-3, F-4).

## 7. Deployment view

*Where the components run and how they are delivered, when the change touches it.*

- CI (`checks.yaml`): Python 3.11-3.14 for the DLA tests and the online/offline demo; flatland-rl installed from
  `env.flatland-rl-ref`; episodes from `env.flatland-benchmarks-episodes-url` (a flatland-scenarios branch, S-33).
- Release (`release-please.yml`): every push to `main` updates the release PR; merging it publishes to PyPI and
  needs the released flatland-rl pin (follow-up, §1.4).
- Docker images (`docker.yml`, Dockerfiles): unchanged, built on `main` from flatland-rl's `latest-py3.12` image.

## 8. Crosscutting concepts

*What the change introduces that other code will follow, and what it changes for existing callers.*

### 8.1 New patterns

*What is new compared with the base branch.*

Compared with `main` c1733cb. What is reused, and what is not new on purpose, is in §4.1.

| ID | New | Defined in |
|---|---|---|
| PAT-1 | Look-ahead: act from the settled next entry point toward the waypoint after it | §4.3 |
| PAT-2 | Forced action for an unavoidable crossing | §4.5 |
| PAT-3 | Entering prevention mirrors the engine's map-entry rule (earliest departure, malfunction counter), not its states | §4.6, §4.7 |
| PAT-4 | DLA derives the agents' state (`derived_state` with the step count) instead of requiring the state machine | §4.8, Q-2 |
| PAT-5 | flatland-rl pinned by commit SHA with the branch name as a comment | §4.8 |

### 8.2 Breaking changes

*What breaks for whom, and the migration path.*

Breaking for users of the baselines package, each with its migration:

- **flatland-rl 5.0 required.** The baselines no longer run with flatland-rl 4.x (renamed symbols, new timing).
  Migration: install flatland-rl 5.0.0 (the follow-up in §1.4 pins it).
- **Python 3.10 dropped** (34a3bce, #91). Migration: Python 3.11 or newer.
- **Benchmark results change.** DLA's rewards and success rates differ from 4.3.0 (golden values per phase).
  Compatibility is not kept on purpose: the engine changed.

## 9. Decisions

*The decisions behind the plan: the questions still open, the decisions taken, and how each review finding was settled.*

### 9.1 Open questions

*Choices not made yet, each with its options and a proposal.*

✅ ~~**(Q-1) Where are the review findings fixed?**~~ **Decided 2026-11-12:** a) on top, on `-19` (#90).
- a) On top, on `-19` (#90): no rebase of the chain; the fix lands one PR after the bug.
- b) In the PR that introduced each bug (F-1 in #83, F-4 in #84, F-3 in #85): clean history per PR, but every
  branch above must be rebased and force-pushed, and their golden values may move again.
- c) A new PR on top of #90.

✅ ~~**(Q-2) How does DLA get the agents' state without every caller turning on the state machine (F-14)?**~~
**Decided 2026-11-12:** c) derived state with the step count, after the check below. A redesign of DLA without train
state is folded into S-34 and S-35 (2026-11-13), as preparation for
[#65](https://github.com/flatland-association/flatland-baselines/pull/65).
- a) Keep it with the caller (now): tests, runners and the online client must remember it.
- b) DLA wraps the env it observes in `_init_env` (`RailEnvStateMachineWrapper(env, skip_state_machine_update=False)`,
  in place and idempotent). DLA changes the env it observes, which the wrapper says affects no rewards or
  observations; the state machine's cost stays.
- c) DLA reads `agent.derived_state(self.rail_env._elapsed_steps)` instead of `agent.state`. flatland-rl documents it
  as equal to the tracked state when read between steps, which is where DLA reads it. Plain `agent.state` on an
  unwrapped env passes no step count and never reports `READY_TO_DEPART`, which is why DLA did worse unwrapped.
  Checked 2026-11-12 at ffd2477 against flatland-rl b99bfd11 in a scratch copy: all ten reads replaced, no wrapper,
  runners with state machine updates skipped; the DLA tests and the offline demo pass with today's golden values
  (20 passed, 50 skipped, 1 xfailed).

✅ ~~**(Q-3) How is the stack merged?**~~ **Decided 2026-11-12:** a), with GitHub's stack merge: merging #90 merges
#80-#90 into `main`, bottom up, in one operation
([Merging stacked pull requests](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/merging-stacked-pull-requests)).
Each PR of the stack is checked against `main`'s rules, and the history must be linear (it is: no merge commits in
`main..-19`). Prerequisites in §11.2. Since 2026-11-12 the pin to flatland-rl 5.0.0 (formerly S-37) follows the merge
(§1.4): `main` pins flatland-rl#541's commit until then, and the release PR waits for it.
- a) Top down: the whole stack lands together, after S-37 pinned flatland-rl 5.0.0.
- b) Bottom up: each PR into `main` in turn. Each merge only updates release-please's release PR (no release), but
  `main` then pins the flatland-rl branches of #80-#85, which no longer exist: CI on `main` fails until the last PR
  is in, and the release PR must not be merged in between.

### 9.2 Decision log

*Every decision by date, with why it was taken and what follows from it.*

| Date | Decision | Why | Consequences |
|---|---|---|---|
| 2026-08-18 | ✅ One baselines PR per upstream flatland-rl PR, pinned to its branch. | Small reviews; breakage shows at the upstream step that causes it. | Seven PRs; pins move often (C-1). |
| 2026-08-25 | ✅ Recalibrate golden values with the upstream change named. | Results stay explainable (NFR-1). | Each phase has a recalibration commit. |
| 2026-09-01 | ✅ Show the two-switch limitation with half-speed agents. | At full speed the agent can't be stopped before the next cell any more; wider rails and `min_free_cell` did not help. | The full-speed case gets its own test. |
| 2026-09-01 | ✅ Skip the recorded-episode replay until the episodes are recorded again. | DLA reacts to the new timing; 41 of 50 episodes differ. | No replay coverage (§2.1, §1.4). |
| 2026-09-02 | ✅ Send DLA's own action for an unavoidable crossing instead of `STOP_MOVING`. | `STOP_MOVING` continued straight at a switch and crashed the demo. | Too wide: also kept blocked full-speed agents moving (F-1); narrowed in S-28 (70e8660) to where `STOP_MOVING` would leave the planned path. |
| 2026-10-14 | ✅ Pin flatland-rl by commit SHA, branch name as a comment. | Upstream branches are deleted after merging (C-4). | CI stays reproducible; episodes still pinned by branch (S-33). |
| ~~2026-10-19~~ | ~~The caller turns on the state machine for DLA.~~ Replaced on 2026-11-12 (Q-2). | flatland-rl#520 made it optional; the runner option came with #541. | Breaking for DLA users (§8.2); Q-2. |
| 2026-10-19 | ✅ Drop Python 3.10, test 3.14 (#91). | flatland-rl 5.0 requires 3.11 (C-5). | Breaking (§8.2). |
| 2026-11-12 | ✅ Fix the review findings on top, on `-19` (#90) (Q-1). | No rebase and force-push of the chain; the stack is merged as a whole. | A fix lands one PR above its bug; S-28–S-36 unblocked. |
| 2026-11-12 | ✅ DLA reads the derived state with the step count, not `agent.state`; a redesign without train state follows later (Q-2). | No caller has to turn on the state machine, and its cost goes away; checked to keep every golden value. Wrapping in `_init_env` (b) was the first choice "for now", but keeps the state machine. | S-31; the breaking change "DLA needs the state machine" goes away (§8.2); follow-up in §1.4 (#65). |
| 2026-11-12 | ✅ Merge the stack top down with GitHub's stack merge (Q-3). | One operation; `main` never pins a deleted flatland-rl branch; the release PR is merged only after. | `main`'s rules must accept every PR of the stack (F-18, F-19, §11.2). |
| 2026-11-12 | ✅ Pin the released flatland-rl 5.0.0 after the merge, in a follow-up (formerly S-37). | The stack need not wait for flatland-rl#541's merge and the 5.0.0 release. | REQ-7 and F-2, F-16, F-17 leave the PR (§1.4); `main` pins flatland-rl#541's commit until the follow-up, so release-please's release PR waits for it, and the commit must stay fetchable (§2.2). |
| 2026-11-12 | ✅ Leave F-5 (record the episodes again) and F-9 (apply the alternative path) out of this PR, tracked in [#94](https://github.com/flatland-association/flatland-baselines/issues/94) and [#93](https://github.com/flatland-association/flatland-baselines/issues/93). | F-9 fails on `main` too and only matters with rerouting switched on (off by default); both change DLA's results and need their own review; F-5 needs a new archive in flatland-scenarios. Fixing F-9 here as a step of phase 8 was the alternative. Both had been put into the follow-ups when the plan was written, without asking. | Two follow-ups in §1.4 with issues; no steps for them. |
| 2026-11-12 | ✅ Leave the follow-up "Pin the released flatland-rl 5.0.0" untracked; re-pin flatland-rl to #541's merge commit once #541 is merged (S-38). | #541 is about to be merged; pinning its merge commit on flatland-rl's `main` keeps `main` fetchable whatever merge method #541 gets. | S-38 before the stack merge (§11.2); the follow-up has no issue. |
| 2026-11-13 | ✅ Keep S-29 on the pinned engine's departure rule; track [flatland-rl#559](https://github.com/flatland-association/flatland-rl/issues/559) (earliest departure no longer enforced by the env) in a follow-up. | #559 is open and undecided; S-29 matches the flatland-rl the stack pins. Not taken: let DLA wait for the earliest departure itself now (ready for #559, but DLA's actions for agents not yet due change), or hold S-29 until #559 is decided (F-3, F-4 stay open). | Follow-up (§1.4): once #559 lands, DLA waits with the same predicate, and entering prevention stays as it is. |
| 2026-11-13 | ✅ Fold S-34 into a reconciliation of `SetPathPolicy` with flatland-rl's `ShortestPathPolicy` after [flatland-rl#528](https://github.com/flatland-association/flatland-rl/pull/528), in this PR; keep S-30 to the fix of F-8. | `ShortestPathPolicy` is a near-copy of `SetPathPolicy` (same look-ahead and path trimming), used on its own in flatland-rl; #528 rewrites it with a pluggable router, intermediate stops, a route index and routing per episode. One implementation upstream instead of two that drift (F-7, F-8). Not taken: a local shared look-ahead (#528 would replace it), or a follow-up in another PR. | S-34 waits for #528's merge and a re-pin; S-35 still needs S-34. |
| 2026-11-13 | ✅ Pin flatland-rl's `main` branch instead of #541's merge commit, as F-16 and F-17 suggested, now that flatland-rl#541 is merged. | The pin follows flatland-rl's development until the 5.0.0 release; replaces S-38's commit pin. | New flatland-rl commits can change CI results without a change here; the follow-up (§1.4) still pins the released 5.0.0 before release-please's release PR. |
| 2026-11-13 | ✅ Pin the episodes archive to v7 at 2fc33319 ([flatland-scenarios#34](https://github.com/flatland-association/flatland-scenarios/pull/34) not merged), with a TODO; flatland-rl's URL in a follow-up, tracked by [flatland-scenarios#37](https://github.com/flatland-association/flatland-scenarios/issues/37). | The offline demo only reads `metadata.csv`, the same in v5 and v7, so both are green; v7 keeps the archive in step with flatland-rl's CI and the convention that the archive version follows the flatland-rl pin. The commit stays reachable through PR #34's ref after the branch is deleted. Not taken: v5 at 2e20db2f on flatland-scenarios' `main` (merged, but an older archive than flatland-rl uses). | Move to a commit on flatland-scenarios' `main` once #34 is merged (TODO in `checks.yaml`); #94 records a new archive anyway. |
| 2026-11-13 | ✅ Fold the follow-up "Redesign DLA to need no train state" into S-34 (`SetPathPolicy`) and S-35 (DLA), in this PR (Q-2 revisited for its remainder). | The eight conditions on `_derived_state` are the code S-34 and S-35 rewrite anyway; deciding from position, done, malfunction and earliest departure makes DLA independent of what flatland-rl's states mean (flatland-rl#559) and fits the stateless observation of flatland-rl#528 and #65. Not taken: a flatland-baselines issue for later, or leaving it untracked. | S-35 gets bigger and still waits for S-34 (flatland-rl#528); the follow-up row is gone. |
| 2026-11-13 | ✅ Do S-35 before S-34. | S-34 waits for flatland-rl#528; S-35 needs nothing from it. | S-34 later works in the split code; DLA's look-ahead now sits in `_lookahead` for S-34 to share. |

### 9.3 Code Review Findings

*Findings from code reviews, PR comments and TODOs, with their status and the commits that fix them.*

The stack #80-#90 (`main` c1733cb .. `178-agents-living-on-the-edge-19`). Code review findings (F-1–F-13) were added
on 2026-11-12, at ffd2477; their line numbers refer to it. PR review comments on #90 (F-14–F-17) were added on
2026-11-12; #80-#89 have no review comments. F-18 (CI) and F-19 (`main`'s ruleset) were added on 2026-11-12, at
ffd2477. TODOs added in the stack (F-20) were added on 2026-11-13, at 536f1f5: the three `TODO update to released version` of the pin refer to no issue; the other TODOs are tracked elsewhere (the skipped episode replay by flatland-scenarios#37, since 8f9b161 first by #94; the episodes pin by flatland-scenarios#34). Code review findings (F-21–F-29) were added on 2026-11-13, at 5a75a4f, from reviews of ffd2477..5a75a4f (F-24–F-29 from `/code-review`); their line numbers refer to it. In Fixed: ✅ fixed,
⏭️ open and taken over by a follow-up (§1.4), ❌ next to ✅: fixed in the code, but the PR comment is not resolved on
GitHub yet. "resolved" in Source marks threads resolved on GitHub.

| Fixed | ID | Source | Category | Severity | Finding | Commit message |
|---|---|---|---|---|---|---|
| ✅ | F-1 | code review | bug | high | `flatland_baselines/deadlock_avoidance_heuristic/policy/deadlock_avoidance_policy.py:473,493-516,176`, `tests/test_limitations_two_switches.py:210`: a full-speed agent reaches the end of its cell every step (`is_cell_exit()`), so a blocked one gets `forced_action` instead of `STOP_MOVING` every step, and with a settled target the forced action commits it to the cell after. It drives on until the engine blocks it, head-on next to the other train. Shown in the full-speed two-switch scenario: agent 0 ends at (1,3) next to agent 1; with the forced action off, both stop in step 2 with a free cell between them, and nothing derails. The test asserts the head-on outcome. Fixed in 70e8660 (S-28): the forced action only where `STOP_MOVING` would lead to another branch than the planned path; both agents now stop with a free cell between them. | fix(dla): send a directional action for an unavoidable crossing only where STOP_MOVING would leave the planned path. |
| ✅ | F-3 | code review | bug | medium | `deadlock_avoidance_policy.py:543-546`: the first-step shortcut for waiting agents matches only `earliest_departure == 0`; the engine lets 0 and 1 enter in step 1 (`rail_env_state_machine_wrapper.py:100`). Two facing agents with earliest departure 0 and 1 enter head-on unchecked. Fixed in 9e18779 (S-29): an off-map agent is entering if `earliest_departure <= _elapsed_steps + 1` and its malfunction counter is <= 1. | fix(dla): mirror the engine's departure rules in entering prevention. |
| ✅ | F-4 | code review | bug | medium | `deadlock_avoidance_policy.py:540-542`: the malfunction-off-map check compares the earliest departure with the step count before `step()` increments it; the engine compares after (`rail_env_state_machine_wrapper.py:113`). An agent whose malfunction ends in its departure step enters unchecked. Fixed in 9e18779 (S-29), with the engine's own rule. | fix(dla): mirror the engine's departure rules in entering prevention. |
| ✅ | F-6 | code review | ci | medium | `.github/workflows/checks.yaml:7`: the episodes archive comes from the flatland-scenarios branch `178-agents-living-on-the-edge-9`, which can change or be deleted. Fixed in f618d51 (S-33): pinned to 2fc33319, the commit that last changed the v7 archive, with a TODO to move to flatland-scenarios' `main` once flatland-scenarios#34 is merged; flatland-rl's URL is a follow-up (§1.4). | ci: pin the benchmark episodes archive to a flatland-scenarios commit. |
| ✅ | F-8 | code review | bug | low | `policy/set_path_policy.py:69`: `SetPathPolicy._act` calls `check_action_on_agent`, private (`_check_action_on_agent`, five values) since flatland-rl 4.3.0; standalone use raises `AttributeError`, and before that already on `self.audit`, which only DLA set. On `main` it fails as well. Fixed in 68740b5 (S-30): `_check_action_on_agent`, and `self.audit` initialised in `SetPathPolicy`; a test runs `SetPathPolicy` on its own. | fix(policies): use flatland-rl's _check_action_on_agent in SetPathPolicy._act and test it. |
| ✅ | F-10 | code review | test-robustness | low | `tests/test_limitations_two_switches.py:446,455`: `num_steps=21` only lands the oscillating agents 2 and 3 on STOPPED; a one-step timing change breaks the test without a regression. Fixed in 68a3bc1 (S-32): actions and positions checked over six steps, the state only for agents 0 and 1. | test(dla): assert queued agents' positions, not their oscillating state. |
| ✅ | F-11 | code review | design | low | `deadlock_avoidance_policy.py:446-571,537,186-267`: `_extract_agent_can_move` has 126 lines and redefines `_about_to_leave_off_map` every step (named for leaving, used for entering); `_find_alternative` repeats six verbose/audit blocks. Fixed in 8e5760c (S-35): `_extract_agent_can_move` split into `_update_agent_can_move`, `_lookahead`, `_prevent_entering_conflicts` and `_plot_debug`; the entering helper is the method `_enters_map_in_coming_step`; verbose output and audit entries go through `_log`/`_audit`; DLA decides without train state. | refactor(dla): split _extract_agent_can_move, decide without train state and log through one audit helper. |
| ✅ | F-12 | code review | docs | low | `CLAUDE.md:86,163-165`: the pin is described as a tag or branch name (it is a commit SHA); "the one and only value" for `<= 1`; the first-step rule repeats F-3. Fixed in 9e18779 (S-29): lines 163-165 describe the engine's rule; and in 21a3c5f: line 86 ("a tag or branch name") is right again, as the pin is the branch `main`. | docs: describe the flatland-rl pin and entering prevention as the code has them. |
| ✅ | F-13 | code review | style | low | `deadlock_avoidance_policy.py:452-453,537`, `online_demo/submission/run_solution.py:40`: per-step state not declared with types; no `EnvAgent` hint; the online client drops the wrapper's return value. Fixed in a2ba274 (S-36): `forced_action`, `switches` and `agent_positions` declared with types in `__init__`, `EnvAgent` hint on `_about_to_leave_off_map`; the wrapper is gone since b9e4a9f (S-31), so its return value is moot. | style(dla): declare per-step state with type hints. |
| ✅ | F-14 | [PR comment](https://github.com/flatland-association/flatland-baselines/pull/90#discussion_r4311347990), resolved | design | medium | `tests/test_policy_grid_runner_evaluator.py:44`: "update dla instead?": DLA could get the agents' state itself instead of every caller passing the option. Fixed in b9e4a9f (S-31): DLA reads the derived state, and the option is gone from every caller (Q-2); deciding without train state at all is folded into S-34 and S-35. | refactor(dla): read the agents' derived state instead of requiring the state machine. |
| ✅ | F-15 | [PR comment](https://github.com/flatland-association/flatland-baselines/pull/90#discussion_r4312300888), resolved | ci | low | `.github/workflows/checks.yaml:1`: "Depends on #91" (dropping Python 3.10). Fixed in 34a3bce (S-24): #91 is merged into this branch. | chore: drop Python 3.10, test 3.14. (#91) |
| ✅ | F-16 | [PR comment](https://github.com/flatland-association/flatland-baselines/pull/90#discussion_r4326418695), resolved | build | medium | `environment.yml:8` (outdated): suggests pinning flatland-rl `@main`; Fixed in 21a3c5f: pinned to `@main` as suggested, now that flatland-rl#541 is merged; the released 5.0.0 stays a follow-up (§1.4). | build: pin flatland-rl to main now that flatland-rl#541 is merged. |
| ✅ | F-17 | [PR comment](https://github.com/flatland-association/flatland-baselines/pull/90#discussion_r4326431817), resolved | build | medium | `.github/workflows/checks.yaml:6` (outdated): "TODO update to released version", answered "depends on flatland-rl#541". Fixed in 21a3c5f: `flatland-rl-ref: main` with the TODO, as suggested, now that #541 is merged; the released 5.0.0 stays a follow-up (§1.4). | build: pin flatland-rl to main now that flatland-rl#541 is merged. |
| ✅ | F-18 | CI | bug | high | `tests/test_policy_grid_runner_evaluator.py:59`, flatland-rl `flatland/evaluators/trajectory_evaluator.py:135` (at b99bfd11): CI on ffd2477 fails `test_gen_trajectories_from_metadata` on Python 3.11-3.14 with numpy's `DTypePromotionError`: under pandas 3.0.6 an expected reward comes back as a string. Cause: `Trajectory._read_trains_rewards_dones_infos` converted tuple rewards (`PunctualityRewards`) only for an `object` column, and pandas 3 reads text as `str`; no flatland-rl test reloads tuple rewards. The re-pin brought in flatland-rl#540 (pandas 3); be92953 passed with pandas 2.3.3. Local checks missed it: the local env has pandas 2.3.3. Fixed in 4ef5ace (S-27): the cause is fixed in flatland-rl c689b22f on #541's branch (30e7a35f before #541's rebase), and the pin moved there. | fix: pin flatland-rl to #541's head with the pandas 3 fix for tuple rewards. |
| ✅ | F-19 | ruleset | ci | high | `.github/workflows/checks.yaml:27`, `main`'s ruleset (id 9165551): it required `test-dla-regression (3.10)`, which #90 no longer runs, and not `(3.14)`, so the stack merge was blocked. Fixed on 2026-11-12 by chenkins in the repository settings: the ruleset requires `(3.11)` to `(3.14)` (strict); no commit and no step. | - |
| ⏭️ | F-2 | code review | build | high | `pyproject.toml:29`, `.github/workflows/release-please.yml:26-48`: the dependency is a direct git URL; merging release-please's release PR publishes to PyPI, and PyPI rejects direct URLs. While `main` carries the git pin, the next release fails. Open: taken over by the follow-up "Pin the released flatland-rl 5.0.0" (§1.4). | build: pin flatland-rl to the released 5.0.0 before merging to main. |
| ⏭️ | F-5 | code review | test-coverage | medium | `tests/test_episodes_deadlock_avoidance.py:15-25`: the 50 recorded-episode replays are skipped unconditionally, while CI still downloads the episodes (`checks.yaml:52,106`); the TODO's bare `#178` meant flatland-rl#178; since 8f9b161 it refers to #94. Open: taken over by the follow-up "Record the 50 benchmark episodes again" (§1.4, [flatland-scenarios#37](https://github.com/flatland-association/flatland-scenarios/issues/37); [#94](https://github.com/flatland-association/flatland-baselines/issues/94) closed as its duplicate). | test(dla): regenerate the recorded episodes against the pinned flatland-rl and replay them again. |
|  | F-7 | code review | design | medium | `policy/set_path_policy.py:60-67`, `deadlock_avoidance_policy.py:493-511`: the look-ahead is copied in two places and has already drifted (F-8); a third copy is flatland-rl's `ShortestPathPolicy` (`rail_env_policies.py`), which [flatland-rl#528](https://github.com/flatland-association/flatland-rl/pull/528) rewrites with a pluggable router. Open: planned in S-34, which bases `SetPathPolicy` on that `ShortestPathPolicy` once #528 is merged. | refactor(dla): share the cell-entry look-ahead between SetPathPolicy and DeadLockAvoidancePolicy. |
| ⏭️ | F-9 | code review | bug | low | `deadlock_avoidance_policy.py:232-239,219`: `_find_alternative` picks an alternative but never assigns it, so rerouting never takes effect; on `main` it is the same. Once applied, it must start from a settled next entry point. Open: taken over by the follow-up "Apply the alternative path" (§1.4, [#93](https://github.com/flatland-association/flatland-baselines/issues/93)). | fix(dla): apply the chosen alternative path, planned from the settled next entry point. |
| ⏭️ | F-20 | TODO | build | medium | `.github/workflows/checks.yaml:5`, `environment.yml:7`, `pyproject.toml:28` (at 536f1f5): `# TODO update to released version` above the flatland-rl pin on `main` (21a3c5f, as F-17 suggested); refers to no issue. Open: taken over by the follow-up "Pin the released flatland-rl 5.0.0" (§1.4), which removes the TODOs. | build: pin flatland-rl to the released 5.0.0 before merging to main. |
|  | F-21 | code review | test-robustness | low | `tests/test_entering_prevention.py:249,271`: the two new entering tests call `min(entered_at[0], entered_at[1])` before checking that both agents entered; if one never enters, `min` raises a `TypeError` (`None` against an int) instead of the assertion message. Fix: assert both entered first (the shared helper's first check), then the step of the first entry. Open: planned in S-39. | test(dla): check that both agents entered before comparing their entry steps. |
|  | F-22 | code review | naming | low | `policy/deadlock_avoidance_policy.py:163,222,323,526`: `_is_on_map` is also false for an agent parked at its target (`remove_agents_at_target=False`), which is on the map; the docstring says so, the name doesn't. Fix: a name for what it means, e.g. `_is_active_on_map` (on the map and not done). Open: planned in S-40. | style(dla): name the on-map helper for agents that are on the map and not done. |
|  | F-23 | code review | type-hints | low | `policy/deadlock_avoidance_policy.py:104,538`: `agent_can_move` is declared `Dict[AgentHandle, Tuple[int, int, int, RailEnvActions]]`, but `_update_agent_can_move` stores a list (older than this range; S-35 moved the line). Fix: store a tuple. Open: planned in S-41. | style(dla): store agent_can_move entries as the tuples they are declared as. |
|  | F-24 | code review | bug | medium | `policy/set_path_policy.py:54,102`: `SetPathPolicy` on its own raises `KeyError` once an agent is done and stays parked at its target (`remove_agents_at_target=False`): `_update_agent` drops the set path when `derived_state` is DONE, but `_act` still reads `self._set_paths[handle]`, as `current_entry_point` stays set. Before S-31, `agent.state` never turned DONE on a plain env, so the path was never dropped. Reproduced at 5a75a4f with `test_set_path_policy`'s env and `remove_agents_at_target=False`: `KeyError: 0` in the step after arrival. DLA overrides `_act` and is not affected. Fix: `_act` returns `DO_NOTHING` for an agent without a set path; a test keeps the agent at its target. Open: planned in S-42. | fix(policies): let SetPathPolicy act for an agent parked at its target. |
| ⏭️ | F-25 | code review | bug | medium | `policy/deadlock_avoidance_policy.py:239-292`: `_find_alternative` picks an alternative path but never assigns it to `self._set_paths[handle]`, since 796aeb2 (2026-01-30, on `main` too): the line `self._set_paths[...] = self._shortest_path_from_non_flexible_waypoints(alternative, ...)` was dropped there. So `use_alternative_at_first_intermediate_and_then_always_first_strategy` and `drop_next_threshold` (`DeadlockAvoidanceHeuristics`) never reroute a blocked agent; the log always says "not changed". Once the path is assigned again, `forced_action`, worked out in `_start_step` from the old path, can steer towards the old branch: recompute it after a change. On `main` the policy failed as well. Same concern as F-9. Open: taken over by the follow-up "Apply the alternative path" (§1.4, [#93](https://github.com/flatland-association/flatland-baselines/issues/93)), decided on 2026-11-13. | fix(dla): apply the alternative path that _find_alternative picks for a blocked agent. |
|  | F-26 | code review | robustness | low | `policy/set_path_policy.py:112,120,187`: `SetPathPolicy`'s audit entries read `self.rail_env._elapsed_steps`, which only `DeadLockAvoidancePolicy` has; with `SetPathPolicy().audit = []` (the attribute S-30 initialised) the first `_update_agent` raises `AttributeError`. They also copy the verbose-print-plus-audit pattern that S-35 folded into DLA's `_log`/`_audit`. Fix: move `_log`/`_audit` into `SetPathPolicy`, with the step count from the env passed in. Open: planned in S-45. | refactor(policies): move the audit and log helpers into SetPathPolicy. |
|  | F-27 | code review | design | low | `policy/deadlock_avoidance_policy.py:172,349`: in the first step (`_elapsed_steps == 0`), an agent with earliest departure 0 or 1 enters in the coming step (`_enters_map_in_coming_step`), but `_is_not_ready_to_depart` still holds for it, so step (2.3) skips it: no other agent sees it as opposing. Two such agents heading towards each other on one track both get leave to enter; only the opt-in entering prevention catches it. Same as before S-35 (READY_TO_DEPART needs `elapsed_steps >= 1`, flatland-rl#280); on `main` the policy failed as well. Fix: take agents that enter in the coming step into the opposing-agent map. Open: planned in S-44. | fix(dla): see agents that enter the map in the first step as opposing agents. |
|  | F-28 | code review | test-coverage | low | `tests/test_set_path_policy.py:77`: the agent has no real choice on this rail (the turn at switch A is forced from its arm), so `SetPathPolicy` sends `MOVE_FORWARD` throughout (checked at 5a75a4f), and `set(turns) <= {...}` also passes when `_act` always returns `MOVE_FORWARD`. Fix: a path through a switch arm with a choice, asserting the exact turn action. Open: planned in S-46. | test(policies): let the standalone SetPathPolicy test take a real branch at a switch. |
|  | F-29 | code review | docs | low | `CLAUDE.md:53`: says the episode replay suite is skipped "(#178)", while the test's skip reason and TODO name flatland-scenarios#37 since 8f9b161. Open: planned in S-47. | docs: name flatland-scenarios#37 as the tracker of the skipped episode replay in CLAUDE.md. |

Issue references: the stack closes no issue. The code refers to flatland-rl#178 (closed) in comments of
`deadlock_avoidance_policy.py`, as context for the crossing timing, not as open work. The skip's TODO in
`tests/test_episodes_deadlock_avoidance.py:15` refers to [flatland-scenarios#37](https://github.com/flatland-association/flatland-scenarios/issues/37) (F-5's follow-up; it referred to [#94](https://github.com/flatland-association/flatland-baselines/issues/94) from 8f9b161 until #94 was closed as a duplicate of #37), until that work removes it.

## 10. Glossary

*The domain and technical words the plan uses, with their code symbols.*

- **Entry point** (`agent.current_entry_point`, formerly `current_configuration`): the cell and direction an agent
  entered by. Not its position within the cell.
- **Settled target, pending target** (`agent.next_entry_point` differing from `current_entry_point`): the next cell
  entry, already decided one cell earlier. Not changeable by this step's action.
- **Look-ahead**: the action computed from the settled target toward the waypoint after it (PAT-1, §4.3).
- **Unavoidable crossing** (`MOVING` and `speed_counter.is_cell_exit()`: distance plus speed reaches the end of the
  cell this step): a crossing the engine completes with any action valid there (flatland-rl#509). Not a position: an
  agent never sits on the exit of its cell; a refused crossing (an action invalid there, or the next cell taken)
  stops it with speed 0 and keeps its distance (flatland-rl#545). Described in the flatland-book's 5.0.0 draft,
  [Speed Profiles](https://github.com/flatland-association/flatland-book/blob/0dd2d374df5e02d67aac6ec8dd11b3eb14e07931/environment/environment/speed_profiles.ipynb) ([flatland-book#42](https://github.com/flatland-association/flatland-book/pull/42) at 0dd2d37), which may still change until 5.0.0; at 0dd2d37 it still describes
  the refused crossing as before #545 (distance 1).
- **Forced action** (`DeadLockAvoidancePolicy.forced_action`): DLA's action toward its path, sent to a blocked agent
  at an unavoidable crossing instead of `STOP_MOVING`, only where `STOP_MOVING` would lead to another branch than
  the planned path (`_stop_moving_leaves_path`).
- **Entering prevention** (`use_entering_prevention`): DLA's check that no two agents enter onto conflicting paths
  in the same step. Not a reservation of the track.
- **Derived state** (`EnvAgent.derived_state(elapsed_steps)`): the agent's `TrainState` computed from its position,
  speed, malfunction and the step count. Equal to the state machine's state between steps. Not plain `agent.state`
  on an unwrapped env, which passes no step count and never reports `READY_TO_DEPART`.
- **Golden values**: the expected results in the tests (rewards, success rates, steps). Not benchmark scores.
- **State machine wrapper** (`RailEnvStateMachineWrapper`): flatland-rl's in-place patch that updates `agent.state`
  on each step. Unwrapped, `agent.state` is the derived state (`derived_state()`).
- **Stack merge**: GitHub's merge of a stacked PR together with every PR below it into the stack's base, bottom up
  (Q-3). Not a merge queue.
- **Pin** (`# DEPENDENCY SWITCH`): the flatland-rl version or commit in `pyproject.toml`, `environment.yml` and
  `checks.yaml`.

## 11. Execution plan and status

*The commits that build the change, in order, and which of them are done: the source of the PR's status.*

### 11.1 Steps

*One row per commit: what it does, the findings it fixes, its message, and its SHAs once committed.*

#### Phase 1: actions at cell entry (#80) ✅

| ID | Findings | Step | Category | Severity | Commit message | Commit(s) | Needs | Blocked by |
|---|---|---|---|---|---|---|---|---|
| ✅ S-1 | - | Follow the renames and the cell-entry timing in DLA and `SetPathPolicy`. | compatibility | medium | fix: track flatland-rl's Configuration/ObservationType renames and the cell-entry action timing change | 14eb698 | flatland-rl#501 | - |
| ✅ S-2 | - | Pin flatland-rl to `-3`. | build | low | chore: deps switch flatland-rl to in-flight branch 178-agents-living-on-the-edge-3 | 8d3eb33 | S-1 | - |
| ✅ S-3 | - | Episodes archive v6 in CI and docs. | ci | low | ci: point benchmark episodes download at v6 archive on the in-flight branch | 30e2849, ab6d420 | S-2 | - |
| ✅ S-4 | - | Recalibrate the online/offline values. | test | medium | Re-calibrate DLA online/offline reward assertions for actions-applied-at-cell-entry | 17d2c76 | S-3 | - |

#### Phase 2: TypeVar names (#81) ✅

| ID | Findings | Step | Category | Severity | Commit message | Commit(s) | Needs | Blocked by |
|---|---|---|---|---|---|---|---|---|
| ✅ S-5 | - | `Observation` → `ObservationT`. | compatibility | medium | fix: track flatland-rl's ObservationBuilder TypeVar rename (Observation -> ObservationT) | 283e8ce | flatland-rl#505 | - |
| ✅ S-6 | - | Pin flatland-rl to `-5`. | build | low | chore: deps switch flatland-rl to in-flight branch 178-agents-living-on-the-edge-5 | 9128281 | S-5 | - |

#### Phase 3: unavoidable crossings (#83) ✅

| ID | Findings | Step | Category | Severity | Commit message | Commit(s) | Needs | Blocked by |
|---|---|---|---|---|---|---|---|---|
| ✅ S-7 | - | Pin flatland-rl to `-9`, episodes v7. | build | low | chore: deps switch flatland-rl to in-flight branch 178-agents-living-on-the-edge-9 | 4ae49e9 | flatland-rl#509 | - |
| ✅ S-8 | - | Recalibrate the grid runner values. | test | medium | fix: re-calibrate DLA success-rate/reward golden values for flatland-rl@edge-9 | 1d1b4fd | S-7 | - |
| ✅ S-9 | - | Two-switch limitation tests with half-speed agents. | test | medium | test: restore two-switch limitation assertions via half-speed agents (#178) | c9b4e05 | S-7 | - |
| ✅ S-10 | - | Skip the recorded-episode replay. | test | medium | test: skip recorded-episode replay pending regeneration against edge-9 (#178) | cc6b67a | S-7 | - |
| ✅ S-11 | - | Forced action for an unavoidable crossing. | fix | high | Fix — always compute DLA's intended directional action, use it instead of blind STOP_MOVING when the crossing is unavoidable anyway | 73ecb5f | S-7 | - |
| ✅ S-12 | - | `CLAUDE.md`: pins, layout, crossing timing. | docs | low | docs: update CLAUDE.md accuracy and add DLA/flatland-rl#178 interaction note | dba35bb | S-11 | - |
| ✅ S-13 | - | Recalibrate the offline demo. | test | medium | fix: re-calibrate offline_demo golden values for flatland-rl@edge-9 | 048ea5f | S-11 | - |

#### Phase 4: entering after a malfunction (#84) ✅

| ID | Findings | Step | Category | Severity | Commit message | Commit(s) | Needs | Blocked by |
|---|---|---|---|---|---|---|---|---|
| ✅ S-14 | - | Pin flatland-rl to `-12`, recalibrate. | build | low | chore: deps switch flatland-rl to 178-agents-living-on-the-edge-12, re-calibrate golden values | 49635a3 | flatland-rl#513 | - |
| ✅ S-15 | - | Test the missed malfunction-off-map entry. | test | medium | test: add coverage for use_entering_prevention missing malfunction-off-map entries | 1777cfa | S-14 | - |
| ✅ S-16 | - | Consider malfunction-off-map agents. | fix | medium | fix: consider malfunction-off-map agents in entering-prevention pairwise check | 245dcee | S-15 | - |
| ✅ S-17 | - | Only once the malfunction ends this step. | fix | medium | fix: only treat a malfunction-off-map agent as entering once its malfunction actually clears | f256035 | S-16 | - |

#### Phase 5: entering on the first step (#85) ✅

| ID | Findings | Step | Category | Severity | Commit message | Commit(s) | Needs | Blocked by |
|---|---|---|---|---|---|---|---|---|
| ✅ S-18 | - | First-step waiting agent with earliest departure 0. | fix | medium | fix: also detect a first-step earliest_departure=0 WAITING agent as entering | 668f1bc | flatland-rl#515 | - |
| ✅ S-19 | - | Pin flatland-rl to `-14`, recalibrate. | build | low | chore: deps switch flatland-rl to 178-agents-living-on-the-edge-14, re-calibrate golden values | 46e74cb | S-18 | - |

#### Phase 6: optional state machine (#89) ✅

| ID | Findings | Step | Category | Severity | Commit message | Commit(s) | Needs | Blocked by |
|---|---|---|---|---|---|---|---|---|
| ✅ S-20 | - | Document entering prevention in `CLAUDE.md`. | docs | low | docs: trim redundant wording in comments, document entering_prevention design in CLAUDE.md | 8d80bed | - | - |
| ✅ S-21 | - | Wrap the test envs, pin flatland-rl to a `-17` commit. | fix | medium | fix: wrap test RailEnv construction with RailEnvStateMachineWrapper, bump flatland-rl pin to edge-17 | ade8235 | flatland-rl#520 | - |
| ✅ S-22 | - | Recalibrate the offline demo. | test | medium | fix: re-calibrate offline_demo environments_v2 golden values for flatland-rl@edge-17 | 01322b0 | S-21 | - |

#### Phase 7: state machine on reload (#90) ✅

| ID | Findings | Step | Category | Severity | Commit message | Commit(s) | Needs | Blocked by |
|---|---|---|---|---|---|---|---|---|
| ✅ S-23 | - | Pin flatland-rl to the fix of the reload gap and ask the runners for state machine updates; keep it in the intermediate-stops test. | fix | medium | fix: bump flatland-rl pin to fix/policy-runner-skip-state-machine-update, request wrapped env for DLA | 52791df, cd1e404, a13b27f, 4543160 | flatland-rl#541 | - |
| ✅ S-24 | F-15 | Drop Python 3.10, test 3.14. | build | medium | chore: drop Python 3.10, test 3.14. (#91) | 34a3bce | - | - |
| ✅ S-25 | - | Track the agents' state in the online and offline demos, recalibrate. | fix | medium | fix: track agent.state for DLA in online/offline demos, re-calibrate envs_v2 for flatland-rl 6788c0a7 | be92953 | S-23 | - |
| ✅ S-26 | - | Pin at flatland-rl#541's rebased head, recalibrate for flatland-rl#545. | fix | medium | fix: pin flatland-rl to the rebased head of fix/policy-runner-skip-state-machine-update, re-calibrate the offline demo for flatland-rl #545. | ffd2477 | S-25 | - |

#### Phase 8: review findings 💡

| ID | Findings | Step | Category | Severity | Commit message | Commit(s) | Needs | Blocked by |
|---|---|---|---|---|---|---|---|---|
| ✅ S-27 | F-18 | Re-pin flatland-rl to c689b22f (#541's head after its rebase on main, with the fix for tuple rewards under pandas 3) in all three places. Checked against 30e7a35f, the same code before the rebase, under pandas 3.0.6: DLA tests and offline demo pass (20 passed, 50 skipped, 1 xfailed). | bug | high | fix: pin flatland-rl to #541's head with the pandas 3 fix for tuple rewards. | 4ef5ace | flatland-rl c689b22f | - |
| ✅ S-28 | F-1 | Steer only the crossings `STOP_MOVING` would get wrong; keep `STOP_MOVING` for a settled target. Test: the full-speed scenario stops with a free cell between the agents, and a switch scenario still takes the planned branch. Recalibrate if the values move. | bug | high | fix(dla): send a directional action for an unavoidable crossing only where STOP_MOVING would leave the planned path. | 70e8660 | - | - |
| ✅ S-29 | F-3, F-4, F-12 | Mirror the engine's departure rules in entering prevention (earliest departure 0 or 1 on the first step; one step ahead after a malfunction), with tests and `CLAUDE.md:163-165`. | bug | medium | fix(dla): mirror the engine's departure rules in entering prevention. | 9e18779 | - | - |
| ✅ S-30 | F-8 | Use `_check_action_on_agent` in `SetPathPolicy._act`, with a test of `SetPathPolicy` on its own. | bug | low | fix(policies): use flatland-rl's _check_action_on_agent in SetPathPolicy._act and test it. | 68740b5 | - | - |
| ✅ S-31 | F-14 | DLA reads `agent.derived_state(self.rail_env._elapsed_steps)` (`SetPathPolicy`: `env._elapsed_steps`) instead of `agent.state`; drop the runner option from the tests and the offline demo, the wrappers from the tests and the online client; `CLAUDE.md` (Q-2). | refactor | medium | refactor(dla): read the agents' derived state instead of requiring the state machine. | b9e4a9f | - | - |
| ✅ S-32 | F-10 | Assert the queued agents' positions, not their state. | test | low | test(dla): assert queued agents' positions, not their oscillating state. | 68a3bc1 | - | - |
| ✅ S-33 | F-6 | Pin the episodes archive in `checks.yaml` to the flatland-scenarios commit that last changed the v7 archive (2fc33319, on the branch of [flatland-scenarios#34](https://github.com/flatland-association/flatland-scenarios/pull/34), not merged), with a TODO to move to a commit on flatland-scenarios' `main` once #34 is merged. Only `metadata.csv` is read (offline demo), the same in v5 and v7; the replay of the recorded episodes is skipped (F-5). flatland-rl's own URL is a follow-up (§1.4). | ci | medium | ci: pin the benchmark episodes archive to a flatland-scenarios commit. | f618d51 | - | - |
| 💡 S-34 | F-7 | Reconcile `SetPathPolicy` with flatland-rl's `ShortestPathPolicy` after [flatland-rl#528](https://github.com/flatland-association/flatland-rl/pull/528): `SetPathPolicy` becomes a `ShortestPathPolicy` whose router is DLA's path planning (k shortest paths with cutoff, always-first strategy at intermediate stops), so the look-ahead (shared with DLA), the route index (routes visiting a cell twice) and the routing per episode come from flatland-rl; `SetPathPolicy` decides without train state (done: `target_entry_point is not None`), as `ShortestPathPolicy` does on its observation; pin flatland-rl to a commit with #528; recalibrate if the values move. Formerly: share the look-ahead between `SetPathPolicy` and DLA locally. | refactor | medium | refactor(policies): base SetPathPolicy on flatland-rl's ShortestPathPolicy, with DLA's path planning as its router. |  | S-30, S-38 | flatland-rl#528 merged |
| ✅ S-35 | F-11 | Split `_extract_agent_can_move`, rename the helper for entering, log through one helper; and decide without train state (the rest of Q-2): replace DLA's seven reads of `_derived_state` by the facts they stand for (on the map, done, in malfunction, due to depart, `is_cell_exit()`), with the expected values unchanged. | refactor | low | refactor(dla): split _extract_agent_can_move, decide without train state and log through one audit helper. | 8e5760c | S-28, S-29 | - |
| ✅ S-36 | F-13 | Type hints for the per-step state (`forced_action`, `switches`, `agent_positions`, `_about_to_leave_off_map`); the wrapper's return value is obsolete since S-31 removed the wrapper. | style | low | style(dla): declare per-step state with type hints. | a2ba274 | S-31 | - |
| ✅ S-38 | - | Re-pin flatland-rl to flatland-rl#541's merge commit on `main` in all three places, so `main` doesn't pin a commit of #541's deleted branch after the stack merge; recalibrate if the values move. Pin 9eec95df (merged on 2026-11-13); its engine code (`rail_env.py`, `speed_counter.py`) is the same as at c689b22f, and DLA tests and offline demo pass unchanged against it (22 passed, 50 skipped, 1 xfailed, with S-28's changes). | build | high | fix: pin flatland-rl to the merge commit of flatland-rl#541 on main. | 15d8573 | S-27 | - |
| 💡 S-39 | F-21 | Check that both agents entered before comparing their entry steps, in the two entering tests of S-29. | test | low | test(dla): check that both agents entered before comparing their entry steps. |  | - | - |
| 💡 S-40 | F-22 | Rename `_is_on_map` after what it means (on the map and not done), in its four uses and `CLAUDE.md`. | style | low | style(dla): name the on-map helper for agents that are on the map and not done. |  | - | - |
| 💡 S-41 | F-23 | Store `agent_can_move` entries as tuples, as declared. | style | low | style(dla): store agent_can_move entries as the tuples they are declared as. |  | - | - |
| 💡 S-42 | F-24 | `SetPathPolicy._act` returns `DO_NOTHING` for an agent without a set path; a test keeps the agent at its target (`remove_agents_at_target=False`). | bug | medium | fix(policies): let SetPathPolicy act for an agent parked at its target. |  | - | - |
| 💡 S-44 | F-27 | Agents that enter in the coming step go into the opposing-agent map in step (2.3); a test with two agents of earliest departure 0 and 1 heading towards each other, without entering prevention. | bug | low | fix(dla): see agents that enter the map in the first step as opposing agents. |  | - | - |
| 💡 S-45 | F-26 | Move `_log`/`_audit` into `SetPathPolicy`, with the step count from the env; use them for its three audit entries. | refactor | low | refactor(policies): move the audit and log helpers into SetPathPolicy. |  | - | - |
| 💡 S-46 | F-28 | The standalone `SetPathPolicy` test takes a real branch at a switch and asserts the exact turn action. | test | low | test(policies): let the standalone SetPathPolicy test take a real branch at a switch. |  | S-42 | - |
| 💡 S-47 | F-29 | `CLAUDE.md` names flatland-scenarios#37 for the skipped episode replay. | docs | low | docs: name flatland-scenarios#37 as the tracker of the skipped episode replay in CLAUDE.md. |  | - | - |

### 11.2 Rebase and merge

*How the branch is rebased or merged when it builds on other open PRs, and the known conflicts.*

- Chain: `main` ← `178-agents-living-on-the-edge-3`
  ([#80](https://github.com/flatland-association/flatland-baselines/pull/80)) ← `-5`
  ([#81](https://github.com/flatland-association/flatland-baselines/pull/81)) ← `-9`
  ([#83](https://github.com/flatland-association/flatland-baselines/pull/83)) ← `-12`
  ([#84](https://github.com/flatland-association/flatland-baselines/pull/84)) ← `-14`
  ([#85](https://github.com/flatland-association/flatland-baselines/pull/85)) ← `-17`
  ([#89](https://github.com/flatland-association/flatland-baselines/pull/89)) ← `-19`
  ([#90](https://github.com/flatland-association/flatland-baselines/pull/90)). Each PR's base is the branch below.
  The plan lives on the top branch.
- Merge (Q-3): GitHub's stack merge on #90, after S-36 and S-38; it merges #80-#90 into `main` bottom up. Then the follow-up
  pins flatland-rl 5.0.0 (§1.4), and only then is release-please's release PR merged. Prerequisites:
  - `main`'s ruleset (`main`, id 9165551) requires `test-dla-regression (3.11)` to `(3.14)` (strict), the versions #90
    runs; it required `(3.10)` before 2026-11-12 (F-19, fixed).
  - Every PR of the stack green against `main`'s rules: #80-#89 passed, but re-runs fail, as their flatland-rl
    branch pins no longer exist; #90 passes since 11887c7 (at ffd2477 `test-dla-regression` failed, F-18, fixed in
    4ef5ace).
  - One approval, code owner review (ruleset); merge methods squash or merge.
- If a lower branch has to change after all: `git rebase --onto <new base head> <old base head> <branch>` for each
  branch above it, in order; the golden values above it may need recalibrating again.

## 12. Change inventory

*Every file the change adds, modifies, renames or removes, and what stays unchanged on purpose.*

### 12.1 New files

*The files the change adds, and what each adds.*

| File | Adds |
|---|---|
| `docs/plans/agents-living-on-the-edge.md` | This plan. |
| `tests/test_set_path_policy.py` (S-30) | A test of `SetPathPolicy` on its own. Planned: an agent parked at its target (S-42), a real branch at a switch (S-46). |

### 12.2 Modified files

*The files the change modifies, and what changes in each.*

| File | Changes |
|---|---|
| `policy/deadlock_avoidance_policy.py` | Renames; look-ahead; `forced_action`; entering prevention for more shortcuts (`_enters_map_in_coming_step` since S-35); `_get_action` raises a `RuntimeError`; forced action only where `STOP_MOVING` leaves the path (`_stop_moving_leaves_path`, S-28); departure rules (S-29); `derived_state` (S-31), replaced by the agents' facts (S-35); type hints (S-36); `_extract_agent_can_move` split, `_log`/`_audit` (S-35). Planned: rename `_is_on_map` (S-40), `agent_can_move` as tuples (S-41), first-step entrants as opposing agents (S-44). |
| `policy/set_path_policy.py` | Renames; look-ahead in `_act`. `_check_action_on_agent`, `audit` initialised (S-30). `derived_state` (S-31). Planned: acts for an agent parked at its target (S-42), `_log`/`_audit` (S-45), based on flatland-rl's `ShortestPathPolicy` (S-34). |
| `observation/full_env_observation.py` | `ObservationT`. |
| `online_demo/submission/run_solution.py` | Wrapped the client's local env; S-31 removed it again. |
| `offline_demo/test_offline.py` | Golden values per phase; the runner option for the state machine came and went (S-31). |
| `tests/test_entering_prevention.py` | Malfunction-off-map tests; earliest departures 0 and 1, and a malfunction ending in the departure step (S-29). Planned: check that both agents entered first (S-39). |
| `tests/test_limitations_two_switches.py`, `tests/test_limitations_switch.py` | Half-speed agents; assertions on the derived state (S-31); full-speed test, stopping with a free cell; steering at a simple switch and stopping before a symmetric switch (S-28); planned: positions instead of states (S-32). |
| `tests/test_episodes_deadlock_avoidance.py` | Skipped, with TODO. |
| `tests/test_policy_grid_runner_evaluator.py`, `tests/test_policy_runner.py`, `tests/test_intermediate_stops.py` | Golden values; no runner option for the state machine (S-31). |
| `tests/rail_test_utils.py` | `is_valid_entry_point`. |
| `pyproject.toml`, `environment.yml`, `.github/workflows/checks.yaml` | flatland-rl pin (`main` since 21a3c5f; 9eec95df, #541's merge commit, in S-38); Python 3.11-3.14; episodes URL by commit (S-33). Planned: flatland-rl 5.0.0 in the follow-up (§1.4). |
| `CLAUDE.md` | Pins, layout, crossing timing, entering prevention. Departure rules (S-29), derived state (S-31). Planned: flatland-scenarios#37 for the skipped replay (S-47). |

Unchanged on purpose: the simple baselines, the Dockerfiles, `docker.yml`, `release-please.yml`, the DLA README.
Outside this repo: S-33 also changes flatland-rl's `checks.yml`.

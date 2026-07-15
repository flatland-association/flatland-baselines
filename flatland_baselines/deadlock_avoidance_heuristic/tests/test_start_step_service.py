from collections import defaultdict

import numpy as np

from flatland.env_generation.env_generator import env_generator_legacy
from flatland.envs.observations import FullEnvObservation
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy
from flatland_baselines.deadlock_avoidance_heuristic.policy.start_step_service import DeadlockAvoidanceStatefulObservationBuilder, \
    DeadlockAvoidanceInternalObservationBuilderState


def _make_service_with_state(num_agents: int, audit: bool = False) -> DeadlockAvoidanceStatefulObservationBuilder:
    service = DeadlockAvoidanceStatefulObservationBuilder(
        min_free_cell=1,
        count_num_opp_agents_towards_min_free_cell=False,
        use_switches_heuristic=False,
        use_entering_prevention=False,
        show_debug_plot=False,
        verbose=False,
        audit=audit,
    )
    service._state = DeadlockAvoidanceInternalObservationBuilderState(
        agent_positions=np.zeros((1, 1), dtype=int) - 1,
        full_shortest_distance_agent_map=np.zeros((num_agents, 1, 1), dtype=int),
        shortest_distance_positions_agent_map=defaultdict(set),
        shortest_distance_positions_directions_agent_map=defaultdict(lambda: defaultdict(set)),
        shortest_distance_agent_len=defaultdict(lambda: 0),
        shortest_distance_agent_map=np.zeros((num_agents, 1, 1), dtype=int),
        opp_agent_map=defaultdict(set),
    )
    return service


def test_policy_and_service_keep_independent_audit_lists():
    """
    `DeadLockAvoidancePolicy` and its `DeadlockAvoidanceStatefulObservationBuilder` intentionally keep separate audit lists:
    `DeadLockAvoidancePolicy._init_env` only passes a bool (`audit=self.audit is not None`) to
    `DeadlockAvoidanceStatefulObservationBuilder`, which then creates its own private `[]`. Entries appended inside
    `DeadlockAvoidanceStatefulObservationBuilder` (e.g. "agent X blocked by Y" from `_check_agent_can_move`) land in
    `policy.start_step_service.audit`, not in `policy.audit` -- the two are distinct list objects by
    design, not accidentally.
    """
    env, _, _ = env_generator_legacy(n_cities=2, line_length=2, obs_builder_object=FullEnvObservation(), seed=1)
    policy = DeadLockAvoidancePolicy(audit=True)
    policy.rail_env = env
    policy._init_env(env)

    assert policy.audit == []
    assert policy.start_step_service.audit == []
    assert policy.audit is not policy.start_step_service.audit


def test_invalidate_opposition_clears_cached_opposition():
    """
    `invalidate_opposition` must clear the *internal* `_state.opp_agent_map[handle]` -- the dict
    `_build_shortest_distance_agent_map` reads as `prev_opp_agents` to decide whether to skip recomputing
    an agent's opposition tracking on the next `start_step()`. If a caller (e.g. `_find_alternative`,
    after rerouting an agent) mutates anything other than this exact dict, the change would silently not
    take effect.
    """
    service = _make_service_with_state(num_agents=2)
    # simulate agent 0 having been tracked, before the reroute, as currently opposed by agent 1.
    service._state.opp_agent_map[0] = {1}

    service.invalidate_opposition(0)

    assert service._state.opp_agent_map[0] == set()

from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import _get_free_from_path

if __name__ == '__main__':
    wp_1 = []
    wp_2 = []

    print(_get_free_from_path(wp_1, wp_2))
    print(_get_free_from_path(wp_2, wp_1))

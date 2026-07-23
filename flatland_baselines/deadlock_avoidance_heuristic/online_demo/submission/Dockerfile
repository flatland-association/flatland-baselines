# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG TAG=latest-py3.12
FROM ghcr.io/flatland-association/flatland-rl:${TAG}

RUN mkdir -p flatland_baselines/deadlock_avoidance_heuristic
COPY flatland_baselines/deadlock_avoidance_heuristic/online_demo/submission/run.sh ./
COPY flatland_baselines/deadlock_avoidance_heuristic/ ./flatland_baselines/deadlock_avoidance_heuristic
COPY flatland_baselines/deadlock_avoidance_heuristic/online_demo/submission/run_solution.py ./

ENTRYPOINT ["bash", "run.sh"]

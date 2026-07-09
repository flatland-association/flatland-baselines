# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG TAG=latest-py3.12
FROM ghcr.io/flatland-association/flatland-rl:${TAG}

COPY ./ ./
RUN pip install -e .

ENV POLICY=flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy.DeadLockAvoidancePolicy
ENV OBS_BUILDER=flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation.FullEnvObservation
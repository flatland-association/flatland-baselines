# TODO use released version
# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG TAG=latest
FROM ghcr.io/flatland-association/flatland-baselines:${TAG}

ENV POLICY=flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy.DeadLockAvoidancePolicy
ENV OBS_BUILDER=flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation.FullEnvObservation
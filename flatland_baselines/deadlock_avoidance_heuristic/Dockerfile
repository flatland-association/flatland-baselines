# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG TAG=latest
FROM ghcr.io/flatland-association/flatland-baselines:${TAG}

ENV POLICY_PKG=flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy
ENV POLICY_CLS=DeadLockAvoidancePolicy
ENV OBS_BUILDER_PKG=flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation
ENV OBS_BUILDER_CLS=FullEnvObservation
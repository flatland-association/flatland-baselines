# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG FLATLAND_RL_REF=latest-py3.12
FROM ghcr.io/flatland-association/flatland-rl:${FLATLAND_RL_REF}

# DEPENDENCY SWITCH: pip install flatland-rl from a git+https URL (e.g.
# git+https://github.com/flatland-association/flatland-rl.git@some-branch), overriding whatever version the
# base image above has baked in. Leave empty (default) to keep the base image's own flatland-rl. A plain
# GitHub tarball URL does not work here: flatland-rl uses setuptools_scm, which needs real git metadata
# (stripped from tarball downloads) to infer a version.
ARG FLATLAND_RL_PIP_URL=""

COPY ./ ./
RUN pip install -e .
RUN if [ -n "$FLATLAND_RL_PIP_URL" ]; then pip install --force-reinstall --no-deps "$FLATLAND_RL_PIP_URL"; fi

ENV POLICY=flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy.DeadLockAvoidancePolicy
ENV OBS_BUILDER=flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation.FullEnvObservation
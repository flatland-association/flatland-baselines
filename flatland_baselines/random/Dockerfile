# TODO use released version
# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG TAG=entrypoint-refactoring
FROM ghcr.io/flatland-association/flatland-baselines:${TAG}

ENV POLICY=tests.trajectories.test_trajectories.RandomPolicy
ENV OBS_BUILDER=flatland.core.env_observation_builder.DummyObservationBuilder

# fail fast (verirfy at build time that POLICY and OBS_BUILDER are on PYTHONPATH)
RUN echo "from flatland.utils.cli_utils import resolve_type" > test.py && \
  echo "assert resolve_type('${POLICY}') is not None" >> test.py && \
  echo "assert resolve_type('${OBS_BUILDER}') is not None" >> test.py

RUN bash entrypoint_generic.sh python test.py



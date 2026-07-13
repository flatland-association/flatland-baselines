# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG FLATLAND_RL_REF=latest-py3.12
FROM ghcr.io/flatland-association/flatland-rl:${FLATLAND_RL_REF}

COPY ./ ./
RUN pip install -e .

ENV POLICY=flatland_baselines.forward_only_heuristic.forward_only_policy.ForwardOnlyPolicy
ENV OBS_BUILDER=flatland.core.env_observation_builder.DummyObservationBuilder

# fail fast (verify at build time that POLICY and OBS_BUILDER are on PYTHONPATH)
RUN echo "from flatland.utils.cli_utils import resolve_type" > test.py && \
  echo "assert resolve_type('${POLICY}') is not None" >> test.py && \
  echo "assert resolve_type('${OBS_BUILDER}') is not None" >> test.py

RUN pip install pytest
RUN python test.py



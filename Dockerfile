# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG FLATLAND_RL_REF=latest-py3.12
FROM ghcr.io/flatland-association/flatland-rl:${FLATLAND_RL_REF}

COPY ./ ./
RUN pip install -e .



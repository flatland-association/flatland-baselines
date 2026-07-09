# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG TAG=latest-py3.12
FROM ghcr.io/flatland-association/flatland-rl:${TAG}

COPY ./ ./
RUN pip install -e .



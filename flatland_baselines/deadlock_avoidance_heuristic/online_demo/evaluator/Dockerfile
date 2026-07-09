# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG TAG=latest-py3.12
FROM ghcr.io/flatland-association/flatland-rl:${TAG}

COPY run.sh ./
COPY evaluator.py ./

RUN mkdir -p /tmp/results/
RUN mkdir -p /tmp/analysis_data_dir/
RUN mkdir -p /tmp/episodes/
RUN mkdir -p /tmp/actions/
RUN mkdir -p /tmp/visualizations

ENTRYPOINT ["bash", "run.sh"]

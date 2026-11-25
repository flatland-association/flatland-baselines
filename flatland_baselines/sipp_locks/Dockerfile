# SIPP with corridor locks (ECML 2026 non-RL winner v4-sipp-locks), on flatland-rl 4.2.6.
#
# Phase 1 (calibration): built like the submitted image ghcr.io/darshanmakwana412/ecml2026:v4-sipp-locks, not like the
# other baselines of this repo:
# - flatland-rl is pinned to 4.2.6, the competition's version, not to the repo pin (pyproject.toml). Hence no
#   `pip install -e .`, which would upgrade flatland-rl, and the FLATLAND_RL_REF build argument of docker.yml is
#   not used.
# - Slim base image: the evaluator gives a pod 20 minutes to start, image pull included.
# Build from the repo root: docker build -f flatland_baselines/sipp_locks/Dockerfile .
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir flatland-rl==4.2.6

COPY flatland_baselines/__init__.py flatland_baselines/__init__.py
COPY flatland_baselines/sipp_locks/ flatland_baselines/sipp_locks/

ENV PYTHONPATH=/app
ENV POLICY=flatland_baselines.sipp_locks.or_policy.PlannerPolicy
ENV OBS_BUILDER=flatland_baselines.sipp_locks.or_policy.PlannerObsBuilder

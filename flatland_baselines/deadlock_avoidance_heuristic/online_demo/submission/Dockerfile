# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/
ARG TAG=latest-py3.12
FROM ghcr.io/flatland-association/flatland-rl:${TAG}

# DEPENDENCY SWITCH: pip install flatland-rl from a git+https URL (e.g.
# git+https://github.com/flatland-association/flatland-rl.git@some-branch), overriding whatever version the
# base image above has baked in. Leave empty (default) to keep the base image's own flatland-rl. A plain
# GitHub tarball URL does not work here: flatland-rl uses setuptools_scm, which needs real git metadata
# (stripped from tarball downloads) to infer a version.
ARG FLATLAND_RL_PIP_URL=""
RUN if [ -n "$FLATLAND_RL_PIP_URL" ]; then pip install --force-reinstall --no-deps "$FLATLAND_RL_PIP_URL"; fi

RUN mkdir -p flatland_baselines/deadlock_avoidance_heuristic
COPY flatland_baselines/deadlock_avoidance_heuristic/online_demo/submission/run.sh ./
COPY flatland_baselines/deadlock_avoidance_heuristic/ ./flatland_baselines/deadlock_avoidance_heuristic
COPY flatland_baselines/deadlock_avoidance_heuristic/online_demo/submission/run_solution.py ./

ENTRYPOINT ["bash", "run.sh"]

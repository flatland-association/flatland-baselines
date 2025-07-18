FROM continuumio/miniconda3

RUN apt-get update && apt-get install gcc build-essential wget zip -y

# Replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Use non-root user
RUN useradd conda --home-dir /home/conda --create-home
RUN chown -R conda /opt/conda
USER conda
WORKDIR /home/conda

# setup flatland-rl conda env
COPY environment.yml ./
RUN conda --version  && \
    conda env create -f environment.yml && \
    conda init bash && \
    source /home/conda/.bashrc && \
    source activate base && \
    conda env list  && \
    conda activate flatland-rl && \
    python -c 'from flatland.evaluators.client import FlatlandRemoteClient'

RUN mkdir -p flatland_baselines/deadlock_avoidance_heuristic
COPY run.sh ./
COPY entrypoint_generic.sh ./
COPY flatland_baselines/deadlock_avoidance_heuristic/ ./flatland_baselines/deadlock_avoidance_heuristic
COPY run_solution.py ./

# TODO should we make generic entrypoint the default in baselines and have run.sh only in starterkit?
ENTRYPOINT ["bash", "run.sh"]

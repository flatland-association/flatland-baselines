FROM continuumio/miniconda3

RUN apt-get update && apt-get install gcc build-essential wget zip ffmpeg -y

# Replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Use non-root user
RUN useradd conda --home-dir /home/conda --create-home
RUN chown -R conda /opt/conda
USER conda
WORKDIR /home/conda

# setup flatland-baselines conda env
COPY environment.yml ./

RUN conda --version  && \
    conda env create -f environment.yml && \
    conda init bash && \
    source /home/conda/.bashrc && \
    source activate base && \
    conda env list  && \
    conda activate flatland-baselines && \
    python -c 'from flatland.evaluators.client import FlatlandRemoteClient'

# DEPENDENCY SWITCH
# TODO use released version
# as long as we install flatland-rl via git+https@main, there seems no way to install flatland-rl[ml], so download requirements-ml.txt and install explicitly:
RUN wget https://raw.githubusercontent.com/flatland-association/flatland-rl/refs/heads/main/requirements-ml.txt -O requirements-ml.txt && \
    conda init bash && \
    source /home/conda/.bashrc && \
    source activate base && \
    conda activate flatland-baselines && \
    python -m pip install -U -r requirements-ml.txt && \
    python -c 'import torch' && \
    python -m pip cache purge && \
    conda clean --all

RUN mkdir -p flatland_baselines/deadlock_avoidance_heuristic
COPY run.sh ./
COPY entrypoint_generic.sh ./
COPY flatland_baselines/deadlock_avoidance_heuristic/ ./flatland_baselines/deadlock_avoidance_heuristic
COPY run_solution.py ./

# TODO should we make generic entrypoint the default in baselines and have run.sh only in starterkit?
ENTRYPOINT ["bash", "run.sh"]

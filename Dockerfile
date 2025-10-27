FROM continuumio/miniconda3

RUN apt-get update && \
    apt-get install gcc build-essential wget zip ffmpeg -y && \
    apt-get clean && \
    ffmpeg --help

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
    python -m pip cache purge && \
    conda clean --all && \
    python -c 'import torch'

RUN mkdir -p flatland_baselines/
COPY --chmod=0755 entrypoint_generic.sh ./
COPY flatland_baselines/ ./flatland_baselines/

ENTRYPOINT ["bash", "entrypoint_generic.sh"]

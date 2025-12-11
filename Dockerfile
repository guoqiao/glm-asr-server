FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# git is needed to install pkg via repo
RUN apt-get update -qq \
    && apt-get install -y -qq \
        git \
        time \
        tree \
        ffmpeg \
    && apt-get clean

WORKDIR /workspace

ADD requirements.txt .
RUN pip install -r requirements.txt

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

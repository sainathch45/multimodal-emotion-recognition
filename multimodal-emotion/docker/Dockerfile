FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /workspace/
RUN pip3 install --upgrade pip && pip3 install -r /workspace/requirements.txt
COPY . /workspace

CMD ["bash"]

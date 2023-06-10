FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

LABEL org.opencontainers.image.source https://github.com/unionai-oss/stanford-alpaca

WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

ARG VERSION
ARG DOCKER_IMAGE

RUN apt-get update && apt-get install build-essential git -y

COPY ./requirements.txt /root/requirements.txt
WORKDIR /root

# Pod tasks should be exposed in the default image
RUN pip install -r requirements.txt
RUN DS_BUILD_OPS=1 DS_BUILD_AIO=0 DS_BUILD_SPARSE_ATTN=0 pip install deepspeed --force-reinstall
RUN pip install multiprocess==0.70.11.1

COPY . /root
WORKDIR /root

ENV FLYTE_INTERNAL_IMAGE "$DOCKER_IMAGE"

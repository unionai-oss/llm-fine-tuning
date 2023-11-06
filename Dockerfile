FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

RUN apt-get update && apt-get install build-essential git -y

COPY ./requirements.txt /root/requirements.txt
WORKDIR /root

RUN pip install -r requirements.txt
# install this here due to 
RUN pip install apache_beam
# reinstall deepspeed to pre-compile plugins
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 DS_BUILD_OPS=1 DS_BUILD_AIO=0 DS_BUILD_SPARSE_ATTN=0 pip install deepspeed==0.10.0 --force-reinstall

COPY . /root
WORKDIR /root

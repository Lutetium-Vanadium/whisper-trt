#---
# name: wtrt
# group: audio
# depends: [pytorch, torch2trt, onnxruntime, whisper]
# requires: '>=36'
# test: test.py
# notes: TensorRT optimized Whisper ASR from https://github.com/NVIDIA-AI-IOT/whisper_trt
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}
    
COPY . /opt/whisper_trt
WORKDIR /opt/whisper_trt

RUN pip3 install -e . && \
    mkdir -p ~/.cache && \
    ln -s /data/models/whisper ~/.cache/whisper && \
    ln -s /data/models/whisper ~/.cache/whisper_trt 


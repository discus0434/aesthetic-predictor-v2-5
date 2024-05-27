FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    CUDA_HOME=/usr/local/cuda-11.8 \
    LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH \
    PATH=/usr/local/cuda-11.8/bin:$PATH \
    PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.10/site-packages \
    TZ=Asia/Tokyo \
    PIP_FIND_LINKS="https://download.pytorch.org/whl/cu118 https://huggingface.github.io/autogptq-index/whl/cu118/"

RUN rm /bin/sh && ln -s /bin/bash /bin/sh \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && echo "export PIP_FIND_LINKS='https://download.pytorch.org/whl/cu118 https://huggingface.github.io/autogptq-index/whl/cu118/'" >> ~/.bashrc

RUN apt update \
    && apt upgrade -y \
    && apt install -y --no-install-recommends \
        libgl1-mesa-dev \
        curl \
        git \
        unzip \
        zip \
        python3.10 \
        python3-pip \
        python3-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /aesthetic-predictor-v2_5

COPY pyproject.toml README.md /aesthetic-predictor-v2_5/
COPY src /aesthetic-predictor-v2_5/src

RUN pip3 install \
    torch==2.2.1+cu118 \
    torchvision==0.17.1+cu118 \
    xformers==0.0.25+cu118 \
    triton==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install \
        numpy==1.26.3 \
        pandas==2.2.1 \
        einops==0.7.0 \
        transformers==4.40.1 \
    && pip3 install -e .

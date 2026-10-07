FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#RUN apt-get update && apt-get install -y \
#    cmake  \
#    build-essential  \
#    libboost-dev  \
#    libboost-system-dev  \
#    libboost-filesystem-dev

## Instalando NVCC para CUDA
#RUN apt-get update && apt-get install -y \
#    wget \
#    gnupg \
#    ca-certificates \
#    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
#    && dpkg -i cuda-keyring_1.1-1_all.deb \
#    && rm cuda-keyring_1.1-1_all.deb \
#    && apt-get update \
#    && apt-get install -y cuda-toolkit-13-3 \
#            libnccl2 \
#            libnccl-dev \
#    && rm -rf /var/lib/apt/lists/*

##Asi Configuramos a NVCC para onligar a que quede en el PATH
#ENV PATH="/usr/local/cuda/bin:${PATH}"
#ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Instalar Pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash

ENV PATH="/root/.pixi/bin:$PATH"

# Entorno Pixi
RUN mkdir -p /entorno_virtual

WORKDIR /espacio_trabajo

CMD ["bash"]
ARG CUDA_BASE_VERSION=11.3.0
ARG UBUNTU_RELEASE_VERSION=20.04

FROM nvidia/cuda:${CUDA_BASE_VERSION}-devel-ubuntu${UBUNTU_RELEASE_VERSION}

ARG CUDNN_VERSION=8.2.0
ARG LLVM_VERSION=12

# ==============================================================================
# CUDA & cuDNN
# ==============================================================================

RUN apt-get update && \
    CUDA_VERSION_=$(printf ${CUDA_VERSION} | grep -oE '[0-9]+.[0-9]+') && \
    CUDNN_MAJOR_VERSION=$(printf ${CUDNN_VERSION} | grep -oE '[0-9]+' | head -1) && \
    apt-get install -y --no-install-recommends \
        cuda-samples-${CUDA_VERSION_} \
        cuda-nsight-compute-${CUDA_VERSION_} \
        cuda-nsight-systems-${CUDA_VERSION_} \
        libcudnn${CUDNN_MAJOR_VERSION}=${CUDNN_VERSION}*+cuda${CUDA_VERSION_} \
        libcudnn${CUDNN_MAJOR_VERSION}-dev=${CUDNN_VERSION}*+cuda${CUDA_VERSION_} && \
    rm -rf /var/lib/apt/lists/*

# ==============================================================================
# LLVM & Clang
# ==============================================================================

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        wget gnupg \
        software-properties-common && \
    rm -rf /var/lib/apt/lists/*

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-add-repository "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-${LLVM_VERSION} main"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        llvm-${LLVM_VERSION} \
        llvm-${LLVM_VERSION}-dev clang-${LLVM_VERSION} && \
    ln -s /usr/lib/llvm-${LLVM_VERSION} /usr/lib/llvm && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=${PATH}:/usr/lib/llvm/bin

# ==============================================================================
# C++ & Python
# ==============================================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        vim build-essential gdb python3-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && rm -f get-pip.py

RUN pip install cmake

# ==============================================================================
# Other Dependencies
# ==============================================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl libtinfo-dev libedit-dev \
        libxml2-dev zlib1g-dev && \
    printf "git config --global credential.helper store\n" >> /root/.bashrc && \
    rm -rf /var/lib/apt/lists/*

RUN pip install numpy scipy decorator attrs psutil typed_ast cython six \
                xgboost tornado pytest synr cloudpickle sklearn
RUN pip install transformers==3.0 && \
    pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /mnt

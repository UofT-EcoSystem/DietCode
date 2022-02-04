#!/bin/bash -e

PROJECT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/..

if [ $1 != "tvm" ] && [ $1 != "tvm_base" ]; then
        printf "Expecting one command line argument (tvm/tvm_base), but got $1 instead\n"
        exit -1
fi

cd ${PROJECT_ROOT}/$1

mkdir -p build && cd build

cmake -DUSE_CUDA=/usr/local/cuda/ \
      -DUSE_LLVM=/usr/lib/llvm/bin/llvm-config \
      -DUSE_CUBLAS=1 \
      -DUSE_CUDNN=1 ..
make -j 4

cd ${PROJECT_ROOT}/$1/python

python3 setup.py build

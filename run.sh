#!/bin/bash
# 设置 CUDA 环境变量
source ./setup_env.sh

make -j4 DEBUG=1 TRACE=1 NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80" CUDA_INC=$CUDA_INC CUDA_LIB=$CUDA_LIB
#!/bin/bash  
# mpic++ -o test/test_dev_process test/test_dev_process.cpp -I$CUDA_INC -I$NCCL_ROOT_DIR/build/include -L$CUDA_LIB -L$NCCL_ROOT_DIR/build/lib -lnccl -Wl,-rpath,$CUDA_LIB:$NCCL_ROOT_DIR/build/lib

mpic++ -o test/test_2node_4gpu_tp_dp test/test_2node_4gpu_tp_dp.cpp -I$CUDA_INC -I$NCCL_ROOT_DIR/build/include -L$CUDA_LIB -L$NCCL_ROOT_DIR/build/lib -lnccl -Wl,-rpath,$CUDA_LIB:$NCCL_ROOT_DIR/build/lib
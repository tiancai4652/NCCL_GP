#!/bin/bash  
# NCCL_GP 环境变量设置脚本  
  
export NCCL_ROOT_DIR=/workspace/fake-nccl/NCCL_GP
export CUDA_LIB=$NCCL_ROOT_DIR/fake_cuda/lib  
export CUDA_INC=$NCCL_ROOT_DIR/fake_cuda/include  
export LD_LIBRARY_PATH=$NCCL_ROOT_DIR/fake_cuda/lib:$NCCL_ROOT_DIR/build/lib  
export NCCL_TOPO_FILE=$NCCL_ROOT_DIR/topo/nvlink_5GPU.xml  
export NCCL_GRAPH_DUMP_FILE=$NCCL_ROOT_DIR/topo/graph_dump.xml  
export GPU_DEV_NUM=5  
export NCCL_DEBUG=TRACE  
export NCCL_DEBUG_SUBSYS=ALL  
  
echo "NCCL_GP 环境变量已设置:"  
echo "  NCCL_ROOT_DIR=$NCCL_ROOT_DIR"  
echo "  CUDA_INC=$CUDA_INC"  
echo "  CUDA_LIB=$CUDA_LIB"
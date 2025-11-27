#!/bin/bash

export NCCL_TOPO_FILE=../topo/2node_16gpu.xml
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export GPU_DEV_NUM=16
export NCCL_GRAPH_DUMP_FILE=../topo/2node_16gpu_graph_dump.xml

# 使用 mpirun 启动 16 个进程
# 每个进程根据 rank 设置自己的 NCCL_HOSTID
mpirun -np 16 --allow-run-as-root --output-filename rank_logs \
    -x LD_LIBRARY_PATH -x NCCL_TOPO_FILE -x NCCL_GRAPH_DUMP_FILE \
    -x GPU_DEV_NUM -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
    bash -c '
      if [ $OMPI_COMM_WORLD_RANK -lt 8 ]; then
        export NCCL_HOSTID=0
      else
        export NCCL_HOSTID=1
      fi
      exec ./test_2node_16gpu_tp_dp
    '


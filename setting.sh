export NCCL_ROOT_DIR=/home/zhangran/work/NCCL-SHARP/NCCL_GP
export CUDA_LIB=$NCCL_ROOT_DIR/fake_cuda/lib
export CUDA_INC=$NCCL_ROOT_DIR/fake_cuda/include
export LD_LIBRARY_PATH=$NCCL_ROOT_DIR/fake_cuda/lib:$NCCL_ROOT_DIR/build/lib
export NCCL_TOPO_FILE=$NCCL_ROOT_DIR/topo/nvlink_5GPU.xml
export NCCL_GRAPH_DUMP_FILE=$NCCL_ROOT_DIR/topo/graph_dump.xml
export GPU_DEV_NUM=5
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL


chmod +x $NCCL_ROOT_DIR/run.sh
chmod +x $NCCL_ROOT_DIR/src/collectives/device/gen_rules.sh

# 追加，确保在 test 目录下运行时能找到动态库
export LD_LIBRARY_PATH=$(pwd)/../build/lib:$(pwd)/../fake_cuda/lib:$LD_LIBRARY_PATH

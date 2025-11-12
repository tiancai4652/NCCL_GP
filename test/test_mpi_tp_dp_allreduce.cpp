/*************************************************************************
 * Copyright (c) 2025, NCCL-SHARP Project. All rights reserved.
 *
 * Multi-node TP/DP AllReduce Test with MPI + NCCL
 * TP Size = 8 (intra-node), DP Size = 16 (inter-node)
 * Total: 16 nodes × 8 GPUs = 128 GPUs
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "flow_extractor.h"

// 使用NCCL内置的宏，但需要避免重定义警告
#undef CUDACHECK
#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#ifndef TEST_NCCLCHECK
#undef NCCLCHECK
#define TEST_NCCLCHECK(cmd) do {                    \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#endif

int main(int argc, char* argv[]) {  
    // 初始化 MPI  
    MPI_Init(&argc, &argv);  
      
    int myRank, nRanks;  
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);  
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);  
      
    // 配置参数  
    int TP_SIZE = 8;  
    int DP_SIZE = 2;  
      
    char* env_tp = getenv("TEST_TP_SIZE");  
    char* env_dp = getenv("TEST_DP_SIZE");  
    if (env_tp) TP_SIZE = atoi(env_tp);  
    if (env_dp) DP_SIZE = atoi(env_dp);  
      
    const int TOTAL_GPUS = TP_SIZE * DP_SIZE;  
      
    if (nRanks != TOTAL_GPUS) {  
        if (myRank == 0) {  
            printf("Error: Expected %d MPI processes, got %d\n", TOTAL_GPUS, nRanks);  
        }  
        MPI_Finalize();  
        return 1;  
    }  
      
    int nodeId = myRank / TP_SIZE;  
    int localRank = myRank % TP_SIZE;  
      
    if (myRank == 0) {  
        printf("========================================\n");  
        printf("Multi-node NCCL TP/DP AllReduce Test\n");  
        printf("========================================\n");  
        printf("Configuration:\n");  
        printf("  Total nodes: %d\n", DP_SIZE);  
        printf("  GPUs per node: %d (TP Size)\n", TP_SIZE);  
        printf("  Total GPUs: %d\n", TOTAL_GPUS);  
        printf("========================================\n\n");  
    }  
      
    printf("Rank %d: nodeId=%d, localRank=%d\n", myRank, nodeId, localRank);  
    MPI_Barrier(MPI_COMM_WORLD);  
      
    // ========== 关键修改: 使用全局通信器初始化 ==========  
    if (myRank == 0) {  
        printf("\n[Step 1] Creating global NCCL communicator...\n");  
    }  
      
    // 1. 创建全局通信器  
    ncclUniqueId globalId;  
    ncclComm_t globalComm;  
      
    if (myRank == 0) {  
        TEST_NCCLCHECK(ncclGetUniqueId(&globalId));  
    }  
    MPI_Bcast(&globalId, sizeof(globalId), MPI_BYTE, 0, MPI_COMM_WORLD);  

    // CUDACHECK(cudaSetDevice(localRank)); 
      
    // 使用全局 myRank 初始化全局通信器  
    TEST_NCCLCHECK(ncclCommInitRank(&globalComm, nRanks, globalId, myRank));  
      
    if (myRank == 0) {  
        printf("✓ Global NCCL communicator initialized\n");  
    }  
      
    // 2. 使用 ncclCommSplit 创建节点内 TP 通信器  
    if (myRank == 0) {  
        printf("\n[Step 2] Creating intra-node TP communicators...\n");  
    }  
      
    ncclComm_t tpComm;  
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;  
      
    // 按 nodeId 分组,按 localRank 排序  
    TEST_NCCLCHECK(ncclCommSplit(globalComm, nodeId, localRank, &tpComm, &config));  
      
    if (myRank == 0) {  
        printf("✓ TP communicators created for all nodes\n");  
    }  
      
    // 3. 设置内存缓冲区  
    if (myRank == 0) {  
        printf("\n[Step 3] Setting up memory buffers...\n");  
    }  
      
    const size_t tensor_elements = 512 * 4096;  
    const size_t bytes = tensor_elements * sizeof(float);  
      
    void* sendbuff = NULL;  
    void* recvbuff = NULL;  
    cudaStream_t stream = (cudaStream_t)0;  
      
    if (myRank == 0) {  
        printf("  Tensor size per GPU: %zu elements (%.2f MB)\n",   
               tensor_elements, bytes / (1024.0 * 1024.0));  
        printf("✓ Buffer setup complete\n");  
    }  
      
    // 4. 启用流提取  
    if (myRank == 0) {  
        printf("\n[Step 4] Enabling flow extraction...\n");  
        TEST_NCCLCHECK(ncclSetFlowExtractionEnabled(1));  
        printf("✓ Flow extraction enabled\n");  
    }  
      
    MPI_Barrier(MPI_COMM_WORLD);  
      
    // 5. 执行节点内 AllReduce (使用 TP 通信器)  
    if (myRank == 0) {  
        printf("\n[Step 5] Performing intra-node AllReduce (TP communication)...\n");  
        printf("  Each node performs AllReduce among its %d GPUs\n", TP_SIZE);  
    }  
      
    // 使用 TP 通信器执行 AllReduce  
    TEST_NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, tensor_elements,   
                                 ncclFloat, ncclSum, tpComm, stream));  
      
    MPI_Barrier(MPI_COMM_WORLD);  
      
    if (myRank == 0) {  
        printf("✓ All nodes completed intra-node AllReduce\n");  
    }  
      
    // 6. 写入流信息  
    if (myRank == 0) {  
        printf("\n[Step 6] Writing aggregated flow information...\n");  
        TEST_NCCLCHECK(ncclWriteAggregatedFlow(tpComm));  
        printf("✓ Flow records written\n");  
    }  
      
    // 7. 清理资源  
    if (myRank == 0) {  
        printf("\n[Step 7] Cleaning up resources...\n");  
    }  
      
    ncclCommDestroy(tpComm);  
    ncclCommDestroy(globalComm);  
      
    MPI_Barrier(MPI_COMM_WORLD);  
      
    if (myRank == 0) {  
        printf("✓ Cleanup complete\n");  
        printf("\n========================================\n");  
        printf("Multi-node TP AllReduce Test completed!\n");  
        printf("========================================\n");  
    }  
      
    MPI_Finalize();  
    return 0;  
}

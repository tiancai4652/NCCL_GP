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

    // 添加调试输出  
    char* ompi_rank = getenv("OMPI_COMM_WORLD_RANK");  
    char* pmi_rank = getenv("PMI_RANK");  
    printf("Rank %d: OMPI_COMM_WORLD_RANK=%s, PMI_RANK=%s\n",   
           myRank, ompi_rank ? ompi_rank : "NULL", pmi_rank ? pmi_rank : "NULL");  
           
      
    //  // 显式设置设备 - 这是最可靠的方式  
    // // 使用 dlsym 动态查找符号,避免链接时依赖  
    // typedef int (*cudaSetDevice_t)(int);  
    // cudaSetDevice_t cuda_set_device = (cudaSetDevice_t)dlsym(RTLD_DEFAULT, "cudaSetDevice");  
    // if (cuda_set_device) {  
    //     printf("cuda_set_device: %d\n", myRank);
    //     cuda_set_device(myRank);  
    // } 
    
    // 配置参数（可通过环境变量覆盖）
    int TP_SIZE = 8;        // 每个节点内的GPU数量
    int DP_SIZE = 16;       // 节点数量
    
    // 从环境变量读取配置
    char* env_tp = getenv("TEST_TP_SIZE");
    char* env_dp = getenv("TEST_DP_SIZE");
    //char* env_gpu_num = getenv("GPU_DEV_NUM");
    if (env_tp) TP_SIZE = atoi(env_tp);
    if (env_dp) DP_SIZE = atoi(env_dp);
    
    const int TOTAL_GPUS = TP_SIZE * DP_SIZE;  // 总GPU数量
    
    // 验证MPI进程数
    if (nRanks != TOTAL_GPUS) {
        if (myRank == 0) {
            printf("Error: Expected %d MPI processes, got %d\n", TOTAL_GPUS, nRanks);
            printf("Usage: mpirun -np %d %s\n", TOTAL_GPUS, argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    // 计算节点信息
    int nodeId = myRank / TP_SIZE;      // 节点ID: 0-15
    int localRank = myRank % TP_SIZE;   // 节点内rank: 0-7
    
    // 设置NCCL调试级别
    if (myRank == 0) {
        setenv("NCCL_DEBUG", "WARN", 1);
    }
    
    // 打印配置信息（只有rank 0打印总体信息）
    if (myRank == 0) {
        printf("========================================\n");
        printf("Multi-node NCCL TP/DP AllReduce Test\n");
        printf("========================================\n");
        printf("Configuration:\n");
        printf("  Total nodes: %d\n", DP_SIZE);
        printf("  GPUs per node: %d (TP Size)\n", TP_SIZE);
        printf("  Total GPUs: %d\n", TOTAL_GPUS);
        printf("  MPI processes: %d\n", nRanks);
        printf("========================================\n\n");
    }
    
    // 每个进程打印自己的信息
    printf("Rank %d: nodeId=%d, localRank=%d\n", myRank, nodeId, localRank);
    MPI_Barrier(MPI_COMM_WORLD);  // 同步输出
    
    // 创建节点内的 MPI 通信器
    MPI_Comm nodeComm;
    // 参数：原通信器，分组键(nodeId)，排序键(localRank)，新通信器
    MPI_Comm_split(MPI_COMM_WORLD, nodeId, localRank, &nodeComm);
    
    if (myRank == 0) {
        printf("\n[Step 1] Creating intra-node NCCL communicators...\n");
    }
    
    // 节点内初始化 NCCL
    ncclUniqueId ncclId;
    ncclComm_t comm;
    
    // 节点内rank 0获取唯一ID并广播给同节点的其他进程
    if (localRank == 0) {
        TEST_NCCLCHECK(ncclGetUniqueId(&ncclId));
    }
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, nodeComm);
    
    // *** 关键修改: 设置虚拟GPU设备 ***
    // 在fake_cuda环境下，每个进程必须使用不同的GPU ID来避免重复GPU检测错误
    // CUDACHECK(cudaSetDevice(myRank));  // 使用全局 myRank 作为虚拟 GPU ID
    
    // 初始化NCCL通信器
    TEST_NCCLCHECK(ncclCommInitRank(&comm, TP_SIZE, ncclId, localRank));
    
    if (myRank == 0) {
        printf("✓ NCCL communicators initialized for all nodes\n");
    }
    
    // 在fake_cuda环境下，模拟GPU内存分配
    if (myRank == 0) {
        printf("\n[Step 2] Setting up memory buffers...\n");
    }
    
    // 测试数据大小
    const size_t tensor_elements = 512 * 4096;  // ~8MB per GPU
    const size_t bytes = tensor_elements * sizeof(float);
    
    // 在fake_cuda环境下使用NULL指针
    void* sendbuff = NULL;
    void* recvbuff = NULL;
    cudaStream_t stream = (cudaStream_t)0;  // 默认流
    
    if (myRank == 0) {
        printf("  Tensor size per GPU: %zu elements (%.2f MB)\n", 
               tensor_elements, bytes / (1024.0 * 1024.0));
        printf("  Total data per node: %.2f MB\n", 
               (bytes * TP_SIZE) / (1024.0 * 1024.0));
        printf("  Note: Using fake_cuda environment\n");
        printf("✓ Buffer setup complete\n");
    }
    
    // 启用流提取（只在rank 0启用）
    if (myRank == 0) {
        printf("\n[Step 3] Enabling flow extraction...\n");
        TEST_NCCLCHECK(ncclSetFlowExtractionEnabled(1));
        printf("✓ Flow extraction enabled\n");
    }
    
    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 执行节点内 AllReduce (TP 通信)
    if (myRank == 0) {
        printf("\n[Step 4] Performing intra-node AllReduce (TP communication)...\n");
        printf("  Each node performs AllReduce among its %d GPUs\n", TP_SIZE);
        printf("  Operation: AllReduce with ncclSum\n");
        printf("  Data type: float32\n");
    }
    
    // 所有节点同时执行AllReduce
    // 参数：发送缓冲区，接收缓冲区，元素数量，数据类型，操作类型，通信器，流
    TEST_NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, tensor_elements, 
                                 ncclFloat, ncclSum, comm, stream));
    
    // 在fake_cuda环境下不需要显式同步
    // CUDACHECK(cudaStreamSynchronize(stream));
    
    MPI_Barrier(MPI_COMM_WORLD);  // 确保所有节点都完成
    
    if (myRank == 0) {
        printf("✓ All nodes completed intra-node AllReduce\n");
    }
    
    // 写入聚合流信息（只在rank 0执行）
    if (myRank == 0) {
        printf("\n[Step 5] Writing aggregated flow information...\n");
        TEST_NCCLCHECK(ncclWriteAggregatedFlow(comm));
        printf("✓ Flow records written to output/<topo>/\n");
    }
    
    // 清理资源
    if (myRank == 0) {
        printf("\n[Step 6] Cleaning up resources...\n");
    }
    
    // 清理NCCL通信器
    ncclCommDestroy(comm);
    
    // 清理MPI通信器
    MPI_Comm_free(&nodeComm);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myRank == 0) {
        printf("✓ Cleanup complete\n");
        
        printf("\n========================================\n");
        printf("Multi-node TP AllReduce Test completed!\n");
        printf("========================================\n");
        printf("\nSummary:\n");
        printf("  - %d nodes participated\n", DP_SIZE);
        printf("  - %d GPUs per node (TP group)\n", TP_SIZE);
        printf("  - Total %d GPUs\n", TOTAL_GPUS);
        printf("  - Tensor size: %.2f MB per GPU\n", bytes / (1024.0 * 1024.0));
        printf("  - Total data moved: %.2f MB per node\n", 
               (bytes * TP_SIZE) / (1024.0 * 1024.0));
        printf("  - Flow records: check output/<topo>/ directory\n");
        printf("\nNote: This simulates intra-node TP communication\n");
        printf("      across %d nodes. Each node performs AllReduce\n", DP_SIZE);
        printf("      among its %d GPUs independently.\n", TP_SIZE);
        printf("\nFor inter-node DP communication, additional\n");
        printf("      NCCL communicators would be needed.\n");
    }
    
    MPI_Finalize();
    return 0;
}

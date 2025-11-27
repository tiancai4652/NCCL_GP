#include <stdio.h>  
#include "cuda_runtime.h"  
#include "nccl.h"  
#include "mpi.h"  
#include <unistd.h>  
#include <stdint.h>  
#include <stdlib.h>  
  
#define MPICHECK(cmd) do {                          \  
  int e = cmd;                                      \  
  if( e != MPI_SUCCESS ) {                          \  
    printf("Failed: MPI error %s:%d '%d'\n",        \  
        __FILE__,__LINE__, e);   \  
    exit(EXIT_FAILURE);                             \  
  }                                                 \  
} while(0)  
  
#define CUDACHECK(cmd) do {                         \  
  cudaError_t e = cmd;                              \  
  if( e != cudaSuccess ) {                          \  
    printf("Failed: Cuda error %s:%d '%s'\n",             \  
        __FILE__,__LINE__,cudaGetErrorString(e));   \  
    exit(EXIT_FAILURE);                             \  
  }                                                 \  
} while(0)  
  
#define NCCLCHECK(cmd) do {                         \  
  ncclResult_t r = cmd;                             \  
  if (r!= ncclSuccess) {                            \  
    printf("Failed, NCCL error %s:%d '%s'\n",             \  
        __FILE__,__LINE__,ncclGetErrorString(r));   \  
    exit(EXIT_FAILURE);                             \  
  }                                                 \  
} while(0)  
  
int main(int argc, char* argv[])  
{  
  printf("test_2node_16gpu_tp_dp\n");
  int size = 32*1024*1024;  
  int myRank, nRanks, localRank = 0;  
    
  // 初始化MPI  
  MPICHECK(MPI_Init(&argc, &argv));  
  // MPI_Barrier(MPI_COMM_WORLD); 
  printf("MPI_Init done\n");
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));  
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));  

  printf("myRank=%d, nRanks=%d\n", myRank, nRanks);
  
  // 2机16卡配置: TP=8, DP=2  
  // 节点0: rank 0-7 (TP组0)  
  // 节点1: rank 8-15 (TP组1)  
  int hostId;
  if (myRank < 8) {  
    hostId = 0;
    localRank = myRank;  
  } else {  
    hostId = 1;
    localRank = myRank - 8;  
  }  
  
  // 设置NCCL_HOSTID（fake_cuda需要数字）
  char hostIdStr[32];
  snprintf(hostIdStr, sizeof(hostIdStr), "%d", hostId);
  setenv("NCCL_HOSTID", hostIdStr, 1);
    
  printf("[Rank %d] Setting NCCL_HOSTID=%d, localRank=%d\n",   
         myRank, hostId, localRank);
    
  // 计算TP和DP的color和key  
  int tp_size = 8;  
  int dp_size = 2;  
  int tp_rank = myRank % tp_size;  // TP组内的rank  
  int dp_rank = myRank / tp_size;  // DP组的编号  
    
  printf("[Rank %d] tp_rank=%d, dp_rank=%d\n", myRank, tp_rank, dp_rank);  
    
  ncclUniqueId id;  
  ncclComm_t global_comm, tp_comm, dp_comm;  
  float *sendbuff, *recvbuff;  
  cudaStream_t s;  
    
  // rank 0获取NCCL unique ID并广播  
  if (myRank == 0) ncclGetUniqueId(&id);  
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));  
    
  // 根据localRank选择GPU  
  CUDACHECK(cudaSetDevice(localRank));  
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));  
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));  
  // CUDACHECK(cudaStreamCreate(&s));  
  s = (cudaStream_t)0;  
    
  // 1. 初始化全局NCCL communicator  
  printf("[Rank %d] Initializing global NCCL comm...\n", myRank);  
  NCCLCHECK(ncclCommInitRank(&global_comm, nRanks, id, myRank));  
    
  // 2. 使用ncclCommSplit创建TP communicator  
  // color=dp_rank表示同一个DP组的ranks会在同一个TP communicator中  
  // key=tp_rank用于在TP组内排序  
  printf("[Rank %d] Creating TP communicator (color=%d, key=%d)...\n",   
         myRank, dp_rank, tp_rank);  
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;  
  NCCLCHECK(ncclGroupStart());  
  NCCLCHECK(ncclCommSplit(global_comm, dp_rank, tp_rank, &tp_comm, &config));  
  NCCLCHECK(ncclGroupEnd());  
    
  // 3. 使用ncclCommSplit创建DP communicator  
  // color=tp_rank表示相同TP位置的ranks会在同一个DP communicator中  
  // key=dp_rank用于在DP组内排序  
  printf("[Rank %d] Creating DP communicator (color=%d, key=%d)...\n",   
         myRank, tp_rank, dp_rank);  
  NCCLCHECK(ncclGroupStart());  
  NCCLCHECK(ncclCommSplit(global_comm, tp_rank, dp_rank, &dp_comm, &config));  
  NCCLCHECK(ncclGroupEnd());  
    
  // 4. 在TP组内执行AllReduce  
  printf("[Rank %d] Performing TP AllReduce...\n", myRank);  
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff,   
                          size, ncclFloat, ncclSum, tp_comm, s));  
  CUDACHECK(cudaStreamSynchronize(s));  
    
  // 5. 在DP组间执行AllReduce  
  printf("[Rank %d] Performing DP AllReduce...\n", myRank);  
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff,   
                          size, ncclFloat, ncclSum, dp_comm, s));  
  CUDACHECK(cudaStreamSynchronize(s));  
    
  // 清理资源  
  CUDACHECK(cudaFree(sendbuff));  
  CUDACHECK(cudaFree(recvbuff));  
  ncclCommDestroy(tp_comm);  
  ncclCommDestroy(dp_comm);  
  ncclCommDestroy(global_comm);  
    
  MPICHECK(MPI_Finalize());  
    
  printf("[Rank %d] Success\n", myRank);  
  return 0;  
}
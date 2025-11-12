/*************************************************************************  
 * Copyright (c) 2025, NCCL-SHARP Project. All rights reserved.  
 *  
 * Single-process TP/DP AllReduce Test with NCCL  
 * TP Size = 8 (intra-node), DP Size = 2 (inter-node)  
 * Total: 16 GPUs (simulating 2 nodes × 8 GPUs)  
 ************************************************************************/  
  
 #include <stdio.h>  
 #include <stdlib.h>  
 #include <string.h>  
 #include "cuda_runtime.h"  
 #include "nccl.h"  
 #include "flow_extractor.h"  
   
 #define CUDACHECK(cmd) do {                         \  
   cudaError_t err = cmd;                            \  
   if (err != cudaSuccess) {                         \  
     printf("Failed: Cuda error %s:%d '%s'\n",       \  
         __FILE__,__LINE__,cudaGetErrorString(err)); \  
     exit(EXIT_FAILURE);                             \  
   }                                                 \  
 } while(0)  
   
 #define NCCLCHECK(cmd) do {                         \  
   ncclResult_t res = cmd;                           \  
   if (res != ncclSuccess) {                         \  
     printf("Failed, NCCL error %s:%d '%s'\n",       \  
         __FILE__,__LINE__,ncclGetErrorString(res)); \  
     exit(EXIT_FAILURE);                             \  
   }                                                 \  
 } while(0)  
   
 int main(int argc, char* argv[])  
 {  
     // 配置参数  
     const int nDev = 16;           // 总GPU数量  
     const int TP_SIZE = 8;         // TP组大小(节点内)  
     const int DP_SIZE = 2;         // DP组大小(跨节点)  
     const int tensor_elements = 1024 * 1024;  // 张量元素数量  
       
     // 设备列表  
     int devs[16];  
     for (int i = 0; i < nDev; i++) {  
         devs[i] = i;  
     }  
       
     // 通信器数组  
     ncclComm_t globalComms[16];    // 全局通信器  
     ncclComm_t tpComms[16];        // TP通信器(节点内)  
     ncclComm_t dpComms[16];        // DP通信器(跨节点)  
       
     // 数据缓冲区  
     float** sendbuff = (float**)malloc(nDev * sizeof(float*));  
     float** recvbuff = (float**)malloc(nDev * sizeof(float*));  
     cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nDev);  
       
     printf("========================================\n");  
     printf("Single-process NCCL TP/DP Test\n");  
     printf("========================================\n");  
     printf("Configuration:\n");  
     printf("  Total GPUs: %d\n", nDev);  
     printf("  TP Size: %d (intra-node)\n", TP_SIZE);  
     printf("  DP Size: %d (inter-node)\n", DP_SIZE);  
     printf("  Tensor elements: %d\n", tensor_elements);  
     printf("========================================\n\n");  
       
     // 步骤1: 初始化全局通信器  
     printf("[Step 1] Initializing global communicator...\n");  
     NCCLCHECK(ncclCommInitAll(globalComms, nDev, devs));  
     printf("✓ Global communicator initialized\n\n");  
       
     // 步骤2: 创建TP组(节点内通信)  
     printf("[Step 2] Creating TP groups (intra-node)...\n");  
     NCCLCHECK(ncclGroupStart());  // 关键修复: 添加 group start  
     for (int i = 0; i < nDev; i++) {  
         int color = i / TP_SIZE;  // GPU 0-7为组0, GPU 8-15为组1  
         int key = i % TP_SIZE;    // 组内rank  
         NCCLCHECK(ncclCommSplit(globalComms[i], color, key, &tpComms[i], NULL));  
     }  
     NCCLCHECK(ncclGroupEnd());    // 关键修复: 添加 group end  
     // 打印信息移到 group 外面  
     for (int i = 0; i < nDev; i++) {  
         printf("  GPU %2d -> TP group %d, rank %d\n", i, i / TP_SIZE, i % TP_SIZE);  
     }  
     printf("✓ TP groups created\n\n");  
       
     // 步骤3: 创建DP组(跨节点通信)  
     printf("[Step 3] Creating DP groups (inter-node)...\n");  
     NCCLCHECK(ncclGroupStart());  // 关键修复: 添加 group start  
     for (int i = 0; i < nDev; i++) {  
         int color = i % TP_SIZE;  // 相同位置的GPU为一组  
         int key = i / TP_SIZE;    // 组内rank  
         NCCLCHECK(ncclCommSplit(globalComms[i], color, key, &dpComms[i], NULL));  
     }  
     NCCLCHECK(ncclGroupEnd());    // 关键修复: 添加 group end  
     // 打印信息移到 group 外面  
     for (int i = 0; i < nDev; i++) {  
         printf("  GPU %2d -> DP group %d, rank %d\n", i, i % TP_SIZE, i / TP_SIZE);  
     }  
     printf("✓ DP groups created\n\n");  
       
     // 步骤4: 分配缓冲区和流  
     printf("[Step 4] Allocating buffers and streams...\n");  
     for (int i = 0; i < nDev; i++) {  
         sendbuff[i] = (float*)malloc(tensor_elements * sizeof(float));  
         recvbuff[i] = (float*)malloc(tensor_elements * sizeof(float));  
           
         // 初始化发送缓冲区  
         for (int j = 0; j < tensor_elements; j++) {  
             sendbuff[i][j] = (float)i;  
         }  
     }  
     printf("✓ Buffers allocated\n\n");  
       
     // 步骤5: 执行TP AllReduce(节点内)  
     printf("[Step 5] Performing TP AllReduce (intra-node)...\n");  
     NCCLCHECK(ncclGroupStart());  
     for (int i = 0; i < nDev; i++) {  
         NCCLCHECK(ncclAllReduce((const void*)sendbuff[i],  
                                 (void*)recvbuff[i],  
                                 tensor_elements,  
                                 ncclFloat,  
                                 ncclSum,  
                                 tpComms[i],  
                                 NULL));  
     }  
     NCCLCHECK(ncclGroupEnd());  
     printf("✓ TP AllReduce completed\n\n");  
       
     // 步骤6: 执行DP AllReduce(跨节点)  
     printf("[Step 6] Performing DP AllReduce (inter-node)...\n");  
     NCCLCHECK(ncclGroupStart());  
     for (int i = 0; i < nDev; i++) {  
         NCCLCHECK(ncclAllReduce((const void*)recvbuff[i],  
                                 (void*)sendbuff[i],  
                                 tensor_elements,  
                                 ncclFloat,  
                                 ncclSum,  
                                 dpComms[i],  
                                 NULL));  
     }  
     NCCLCHECK(ncclGroupEnd());  
     printf("✓ DP AllReduce completed\n\n");  
       
     // 步骤7: 写入流信息  
     printf("[Step 7] Writing flow information...\n");  
     NCCLCHECK(ncclWriteAggregatedFlow(tpComms[0]));  
     printf("✓ Flow records written for TP communicators\n");  
     NCCLCHECK(ncclWriteAggregatedFlow(dpComms[0]));  
     printf("✓ Flow records written for DP communicators\n\n");  
       
     // 步骤8: 清理资源  
     printf("[Step 8] Cleaning up resources...\n");  
     for (int i = 0; i < nDev; i++) {  
         ncclCommDestroy(tpComms[i]);  
         ncclCommDestroy(dpComms[i]);  
         ncclCommDestroy(globalComms[i]);  
         free(sendbuff[i]);  
         free(recvbuff[i]);  
     }  
     free(sendbuff);  
     free(recvbuff);  
     free(streams);  
     printf("✓ Cleanup complete\n\n");  
       
     printf("========================================\n");  
     printf("TP/DP Test completed successfully!\n");  
     printf("========================================\n");  
       
     return 0;  
 }
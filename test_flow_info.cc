#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "nccl.h"
#include "src/include/flow_info.h"

int main(int argc, char* argv[]) {
    printf("=== NCCL流信息提取工具测试 ===\n");
    
    // 解析命令行参数
    int nRanks = 4;           // 默认4个节点
    size_t dataSize = 1024;   // 默认1KB数据
    ncclFunc_t collType = ncclFuncAllReduce;  // 默认AllReduce
    
    if (argc > 1) nRanks = atoi(argv[1]);
    if (argc > 2) dataSize = atoi(argv[2]);
    if (argc > 3) {
        if (strcmp(argv[3], "allgather") == 0) collType = ncclFuncAllGather;
        else if (strcmp(argv[3], "broadcast") == 0) collType = ncclFuncBroadcast;
        else if (strcmp(argv[3], "reduce") == 0) collType = ncclFuncReduce;
    }
    
    printf("测试参数:\n");
    printf("  节点数: %d\n", nRanks);
    printf("  数据大小: %zu bytes\n", dataSize);
    printf("  集合通信类型: %d\n", collType);
    printf("\n");
    
    // 启用流信息收集
    ncclFlowCollector::getInstance()->enable();
    printf("流信息收集已启用\n\n");
    
    // 初始化NCCL
    ncclComm_t comm;
    ncclUniqueId id;
    
    // 模拟NCCL初始化过程
    if (ncclGetUniqueId(&id) != ncclSuccess) {
        printf("错误: 无法获取NCCL唯一ID\n");
        return -1;
    }
    
    // 模拟多进程环境中的单个进程
    int rank = 0;  // 假设当前是rank 0
    if (ncclCommInitRank(&comm, nRanks, id, rank) != ncclSuccess) {
        printf("错误: NCCL通信器初始化失败\n");
        return -1;
    }
    
    printf("NCCL通信器初始化成功\n");
    
    // 分配测试数据
    float *sendbuf, *recvbuf;
    size_t count = dataSize / sizeof(float);
    
    cudaMalloc(&sendbuf, dataSize);
    cudaMalloc(&recvbuf, dataSize);
    
    if (!sendbuf || !recvbuf) {
        printf("错误: 内存分配失败\n");
        return -1;
    }
    
    printf("测试数据分配成功，元素数量: %zu\n", count);
    
    // 执行集合通信操作（这里会触发流信息提取）
    printf("\n开始执行集合通信操作...\n");
    
    ncclResult_t result;
    switch (collType) {
        case ncclFuncAllReduce:
            result = ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, 0);
            break;
        case ncclFuncAllGather:
            result = ncclAllGather(sendbuf, recvbuf, count, ncclFloat, comm, 0);
            break;
        case ncclFuncBroadcast:
            result = ncclBroadcast(sendbuf, recvbuf, count, ncclFloat, 0, comm, 0);
            break;
        case ncclFuncReduce:
            result = ncclReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, 0, comm, 0);
            break;
        default:
            printf("错误: 不支持的集合通信类型\n");
            return -1;
    }
    
    if (result != ncclSuccess) {
        printf("错误: 集合通信操作失败，错误码: %d\n", result);
        return -1;
    }
    
    printf("集合通信操作完成\n\n");
    
    // 输出收集到的流信息
    printf("=== 流信息提取结果 ===\n");
    ncclFlowCollector::getInstance()->printFlowInfo();
    
    // 保存流信息到日志文件
    char logFileName[256];
    snprintf(logFileName, sizeof(logFileName), "nccl_flow_log_rank%d.txt", rank);
    ncclFlowCollector::getInstance()->saveToFile(logFileName);
    printf("\n流信息已保存到文件: %s\n", logFileName);
    
    // 清理资源
    cudaFree(sendbuf);
    cudaFree(recvbuf);
    ncclCommDestroy(comm);
    
    printf("\n测试完成！\n");
    return 0;
}
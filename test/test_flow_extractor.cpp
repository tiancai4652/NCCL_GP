/*************************************************************************
 * Copyright (c) 2025, NCCL-SHARP Project. All rights reserved.
 *
 * Test program for NCCL Flow Extractor
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
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

void print_flow_summary(struct ncclCollectiveFlow* flow) {
    printf("\n=== Flow Information Summary ===\n");
    printf("Collective Type: %d\n", (int)flow->collType);
    printf("Algorithm: %s\n", ncclAlgorithmToString(flow->algorithm));
    printf("Protocol: %s\n", ncclProtocolToString(flow->protocol));
    printf("Pattern: %s\n", ncclPatternToString(flow->pattern));
    printf("My Rank: %d / %d\n", flow->myRank, flow->nRanks);
    printf("Channels: %d\n", flow->nChannels);
    printf("Total Steps: %d\n", flow->totalSteps);
    printf("Total Bytes: %zu\n", flow->totalBytes);
    printf("Estimated Time: %.2f μs\n", flow->totalTime);
    printf("Topology: %s\n", flow->topoInfo);
    
    printf("\n=== Channel Details ===\n");
    for (int c = 0; c < flow->nChannels; c++) {
        struct ncclChannelFlow* channel = &flow->channels[c];
        printf("Channel %d: %d steps\n", channel->channelId, channel->nSteps);
        
        for (int s = 0; s < channel->nSteps && s < 5; s++) { // 只显示前5步
            struct ncclFlowStep* step = &channel->steps[s];
            printf("  Step %d: %s src=%d dst=%d size=%zu desc='%s'\n",
                   step->stepId, 
                   ncclFlowOpTypeToString(step->opType),
                   step->srcRank,
                   step->dstRank,
                   step->dataSize,
                   step->description);
        }
        if (channel->nSteps > 5) {
            printf("  ... (%d more steps)\n", channel->nSteps - 5);
        }
    }
    printf("\n");
}

void test_collective_flow(ncclComm_t comm, ncclFunc_t collType, const char* testName) {
    printf("\n>>> Testing %s <<<\n", testName);
    
    struct ncclCollectiveFlow* flow = NULL;
    size_t count = 1024;
    ncclDataType_t dataType = ncclFloat;
    int root = 0;
    
    // 获取流信息
    ncclResult_t result = ncclGetCollectiveFlow(collType, count, dataType, root, comm, &flow);
    
    if (result == ncclSuccess && flow != NULL) {
        print_flow_summary(flow);
        
        // 测试JSON输出
        char* jsonStr = NULL;
        TEST_NCCLCHECK(ncclFlowToJson(flow, &jsonStr));
        
        if (jsonStr) {
            printf("=== JSON Output (first 500 chars) ===\n");
            char preview[501];
            strncpy(preview, jsonStr, 500);
            preview[500] = '\0';
            printf("%s%s\n", preview, strlen(jsonStr) > 500 ? "..." : "");
            
            // 保存到文件
            char filename[256];
                         snprintf(filename, sizeof(filename), "flow_%s_rank%d.json", testName, flow->myRank);
            FILE* fp = fopen(filename, "w");
            if (fp) {
                fprintf(fp, "%s", jsonStr);
                fclose(fp);
                printf("Flow saved to: %s\n", filename);
            }
            
            free(jsonStr);
        }
        
        // 释放内存
        TEST_NCCLCHECK(ncclFreeCollectiveFlow(flow));
        printf("Test %s: PASSED\n", testName);
    } else {
        printf("Test %s: FAILED (result=%d)\n", testName, result);
    }
}

void test_helper_functions() {
    printf("\n>>> Testing Helper Functions <<<\n");
    
    // 测试字符串转换函数
    printf("Algorithm strings:\n");
    for (int i = 0; i < 6; i++) {
        printf("  %d -> %s\n", i, ncclAlgorithmToString(i));
    }
    
    printf("Protocol strings:\n");
    for (int i = 0; i < 3; i++) {
        printf("  %d -> %s\n", i, ncclProtocolToString(i));
    }
    
    printf("OpType strings:\n");
    for (int i = 0; i < 5; i++) {
        printf("  %d -> %s\n", i, ncclFlowOpTypeToString((ncclFlowOpType_t)i));
    }
    
    // 测试开关
    printf("Testing enable/disable:\n");
    TEST_NCCLCHECK(ncclSetFlowExtractionEnabled(0));
    printf("  Disabled flow extraction\n");
    TEST_NCCLCHECK(ncclSetFlowExtractionEnabled(1));
    printf("  Enabled flow extraction\n");
    
    printf("Helper functions: PASSED\n");
}

void test_error_handling(ncclComm_t comm) {
    printf("\n>>> Testing Error Handling <<<\n");
    
    struct ncclCollectiveFlow* flow = NULL;
    
    // 测试NULL参数
    ncclResult_t result = ncclGetCollectiveFlow(ncclFuncAllReduce, 1024, ncclFloat, 0, NULL, &flow);
    if (result != ncclSuccess) {
        printf("  NULL comm: PASSED (correctly rejected)\n");
    } else {
        printf("  NULL comm: FAILED (should have been rejected)\n");
    }
    
    result = ncclGetCollectiveFlow(ncclFuncAllReduce, 1024, ncclFloat, 0, comm, NULL);
    if (result != ncclSuccess) {
        printf("  NULL flow: PASSED (correctly rejected)\n");
    } else {
        printf("  NULL flow: FAILED (should have been rejected)\n");
    }
    
    // 测试禁用状态
    TEST_NCCLCHECK(ncclSetFlowExtractionEnabled(0));
    result = ncclGetCollectiveFlow(ncclFuncAllReduce, 1024, ncclFloat, 0, comm, &flow);
    if (result != ncclSuccess) {
        printf("  Disabled extraction: PASSED (correctly rejected)\n");
    } else {
        printf("  Disabled extraction: FAILED (should have been rejected)\n");
        if (flow) ncclFreeCollectiveFlow(flow);
    }
    TEST_NCCLCHECK(ncclSetFlowExtractionEnabled(1)); // 重新启用
    
    // 测试释放NULL指针
    result = ncclFreeCollectiveFlow(NULL);
    if (result == ncclSuccess) {
        printf("  Free NULL: PASSED (handled gracefully)\n");
    } else {
        printf("  Free NULL: FAILED (should handle gracefully)\n");
    }
    
    printf("Error handling: PASSED\n");
}

int main(int argc, char* argv[]) {
    // disable NCCL debug to avoid unresolved references in test link
    setenv("NCCL_DEBUG", "WARN", 1);
    printf("NCCL Flow Extractor Test Suite\n");
    printf("===============================\n");
    
    // 获取设备数量
    int nDev = 3;
    char* env = getenv("GPU_DEV_NUM");
    if (env) {
        nDev = atoi(env);
        printf("Using %d devices (from GPU_DEV_NUM)\n", nDev);
    } else {
        printf("Using %d devices (default)\n", nDev);
    }
    
    // 初始化通信器
    ncclComm_t* comms = (ncclComm_t*)malloc(nDev * sizeof(ncclComm_t));
    int* devs = (int*)malloc(nDev * sizeof(int));
    
    for (int i = 0; i < nDev; i++) {
        devs[i] = i;
    }
    
    printf("Initializing NCCL communicators...\n");
    TEST_NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
    printf("NCCL initialization complete\n");
    
    // 使用第一个通信器进行信息提取测试
    ncclComm_t comm = comms[0];
    
    // 运行测试
    test_helper_functions();
    test_error_handling(comm);
    
    // 测试不同的集合通信操作（静态提取）
    test_collective_flow(comm, ncclFuncAllReduce, "AllReduce");
    test_collective_flow(comm, ncclFuncAllGather, "AllGather");
    test_collective_flow(comm, ncclFuncBroadcast, "Broadcast");
    test_collective_flow(comm, ncclFuncReduce, "Reduce");
    test_collective_flow(comm, ncclFuncReduceScatter, "ReduceScatter");

    // 触发一次真实的 NCCL 调用以产生 proxyOp（使用 fake CUDA 不会进行真实传输）
    printf("\nTriggering a real NCCL AllReduce to generate proxy ops...\n");
    const size_t count = 1024;
    TEST_NCCLCHECK(ncclSetFlowExtractionEnabled(1));
    TEST_NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
        TEST_NCCLCHECK(ncclAllReduce((const void*)NULL, (void*)NULL, count, ncclFloat, ncclSum, comms[i], (cudaStream_t)0));
    }
    TEST_NCCLCHECK(ncclGroupEnd());
    // no cuda stream sync or free needed when using NULL/0
    
    printf("Real NCCL call done. Check proxy_flow_rank*.jsonl for records.\n");

    // 生成聚合输出
    TEST_NCCLCHECK(ncclWriteAggregatedFlow(comm));
    printf("Aggregated flow written: flow_rank%d.json\n", 0);
    
    // 清理
    printf("\nCleaning up...\n");
    for (int i = 0; i < nDev; i++) {
        if (comms[i]) {
            ncclCommDestroy(comms[i]);
        }
    }
    
    free(comms);
    free(devs);
    
    printf("\n=== All Tests Completed ===\n");
    printf("Check the generated JSON files for detailed flow information.\n");
    
    return 0;
} 
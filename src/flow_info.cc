/*************************************************************************
 * Copyright (c) 2024, Flow Info Extraction Tool
 * 
 * NCCL集合通信流信息提取工具实现
 ************************************************************************/

#include "flow_info.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 静态实例
ncclFlowCollector* ncclFlowCollector::instance = nullptr;

// 获取单例实例
ncclFlowCollector* ncclFlowCollector::getInstance() {
    if (instance == nullptr) {
        instance = new ncclFlowCollector();
    }
    return instance;
}

// 构造函数
ncclFlowCollector::ncclFlowCollector() : enabled(false), logFile(nullptr) {
    memset(&currentFlow, 0, sizeof(currentFlow));
    currentFlow.rank = 0;
    currentFlow.nRanks = 1;
    currentFlow.totalBytes = 0;
    currentFlow.dataType = 0;
    currentFlow.totalSteps = 0;
    currentFlow.totalPhases = 0;
    currentFlow.estimatedTime = 0.0;
}

// 析构函数
ncclFlowCollector::~ncclFlowCollector() {
    closeLogFile();
}

// 初始化流信息收集
void ncclFlowCollector::initFlow(ncclComm* comm, int collective, size_t bytes, int dataType) {
    if (!enabled) return;
    
    // 重置流信息
    currentFlow.steps.clear();
    currentFlow.totalBytes = bytes;
    currentFlow.dataType = dataType;
    currentFlow.algInfo.collective = collective;
    currentFlow.totalSteps = 0;
    currentFlow.totalPhases = 0;
    currentFlow.estimatedTime = 0.0;
    
    // 模拟从comm获取信息（实际实现中会从真实的comm结构获取）
    currentFlow.rank = 0;  // 默认rank 0
    currentFlow.nRanks = 4; // 默认4个节点
    
    printf("[流信息] 初始化流信息收集 - 集合通信类型: %d, 数据量: %zu bytes\n", 
           collective, bytes);
}

// 设置算法信息
void ncclFlowCollector::setAlgorithmInfo(int algorithm, int protocol, int nChannels, int nThreads, 
                                        size_t chunkSize, float bandwidth, float latency, const char* reason) {
    if (!enabled) return;
    
    currentFlow.algInfo.algorithm = algorithm;
    currentFlow.algInfo.protocol = protocol;
    currentFlow.algInfo.nChannels = nChannels;
    currentFlow.algInfo.nThreads = nThreads;
    currentFlow.algInfo.chunkSize = chunkSize;
    currentFlow.algInfo.bandwidth = bandwidth;
    currentFlow.algInfo.latency = latency;
    
    if (reason) {
        strncpy(currentFlow.algInfo.reason, reason, sizeof(currentFlow.algInfo.reason) - 1);
        currentFlow.algInfo.reason[sizeof(currentFlow.algInfo.reason) - 1] = '\0';
    }
    
    printf("[流信息] 算法选择 - 算法: %d, 协议: %d, 通道数: %d, 线程数: %d\n", 
           algorithm, protocol, nChannels, nThreads);
    printf("[流信息] 选择原因: %s\n", reason ? reason : "未指定");
}

// 添加流步骤
void ncclFlowCollector::addFlowStep(ncclFlowStepType type, int srcRank, int dstRank, 
                                   size_t dataSize, int channel, int phase, const char* description) {
    if (!enabled) return;
    
    ncclFlowStep step;
    step.stepId = currentFlow.steps.size();
    step.type = type;
    step.srcRank = srcRank;
    step.dstRank = dstRank;
    step.dataSize = dataSize;
    step.channel = channel;
    step.phase = phase;
    
    if (description) {
        strncpy(step.description, description, sizeof(step.description) - 1);
        step.description[sizeof(step.description) - 1] = '\0';
    } else {
        snprintf(step.description, sizeof(step.description), 
                 "步骤%d: 类型%d, 通道%d", step.stepId, type, channel);
    }
    
    currentFlow.steps.push_back(step);
    currentFlow.totalSteps++;
    
    if (phase > currentFlow.totalPhases) {
        currentFlow.totalPhases = phase;
    }
}

// 简化的添加步骤接口
void ncclFlowCollector::addStep(int channel, int stepId, const char* description) {
    if (!enabled) return;
    
    addFlowStep(FLOW_STEP_SEND, currentFlow.rank, (currentFlow.rank + 1) % currentFlow.nRanks,
                currentFlow.totalBytes / currentFlow.algInfo.nChannels, channel, 0, description);
}

// 完成流信息收集
void ncclFlowCollector::finalizeFlow() {
    if (!enabled) return;
    
    // 计算预估执行时间
    if (currentFlow.algInfo.bandwidth > 0) {
        double transferTime = (double)currentFlow.totalBytes / (currentFlow.algInfo.bandwidth * 1e9) * 1000; // ms
        currentFlow.estimatedTime = transferTime + currentFlow.algInfo.latency / 1000.0; // ms
    }
    
    printf("[流信息] 流信息收集完成 - 总步骤数: %d, 预估时间: %.2f ms\n", 
           currentFlow.totalSteps, currentFlow.estimatedTime);
}

// 设置日志文件
void ncclFlowCollector::setLogFile(const char* filename) {
    closeLogFile();
    if (filename) {
        logFile = fopen(filename, "w");
        if (!logFile) {
            printf("[流信息] 警告: 无法打开日志文件 %s\n", filename);
        }
    }
}

// 关闭日志文件
void ncclFlowCollector::closeLogFile() {
    if (logFile) {
        fclose(logFile);
        logFile = nullptr;
    }
}

// 输出流信息到控制台
void ncclFlowCollector::printFlowInfo() {
    if (!enabled) {
        printf("[流信息] 流信息收集未启用\n");
        return;
    }
    
    printf("\n=== NCCL流信息提取结果 ===\n");
    printf("基本信息:\n");
    printf("  当前节点: %d/%d\n", currentFlow.rank, currentFlow.nRanks);
    printf("  数据总量: %zu bytes\n", currentFlow.totalBytes);
    printf("  数据类型: %d\n", currentFlow.dataType);
    
    printf("\n算法选择信息:\n");
    printf("  集合通信类型: %d\n", currentFlow.algInfo.collective);
    printf("  选择算法: %d\n", currentFlow.algInfo.algorithm);
    printf("  选择协议: %d\n", currentFlow.algInfo.protocol);
    printf("  通道数: %d\n", currentFlow.algInfo.nChannels);
    printf("  线程数: %d\n", currentFlow.algInfo.nThreads);
    printf("  块大小: %zu bytes\n", currentFlow.algInfo.chunkSize);
    printf("  预期带宽: %.2f GB/s\n", currentFlow.algInfo.bandwidth);
    printf("  预期延迟: %.2f us\n", currentFlow.algInfo.latency);
    printf("  选择原因: %s\n", currentFlow.algInfo.reason);
    
    printf("\n流执行步骤:\n");
    if (currentFlow.steps.empty()) {
        printf("  无流步骤记录\n");
    } else {
        for (size_t i = 0; i < currentFlow.steps.size(); i++) {
            const ncclFlowStep& step = currentFlow.steps[i];
            printf("  步骤%d: [通道%d] %s\n", step.stepId, step.channel, step.description);
            if (step.srcRank >= 0 && step.dstRank >= 0) {
                printf("         从节点%d到节点%d, 数据量: %zu bytes\n", 
                       step.srcRank, step.dstRank, step.dataSize);
            }
        }
    }
    
    printf("\n统计信息:\n");
    printf("  总步骤数: %d\n", currentFlow.totalSteps);
    printf("  总阶段数: %d\n", currentFlow.totalPhases);
    printf("  预估执行时间: %.2f ms\n", currentFlow.estimatedTime);
    printf("========================\n\n");
}

// 保存流信息到文件
void ncclFlowCollector::saveToFile(const char* filename) {
    if (!enabled) return;
    
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("[流信息] 错误: 无法创建日志文件 %s\n", filename);
        return;
    }
    
    // 写入时间戳
    time_t now = time(0);
    char* timeStr = ctime(&now);
    fprintf(file, "# NCCL流信息提取日志\n");
    fprintf(file, "# 生成时间: %s\n", timeStr);
    
    // 写入基本信息
    fprintf(file, "\n[基本信息]\n");
    fprintf(file, "当前节点=%d\n", currentFlow.rank);
    fprintf(file, "总节点数=%d\n", currentFlow.nRanks);
    fprintf(file, "数据总量=%zu\n", currentFlow.totalBytes);
    fprintf(file, "数据类型=%d\n", currentFlow.dataType);
    
    // 写入算法信息
    fprintf(file, "\n[算法选择]\n");
    fprintf(file, "集合通信类型=%d\n", currentFlow.algInfo.collective);
    fprintf(file, "算法=%d\n", currentFlow.algInfo.algorithm);
    fprintf(file, "协议=%d\n", currentFlow.algInfo.protocol);
    fprintf(file, "通道数=%d\n", currentFlow.algInfo.nChannels);
    fprintf(file, "线程数=%d\n", currentFlow.algInfo.nThreads);
    fprintf(file, "块大小=%zu\n", currentFlow.algInfo.chunkSize);
    fprintf(file, "预期带宽=%.2f\n", currentFlow.algInfo.bandwidth);
    fprintf(file, "预期延迟=%.2f\n", currentFlow.algInfo.latency);
    fprintf(file, "选择原因=%s\n", currentFlow.algInfo.reason);
    
    // 写入流步骤
    fprintf(file, "\n[流步骤]\n");
    for (size_t i = 0; i < currentFlow.steps.size(); i++) {
        const ncclFlowStep& step = currentFlow.steps[i];
        fprintf(file, "步骤%d,类型=%d,源节点=%d,目标节点=%d,数据量=%zu,通道=%d,阶段=%d,描述=%s\n",
                step.stepId, step.type, step.srcRank, step.dstRank, 
                step.dataSize, step.channel, step.phase, step.description);
    }
    
    // 写入统计信息
    fprintf(file, "\n[统计信息]\n");
    fprintf(file, "总步骤数=%d\n", currentFlow.totalSteps);
    fprintf(file, "总阶段数=%d\n", currentFlow.totalPhases);
    fprintf(file, "预估执行时间=%.2f\n", currentFlow.estimatedTime);
    
    fclose(file);
    printf("[流信息] 流信息已保存到文件: %s\n", filename);
}
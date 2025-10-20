/*************************************************************************
 * Copyright (c) 2025, NCCL-SHARP Project. All rights reserved.
 *
 * Flow Extractor for NCCL Collective Communication Operations
 ************************************************************************/

#ifndef NCCL_FLOW_EXTRACTOR_H_
#define NCCL_FLOW_EXTRACTOR_H_

#include "nccl.h"
#include "include/info.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NCCL_EXPORT
#define NCCL_EXPORT __attribute__((visibility("default")))
#endif

// 流操作类型
typedef enum {
    NCCL_FLOW_OP_SEND = 0,
    NCCL_FLOW_OP_RECV = 1,
    NCCL_FLOW_OP_REDUCE = 2,
    NCCL_FLOW_OP_BROADCAST = 3,
    NCCL_FLOW_OP_WAIT = 4
} ncclFlowOpType_t;

// 单个通信步骤的详细信息
struct ncclFlowStep {
    int stepId;                    // 步骤ID (全局序号)
    ncclFlowOpType_t opType;       // 操作类型
    int srcRank;                   // 源节点rank (-1表示无效)
    int dstRank;                   // 目标节点rank (-1表示无效)  
    size_t dataSize;               // 数据量 (bytes)
    size_t dataOffset;             // 数据偏移量
    int channelId;                 // 通道ID
    int chunkId;                   // 数据块ID
    double estimatedTime;          // 预计耗时 (microseconds)
    char description[64];          // 步骤描述
};

// 通道级别的流信息
struct ncclChannelFlow {
    int channelId;                 // 通道ID
    int nSteps;                    // 该通道的步骤数
    struct ncclFlowStep* steps;    // 步骤数组
};

// 完整的集合通信流信息
struct ncclCollectiveFlow {
    // 基本信息
    ncclFunc_t collType;           // 集合通信类型
    int algorithm;                 // 选择的算法 (NCCL_ALGO_*)
    int protocol;                  // 选择的协议 (NCCL_PROTO_*)
    ncclPattern_t pattern;         // 通信模式
    
    // 拓扑信息
    int myRank;                    // 当前节点rank
    int nRanks;                    // 总节点数
    int nChannels;                 // 通道数
    char topoInfo[256];            // 拓扑信息摘要
    
    // 流信息
    int totalSteps;                // 总步数 (所有通道)
    struct ncclChannelFlow* channels; // 各通道流信息
    
    // 性能信息
    size_t totalBytes;             // 总数据量
    double totalTime;              // 总预计时间
    int nLoops;                    // 循环次数
    int chunkSize;                 // 块大小
};

// 主要API函数

NCCL_EXPORT ncclResult_t ncclSetFlowExtractionEnabled(int enable);
  
  // 获取算法和协议的字符串名称 (辅助函数)
NCCL_EXPORT const char* ncclAlgorithmToString(int algorithm);
NCCL_EXPORT const char* ncclProtocolToString(int protocol);
NCCL_EXPORT const char* ncclPatternToString(ncclPattern_t pattern);
NCCL_EXPORT const char* ncclFlowOpTypeToString(ncclFlowOpType_t opType);


// 记录：从NCCL的proxyOp与info直接生成并落盘流信息（仅当启用时生效）
NCCL_EXPORT ncclResult_t ncclRecordProxyOp(const struct ncclInfo* info,
                                           const struct ncclProxyOp* proxyOp,
                                           struct ncclComm* comm);

// 聚合：将 flow_steps_rank<rank>.jsonl 与 proxy_flow_rank<rank>.jsonl 聚合输出 flow_rank<rank>.json
NCCL_EXPORT ncclResult_t ncclWriteAggregatedFlow(struct ncclComm* comm);

// 逐步记录：按给定 peer 和方向（type: 0=RECV,1=SEND）输出每步一条
NCCL_EXPORT ncclResult_t ncclRecordProxyPeerSteps(struct ncclComm* comm,
                                                 int channelId,
                                                 int type,
                                                 int peer,
                                                 const struct ncclProxyOp* op);

// 权威提取：直接调用真实NCCL集合通信以触发proxy记录，然后写出聚合文件
// 不复制选择/规划逻辑，最大限度复用NCCL内部实现
NCCL_EXPORT ncclResult_t ncclExtractFlow(
    ncclFunc_t collType,
    size_t count,
    ncclDataType_t dataType,
    int root,
    ncclComm_t comm);

#ifdef __cplusplus
}
#endif

#endif // NCCL_FLOW_EXTRACTOR_H_ 
/*************************************************************************
 * Copyright (c) 2025, NCCL-SHARP Project. All rights reserved.
 *
 * Flow Extractor Implementation for NCCL Collective Communication Operations
 ************************************************************************/

#include "flow_extractor.h"
#include "include/debug.h"
#include "include/utils.h"
#include "include/comm.h"
#include "include/devcomm.h"
#include "include/graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 全局开关：是否启用流信息提取
static int flowExtractionEnabled = 1;

// 算法名称映射
static const char* algorithmNames[] = {
    "TREE",
    "RING", 
    "COLLNET_DIRECT",
    "COLLNET_CHAIN",
    "NVLS",
    "NVLS_TREE"
};

// 协议名称映射
static const char* protocolNames[] = {
    "LL",
    "LL128", 
    "SIMPLE"
};

// 模式名称映射
static const char* patternNames[] = {
    "RING",
    "RING_TWICE",
    "PIPELINE_FROM",
    "PIPELINE_TO", 
    "TREE_UP",
    "TREE_DOWN",
    "TREE_UP_DOWN",
    "COLLNET_CHAIN",
    "COLLNET_DIRECT",
    "NVLS",
    "NVLS_TREE",
    "SEND",
    "RECV"
};

// 操作类型名称映射
static const char* opTypeNames[] = {
    "SEND",
    "RECV",
    "REDUCE",
    "BROADCAST",
    "WAIT"
};

// 辅助函数实现
const char* ncclAlgorithmToString(int algorithm) {
    if (algorithm >= 0 && algorithm < NCCL_NUM_ALGORITHMS) {
        return algorithmNames[algorithm];
    }
    return "UNKNOWN";
}

const char* ncclProtocolToString(int protocol) {
    if (protocol >= 0 && protocol < NCCL_NUM_PROTOCOLS) {
        return protocolNames[protocol];
    }
    return "UNKNOWN";
}

const char* ncclPatternToString(ncclPattern_t pattern) {
    if (pattern >= 0 && pattern < sizeof(patternNames)/sizeof(patternNames[0])) {
        return patternNames[pattern];
    }
    return "UNKNOWN";
}

const char* ncclFlowOpTypeToString(ncclFlowOpType_t opType) {
    if (opType >= 0 && opType < sizeof(opTypeNames)/sizeof(opTypeNames[0])) {
        return opTypeNames[opType];
    }
    return "UNKNOWN";
}

// 内部：获取 CollNet 支持类型（等价于 enqueue.cc:getCollNetSupport）
static inline ncclResult_t flowGetCollNetSupport(struct ncclInfo* info, int* collNetTypeSupport) {
    ncclRedOp_t netOp = info->op == ncclAvg || info->op >= ncclNumOps ? ncclSum : info->op;
    *collNetTypeSupport = info->comm->collNetSupportMatrix[netOp][info->datatype];
    return ncclSuccess;
}

// 内部：算法与协议选择（等价于 enqueue.cc:getAlgoInfo 的简化复用版）
static ncclResult_t flowGetAlgoInfo(struct ncclInfo* info, int collNetTypeSupport, int numPipeOps) {
    struct ncclComm* comm = info->comm;
    if (comm->nRanks == 1) {
        info->algorithm = NCCL_ALGO_RING;
        info->protocol = NCCL_PROTO_SIMPLE;
    } else {
        float minTime = 3600000000.0f;
        info->algorithm = -1;
        info->protocol = -1;
        for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
            if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetTypeSupport != 1) continue;
            if (a == NCCL_ALGO_NVLS && comm->nNodes > 1) continue;
            // note: NVLS support macro在原始代码中检查datatype/op，这里简化忽略
            for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
                float time = -1.0f;
                ncclResult_t rc = ncclTopoGetAlgoTime(info, a, p, numPipeOps, &time);
                if (rc != ncclSuccess) continue;
                if (time >= 0 && time < minTime) {
                    info->algorithm = a;
                    info->protocol = p;
                    minTime = time;
                }
            }
        }
        if (info->algorithm == -1 || info->protocol == -1) {
            WARN("Error : no algorithm/protocol available");
            return ncclInternalError;
        }
        TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", info->nBytes, info->algorithm, info->protocol, (double)minTime);
    }
    // 选择线程数
    info->nThreads = comm->maxThreads[info->algorithm][info->protocol];
    if (info->nThreads <= 0) info->nThreads = 256;
    // 通道数
    if (info->nChannels <= 0) info->nChannels = comm->nChannels;
    return ncclSuccess;
}

// 启用/禁用流信息提取
extern "C" ncclResult_t ncclSetFlowExtractionEnabled(int enable) {
    flowExtractionEnabled = enable;
    INFO(NCCL_INIT, "Flow extraction %s", enable ? "enabled" : "disabled");
    return ncclSuccess;
}

// 创建基础流信息结构
static ncclResult_t createBaseFlow(struct ncclInfo* info, struct ncclCollectiveFlow** flow) {
    struct ncclCollectiveFlow* f;
    NCCLCHECK(ncclCalloc(&f, 1));
    
    // 填充基本信息（移除估算时间）
    f->collType = info->coll;
    f->algorithm = info->algorithm;
    f->protocol = info->protocol;
    f->pattern = info->pattern;
    f->myRank = info->comm->rank;
    f->nRanks = info->comm->nRanks;
    f->nChannels = info->nChannels;
    f->totalBytes = info->nBytes;
    f->nLoops = 0; // 将在具体算法中计算
    f->chunkSize = 0; // 将在具体算法中计算
    
    // 生成拓扑信息摘要
    snprintf(f->topoInfo, sizeof(f->topoInfo), 
             "nRanks=%d nChannels=%d algo=%s proto=%s pattern=%s",
             f->nRanks, f->nChannels, 
             ncclAlgorithmToString(f->algorithm),
             ncclProtocolToString(f->protocol),
             ncclPatternToString(f->pattern));
    
    // 分配通道数组
    NCCLCHECK(ncclCalloc(&f->channels, f->nChannels));
    for (int c = 0; c < f->nChannels; c++) {
        f->channels[c].channelId = c;
        f->channels[c].nSteps = 0;
        f->channels[c].steps = NULL;
    }
    
    *flow = f;
    return ncclSuccess;
}

// Ring算法流生成
ncclResult_t ncclGenerateRingFlow(struct ncclInfo* info, struct ncclCollectiveFlow* flow) {
    struct ncclComm* comm = info->comm;
    int nRanks = comm->nRanks;
    int myRank = comm->rank;
    
    TRACE(NCCL_COLL, "Generating Ring flow for rank %d", myRank);
    
    // Ring算法的步骤数
    int stepsPerLoop = info->nstepsPerLoop;  // 对于Ring，通常是 nRanks-1
    int chunksPerLoop = info->nchunksPerLoop; // 对于Ring，通常是 nRanks
    
    // 计算每个通道的步骤
    for (int c = 0; c < flow->nChannels; c++) {
        struct ncclChannelFlow* channel = &flow->channels[c];
        struct ncclRing* ring = &comm->channels[c].ring;
        
        // 估算步骤数 (简化版本，实际可能更复杂)
        int nSteps = 0;
        
        if (info->pattern == ncclPatternRing) {
            // AllGather/ReduceScatter: nRanks-1 步
            nSteps = nRanks - 1;
        } else if (info->pattern == ncclPatternRingTwice) {
            // AllReduce: 2*(nRanks-1) 步 (reduce-scatter + allgather)
            nSteps = 2 * (nRanks - 1);  
        }
        
        channel->nSteps = nSteps;
        if (nSteps > 0) {
            NCCLCHECK(ncclCalloc(&channel->steps, nSteps));
            
            // 生成具体的通信步骤
            for (int s = 0; s < nSteps; s++) {
                struct ncclFlowStep* step = &channel->steps[s];
                step->stepId = s;
                step->channelId = c;
                step->chunkId = s % chunksPerLoop;
                
                if (info->pattern == ncclPatternRingTwice) {
                    if (s < nRanks - 1) {
                        // Reduce-scatter 阶段
                        step->opType = NCCL_FLOW_OP_SEND;
                        step->dstRank = ring->next;
                        step->srcRank = myRank;
                        snprintf(step->description, sizeof(step->description), 
                                "ReduceScatter step %d to rank %d", s, ring->next);
                    } else {
                        // AllGather 阶段  
                        step->opType = NCCL_FLOW_OP_RECV;
                        step->srcRank = ring->prev;
                        step->dstRank = myRank;
                        snprintf(step->description, sizeof(step->description),
                                "AllGather step %d from rank %d", s - (nRanks - 1), ring->prev);
                    }
                } else {
                    // 简单的Ring模式
                    if (s % 2 == 0) {
                        step->opType = NCCL_FLOW_OP_SEND;
                        step->dstRank = ring->next;
                        step->srcRank = myRank;
                    } else {
                        step->opType = NCCL_FLOW_OP_RECV;
                        step->srcRank = ring->prev;  
                        step->dstRank = myRank;
                    }
                    snprintf(step->description, sizeof(step->description),
                            "Ring step %d", s);
                }
                
                // 估算数据大小 (简化)
                step->dataSize = info->nBytes / (nRanks * flow->nChannels);
                step->dataOffset = s * step->dataSize;
                step->estimatedTime = 100.0; // 简化的时间估算
            }
        }
        
        flow->totalSteps += nSteps;
    }
    
    return ncclSuccess;
}

// Tree算法流生成
ncclResult_t ncclGenerateTreeFlow(struct ncclInfo* info, struct ncclCollectiveFlow* flow) {
    struct ncclComm* comm = info->comm;
    int myRank = comm->rank;
    
    TRACE(NCCL_COLL, "Generating Tree flow for rank %d", myRank);
    
    // Tree算法的步骤相对简单
    for (int c = 0; c < flow->nChannels; c++) {
        struct ncclChannelFlow* channel = &flow->channels[c];
        struct ncclTree* tree = &comm->channels[c].tree;
        
        int nSteps = 0;
        
        // 根据模式确定步骤数
        if (info->pattern == ncclPatternTreeUp) {
            nSteps = tree->depth; // 向上聚合
        } else if (info->pattern == ncclPatternTreeDown) {
            nSteps = tree->depth; // 向下广播
        } else if (info->pattern == ncclPatternTreeUpDown) {
            nSteps = 2 * tree->depth; // 先上后下
        }
        
        channel->nSteps = nSteps;
        if (nSteps > 0) {
            NCCLCHECK(ncclCalloc(&channel->steps, nSteps));
            
            for (int s = 0; s < nSteps; s++) {
                struct ncclFlowStep* step = &channel->steps[s];
                step->stepId = s;
                step->channelId = c;
                step->chunkId = 0; // Tree通常不分块
                
                if (info->pattern == ncclPatternTreeUpDown && s >= tree->depth) {
                    // Down阶段
                    step->opType = NCCL_FLOW_OP_SEND;
                    step->srcRank = myRank;
                    // 简化：发给第一个down节点
                    step->dstRank = (tree->down[0] != -1) ? tree->down[0] : myRank;
                    snprintf(step->description, sizeof(step->description),
                            "Tree down step %d", s - tree->depth);
                } else {
                    // Up阶段
                    step->opType = NCCL_FLOW_OP_RECV;
                    step->dstRank = myRank;
                    step->srcRank = (tree->up != -1) ? tree->up : myRank;
                    snprintf(step->description, sizeof(step->description),
                            "Tree up step %d", s);
                }
                
                step->dataSize = info->nBytes / flow->nChannels;
                step->dataOffset = 0;
                step->estimatedTime = 200.0; // Tree一般比Ring慢一些
            }
        }
        
        flow->totalSteps += nSteps;
    }
    
    return ncclSuccess;
}

// CollNet算法流生成 (简化版本)
ncclResult_t ncclGenerateCollNetFlow(struct ncclInfo* info, struct ncclCollectiveFlow* flow) {
    TRACE(NCCL_COLL, "Generating CollNet flow (simplified)");
    
    // CollNet算法相对复杂，这里提供简化实现
    for (int c = 0; c < flow->nChannels; c++) {
        struct ncclChannelFlow* channel = &flow->channels[c];
        
        // CollNet通常步骤较少但吞吐更高
        int nSteps = 2; // 简化：网络卡聚合 + 广播
        channel->nSteps = nSteps;
        NCCLCHECK(ncclCalloc(&channel->steps, nSteps));
        
        for (int s = 0; s < nSteps; s++) {
            struct ncclFlowStep* step = &channel->steps[s];
            step->stepId = s;
            step->channelId = c;
            step->chunkId = s;
            
            if (s == 0) {
                step->opType = NCCL_FLOW_OP_SEND;
                snprintf(step->description, sizeof(step->description), "CollNet aggregate");
            } else {
                step->opType = NCCL_FLOW_OP_RECV;
                snprintf(step->description, sizeof(step->description), "CollNet broadcast");
            }
            
            step->dataSize = info->nBytes / flow->nChannels;
            step->dataOffset = s * step->dataSize;
            step->estimatedTime = 50.0; // CollNet通常更快
            step->srcRank = (s == 0) ? info->comm->rank : -1;
            step->dstRank = (s == 1) ? info->comm->rank : -1;
        }
        
        flow->totalSteps += nSteps;
    }
    
    return ncclSuccess;
}

// 基于ncclInfo生成流信息
extern "C" ncclResult_t ncclComputeFlowFromInfo(struct ncclInfo* info, struct ncclCollectiveFlow** flow) {
    if (!flowExtractionEnabled) {
        return ncclInternalError;
    }
    
    NCCLCHECK(createBaseFlow(info, flow));
    
    // 根据算法类型生成具体的流步骤
    switch (info->algorithm) {
        case NCCL_ALGO_RING:
            NCCLCHECK(ncclGenerateRingFlow(info, *flow));
            break;
        case NCCL_ALGO_TREE:
            NCCLCHECK(ncclGenerateTreeFlow(info, *flow));
            break;
        case NCCL_ALGO_COLLNET_DIRECT:
        case NCCL_ALGO_COLLNET_CHAIN:
            NCCLCHECK(ncclGenerateCollNetFlow(info, *flow));
            break;
        case NCCL_ALGO_NVLS:
        case NCCL_ALGO_NVLS_TREE:
            // NVLS 暂时使用CollNet的简化版本
            NCCLCHECK(ncclGenerateCollNetFlow(info, *flow));
            break;
        default:
            WARN("Unsupported algorithm %d for flow extraction", info->algorithm);
            return ncclInternalError;
    }
    
    INFO(NCCL_COLL, "Generated flow: %d total steps", (*flow)->totalSteps);
    
    return ncclSuccess;
}

// 主API函数：获取集合通信流信息
extern "C" ncclResult_t ncclGetCollectiveFlow(
    ncclFunc_t collType,
    size_t count,
    ncclDataType_t dataType,
    int root,
    ncclComm_t comm,
    struct ncclCollectiveFlow** flow) {
    
    if (!flowExtractionEnabled) {
        WARN("Flow extraction is disabled");
        return ncclInvalidArgument;
    }
    if (flow == NULL || comm == NULL) {
        WARN("Invalid arguments to ncclGetCollectiveFlow");
        return ncclInvalidArgument;
    }
    
    // 构建ncclInfo结构
    struct ncclInfo info = {};
    info.coll = collType;
    info.comm = comm;
    info.count = count;
    info.datatype = dataType;
    info.root = root;
    info.sendbuff = NULL; // 流信息提取不需要实际buffer
    info.recvbuff = NULL;
    info.stream = NULL;
    info.chunkSteps = 1;
    info.sliceSteps = 1;
    
    // 设置派生信息
    NCCLCHECK(ncclInfoSetDerived(&info, comm->nRanks));
    
    // 复用NCCL的算法选择逻辑
    int collNetSupport = 0;
    (void)flowGetCollNetSupport(&info, &collNetSupport);
    NCCLCHECK(flowGetAlgoInfo(&info, collNetSupport, 1));
    
    // 设置通信模式（复制 enqueue.cc:getPatternInfo 逻辑的结果）
    switch (collType) {
        case ncclFuncBroadcast:
            info.pattern = info.algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom; break;
        case ncclFuncReduce:
            info.pattern = info.algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo; break;
        case ncclFuncReduceScatter:
        case ncclFuncAllGather:
            info.pattern = (info.algorithm == NCCL_ALGO_NVLS) ? ncclPatternNvls : ncclPatternRing; break;
        case ncclFuncAllReduce:
            info.pattern = (info.algorithm == NCCL_ALGO_NVLS) ? ncclPatternNvls :
                           (info.algorithm == NCCL_ALGO_NVLS_TREE) ? ncclPatternNvlsTree :
                           (info.algorithm == NCCL_ALGO_COLLNET_DIRECT) ? ncclPatternCollnetDirect :
                           (info.algorithm == NCCL_ALGO_COLLNET_CHAIN) ? ncclPatternCollnetChain :
                           (info.algorithm == NCCL_ALGO_TREE) ? ncclPatternTreeUpDown :
                           ncclPatternRingTwice; break;
        default:
            WARN("Unsupported collective type %d", collType);
            return ncclInvalidArgument;
    }
    
    // 设置循环信息（复制 enqueue.cc:getLoopInfo 结果）
    switch (info.pattern) {
        case ncclPatternTreeUp:
        case ncclPatternTreeDown:
        case ncclPatternTreeUpDown:
        case ncclPatternPipelineFrom:
        case ncclPatternPipelineTo:
        case ncclPatternCollnetChain:
            info.nstepsPerLoop = info.nchunksPerLoop = 1; break;
        case ncclPatternNvls:
            info.nstepsPerLoop = 1; info.nchunksPerLoop = info.comm->channels[0].nvls.nHeads; break;
        case ncclPatternCollnetDirect:
            info.nstepsPerLoop = 1; info.nchunksPerLoop = info.comm->channels[0].collnetDirect.nHeads; break;
        case ncclPatternRing:
            info.nstepsPerLoop = info.comm->nRanks-1; info.nchunksPerLoop = info.comm->nRanks; break;
        case ncclPatternRingTwice:
            info.nstepsPerLoop = 2*(info.comm->nRanks-1); info.nchunksPerLoop = info.comm->nRanks; break;
        case ncclPatternNvlsTree:
            info.nstepsPerLoop = 1; info.nchunksPerLoop = info.comm->channels[0].nvls.nHeads; break;
        default:
            WARN("Unknown pattern %d", info.pattern);
            return ncclInternalError;
    }
    
    // 生成流信息
    NCCLCHECK(ncclComputeFlowFromInfo(&info, flow));
    return ncclSuccess;
}

// 释放流信息内存
extern "C" ncclResult_t ncclFreeCollectiveFlow(struct ncclCollectiveFlow* flow) {
    if (flow == NULL) return ncclSuccess;
    
    // 释放各通道的步骤数组
    if (flow->channels) {
        for (int c = 0; c < flow->nChannels; c++) {
            if (flow->channels[c].steps) {
                free(flow->channels[c].steps);
            }
        }
        free(flow->channels);
    }
    
    // 释放主结构
    free(flow);
    
    return ncclSuccess;
}

// JSON格式输出 (简化版本)
extern "C" ncclResult_t ncclFlowToJson(struct ncclCollectiveFlow* flow, char** jsonStr) {
    if (flow == NULL || jsonStr == NULL) {
        return ncclInvalidArgument;
    }
    
    // 估算JSON大小
    size_t estimated_size = 4096 + flow->totalSteps * 512;
    char* json = (char*)malloc(estimated_size);
    if (json == NULL) {
        return ncclSystemError;
    }
    
    int pos = 0;
    pos += snprintf(json + pos, estimated_size - pos,
        "{\n"
        "  \"collective_type\": \"%s\",\n"
        "  \"algorithm\": \"%s\",\n" 
        "  \"protocol\": \"%s\",\n"
        "  \"pattern\": \"%s\",\n"
        "  \"my_rank\": %d,\n"
        "  \"total_ranks\": %d,\n"
        "  \"total_steps\": %d,\n"
        "  \"total_channels\": %d,\n"
        "  \"total_bytes\": %zu,\n"
        "  \"topology_summary\": \"%s\",\n"
        "  \"channels\": [\n",
        ncclFuncStr[flow->collType],
        ncclAlgorithmToString(flow->algorithm),
        ncclProtocolToString(flow->protocol), 
        ncclPatternToString(flow->pattern),
        flow->myRank,
        flow->nRanks,
        flow->totalSteps,
        flow->nChannels,
        flow->totalBytes,
        flow->topoInfo);
        
    // 输出各通道信息
    for (int c = 0; c < flow->nChannels; c++) {
        struct ncclChannelFlow* channel = &flow->channels[c];
        pos += snprintf(json + pos, estimated_size - pos,
            "    {\n"
            "      \"channel_id\": %d,\n"
            "      \"steps\": [\n", channel->channelId);
            
        for (int s = 0; s < channel->nSteps; s++) {
            struct ncclFlowStep* step = &channel->steps[s];
            pos += snprintf(json + pos, estimated_size - pos,
                "        {\n"
                "          \"step_id\": %d,\n"
                "          \"operation\": \"%s\",\n"
                "          \"src_rank\": %d,\n"
                "          \"dst_rank\": %d,\n"
                "          \"data_size\": %zu,\n"
                "          \"data_offset\": %zu,\n"
                "          \"channel_id\": %d,\n"
                "          \"chunk_id\": %d,\n"
                "          \"description\": \"%s\"\n"
                "        }%s\n",
                step->stepId,
                ncclFlowOpTypeToString(step->opType),
                step->srcRank,
                step->dstRank,
                step->dataSize,
                step->dataOffset,
                step->channelId,
                step->chunkId,
                step->description,
                (s < channel->nSteps - 1) ? "," : "");
        }
        
        pos += snprintf(json + pos, estimated_size - pos,
            "      ]\n"
            "    }%s\n", (c < flow->nChannels - 1) ? "," : "");
    }
    
    pos += snprintf(json + pos, estimated_size - pos, "  ]\n}\n");
    
    *jsonStr = json;
    return ncclSuccess;
}

// XML格式输出 (简化版本)
extern "C" ncclResult_t ncclFlowToXml(struct ncclCollectiveFlow* flow, char** xmlStr) {
    // 为简化，暂时返回未实现
    if (xmlStr) *xmlStr = NULL;
    return ncclInternalError;
}

// 记录：从真实proxyOp记录一条流信息到按opCount聚合的文件
extern "C" ncclResult_t ncclRecordProxyOp(const struct ncclInfo* info,
                                           const struct ncclProxyOp* proxyOp,
                                           struct ncclComm* comm) {
    if (!flowExtractionEnabled || info == nullptr || proxyOp == nullptr || comm == nullptr) return ncclSuccess;
    char path[256];
    snprintf(path, sizeof(path), "proxy_flow_rank%d.jsonl", comm->rank);
    FILE* fp = fopen(path, "a");
    if (!fp) return ncclSystemError;
    // 简化记录：每个proxyOp一条记录，包含关键信息；peer信息可从channel ring/tree推导
    const int chan = proxyOp->channelId;
    int prev = comm->channels[chan].ring.prev;
    int next = comm->channels[chan].ring.next;
    const char* pattern = ncclPatternToString((ncclPattern_t)proxyOp->pattern);
    const char* proto = ncclProtocolToString(proxyOp->protocol);
    fprintf(fp,
      "{\"opCount\":%lu,\"rank\":%d,\"channel\":%d,\"nsteps\":%d,\"nbytes\":%zd,\"chunkSize\":%d,\"sliceSteps\":%d,\"chunkSteps\":%d,\"dtype\":%u,\"redOp\":%u,\"pattern\":\"%s\",\"protocol\":\"%s\",\"ringPrev\":%d,\"ringNext\":%d}\n",
      proxyOp->opCount, comm->rank, chan, proxyOp->nsteps, proxyOp->nbytes, proxyOp->chunkSize,
      proxyOp->sliceSteps, proxyOp->chunkSteps, proxyOp->dtype, proxyOp->redOp, pattern, proto, prev, next);
    fclose(fp);

    // 逐步展开：为每个step写两条（SEND/RECV），便于仿真器直接消费
    char stepsPath[256];
    snprintf(stepsPath, sizeof(stepsPath), "flow_steps_rank%d.jsonl", comm->rank);
    FILE* fps = fopen(stepsPath, "a");
    if (!fps) return ncclSystemError;
    // 判定阶段
    const bool isRing = (proxyOp->pattern == (uint8_t)ncclPatternRing);
    const bool isRingTwice = (proxyOp->pattern == (uint8_t)ncclPatternRingTwice);
    const char* stageReduce = "reduce-scatter";
    const char* stageGather = "allgather";
    const char* stageRing = "ring";
    for (int s = 0; s < proxyOp->nsteps; ++s) {
      const char* stage = stageRing;
      if (isRingTwice) {
        int half = proxyOp->nsteps/2;
        stage = (s < half) ? stageReduce : stageGather;
      } else if (isRing) {
        stage = stageRing;
      }
      // SEND 条目
      fprintf(fps,
        "{\"opCount\":%lu,\"rank\":%d,\"channel\":%d,\"step\":%d,\"op\":\"SEND\",\"peer\":%d,\"bytes\":%zd,\"pattern\":\"%s\",\"protocol\":\"%s\",\"stage\":\"%s\"}\n",
        proxyOp->opCount, comm->rank, chan, s, next, proxyOp->nbytes, pattern, proto, stage);
      // RECV 条目
      fprintf(fps,
        "{\"opCount\":%lu,\"rank\":%d,\"channel\":%d,\"step\":%d,\"op\":\"RECV\",\"peer\":%d,\"bytes\":%zd,\"pattern\":\"%s\",\"protocol\":\"%s\",\"stage\":\"%s\"}\n",
        proxyOp->opCount, comm->rank, chan, s, prev, proxyOp->nbytes, pattern, proto, stage);
    }
    fclose(fps);
    return ncclSuccess;
} 

extern "C" ncclResult_t ncclWriteAggregatedFlow(struct ncclComm* comm) {
    if (comm == nullptr) return ncclInvalidArgument;
    char stepsPath[256], proxyPath[256], outPath[256];
    snprintf(stepsPath, sizeof(stepsPath), "flow_steps_rank%d.jsonl", comm->rank);
    snprintf(proxyPath, sizeof(proxyPath), "proxy_flow_rank%d.jsonl", comm->rank);
    snprintf(outPath, sizeof(outPath), "flow_rank%d.json", comm->rank);

    FILE* fps = fopen(stepsPath, "r");
    FILE* fpp = fopen(proxyPath, "r");
    if (fps == NULL || fpp == NULL) {
      if (fps) fclose(fps);
      if (fpp) fclose(fpp);
      return ncclSystemError;
    }

    // 简单读取到内存（不做完整解析，仅拼接为数组并排序时保留原顺序）
    // 为保持实现简洁，这里不进行真正的排序，而是按文件自然顺序输出
    // 若需要稳定排序，可后续替换为最小JSON解析与排序

    // 从 proxy 中取 meta（第一行）
    char meta[1024]; meta[0] = '\0';
    if (fgets(meta, sizeof(meta), fpp) == NULL) {
      fclose(fps); fclose(fpp);
      return ncclSystemError;
    }

    // 写聚合输出
    FILE* fo = fopen(outPath, "w");
    if (fo == NULL) { fclose(fps); fclose(fpp); return ncclSystemError; }

    fprintf(fo, "{\n");
    fprintf(fo, "  \"rank\": %d,\n", comm->rank);
    fprintf(fo, "  \"meta\": %s", meta); // meta行已包含换行，末尾不加逗号
    fprintf(fo, "  ,\n  \"steps\": [\n");

    // 逐行复制 steps
    char line[1024];
    int first = 1;
    while (fgets(line, sizeof(line), fps)) {
      // 去除行末换行
      size_t len = strlen(line);
      while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) { line[--len] = '\0'; }
      if (!first) fprintf(fo, ",\n");
      fprintf(fo, "    %s", line);
      first = 0;
    }
    fprintf(fo, "\n  ]\n}\n");

    fclose(fps);
    fclose(fpp);
    fclose(fo);
    return ncclSuccess;
} 
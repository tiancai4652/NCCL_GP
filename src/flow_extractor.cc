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
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

// 全局开关：是否启用流信息提取
static int flowExtractionEnabled = 1;

// 计算输出目录：output/<topo_base>
static void getOutputDir(char* outDir, size_t outDirSize) {
    const char* topo = getenv("NCCL_TOPO_FILE");
    const char* base = topo ? strrchr(topo, '/') : NULL;
    base = base ? base + 1 : topo;
    char name[128] = {0};
    if (base && *base) {
        // 去除扩展名
        size_t len = strnlen(base, sizeof(name)-1);
        size_t dot = len;
        for (size_t i = 0; i < len; ++i) {
            if (base[i] == '.') { dot = i; break; }
        }
        if (dot > sizeof(name)-1) dot = sizeof(name)-1;
        memcpy(name, base, dot);
        name[dot] = '\0';
    } else {
        strncpy(name, "unknown_topo", sizeof(name)-1);
    }
    // 构造 output/<name>
    snprintf(outDir, outDirSize, "output/%s", name);
}

static void ensureDir(const char* path) {
    // 创建 output 和 output/<name>
    // 首先创建 output
    (void)mkdir("output", 0777);
    if (errno != 0 && errno != EEXIST) {
        // 忽略错误，尽最大努力
        errno = 0;
    }
    // 然后创建 path
    (void)mkdir(path, 0777);
    if (errno != 0 && errno != EEXIST) {
        errno = 0;
    }
}

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

// 删除/禁用：模式化flow相关实现（保留字符串转换与权威提取路径）
// - createBaseFlow
// - ncclGenerateRingFlow
// - ncclGenerateTreeFlow
// - ncclGenerateCollNetFlow
// - ncclComputeFlowFromInfo
// - ncclGetCollectiveFlow
// - ncclFreeCollectiveFlow
// - ncclFlowToJson / ncclFlowToXml
// 上述函数已移除，避免复制/简化 NCCL 逻辑对仿真产生误导

// 记录：从真实proxyOp记录一条流信息到按opCount聚合的文件
extern "C" ncclResult_t ncclRecordProxyOp(const struct ncclInfo* info,
                                           const struct ncclProxyOp* proxyOp,
                                           struct ncclComm* comm) {
    if (!flowExtractionEnabled || info == nullptr || proxyOp == nullptr || comm == nullptr) return ncclSuccess;
    char outDir[256];
    getOutputDir(outDir, sizeof(outDir));
    ensureDir(outDir);
    char path[512];
    snprintf(path, sizeof(path), "%s/proxy_flow_rank%d.jsonl", outDir, comm->rank);
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
    char stepsPath[512];
    snprintf(stepsPath, sizeof(stepsPath), "%s/flow_steps_rank%d.jsonl", outDir, comm->rank);
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

// 新增：逐 peer 的步级记录（Tree/CollNet/NVLS/Pipeline/Ring 均可使用）
extern "C" ncclResult_t ncclRecordProxyPeerSteps(struct ncclComm* comm,
                                                  int channelId,
                                                  int type,
                                                  int peer,
                                                  const struct ncclProxyOp* op) {
  if (!flowExtractionEnabled) return ncclSuccess;
  if (comm == nullptr || op == nullptr) return ncclInvalidArgument;
  if (peer < 0) return ncclSuccess;

  char outDir[256];
  getOutputDir(outDir, sizeof(outDir));
  ensureDir(outDir);

  char stepsPath[512];
  snprintf(stepsPath, sizeof(stepsPath), "%s/flow_steps_rank%d.jsonl", outDir, comm->rank);
  FILE* fps = fopen(stepsPath, "a");
  if (!fps) return ncclSystemError;

  // 操作方向
  const char* opStr = (type == 0) ? "RECV" : "SEND"; // 0=RECV,1=SEND
  const char* pattern = ncclPatternToString((ncclPattern_t)op->pattern);
  const char* proto = ncclProtocolToString(op->protocol);

  // 阶段语义标签
  const char* stage = "generic";
  switch ((ncclPattern_t)op->pattern) {
    case ncclPatternRing: stage = "ring"; break;
    case ncclPatternRingTwice: /* 按半程拆分 */ stage = nullptr; break;
    case ncclPatternPipelineFrom: stage = "pipeline-from"; break;
    case ncclPatternPipelineTo: stage = "pipeline-to"; break;
    case ncclPatternTreeUp: stage = "tree-up"; break;
    case ncclPatternTreeDown: stage = "tree-down"; break;
    case ncclPatternTreeUpDown: /* 按半程拆分 */ stage = nullptr; break;
    case ncclPatternCollnetChain: stage = "collnet-chain"; break;
    case ncclPatternCollnetDirect: stage = "collnet-direct"; break;
    case ncclPatternNvls: stage = "nvls"; break;
    case ncclPatternNvlsTree: stage = "nvls-tree"; break;
    default: stage = "generic"; break;
  }

  for (int s = 0; s < op->nsteps; ++s) {
    const char* curStage = stage;
    if (stage == nullptr) {
      // RingTwice / TreeUpDown：前半与后半阶段标签不同
      int half = op->nsteps/2;
      if ((ncclPattern_t)op->pattern == ncclPatternRingTwice) {
        curStage = (s < half) ? "reduce-scatter" : "allgather";
      } else if ((ncclPattern_t)op->pattern == ncclPatternTreeUpDown) {
        curStage = (s < half) ? "tree-up" : "tree-down";
      }
    }
    fprintf(fps,
      "{\"opCount\":%lu,\"rank\":%d,\"channel\":%d,\"step\":%d,\"op\":\"%s\",\"peer\":%d,\"bytes\":%zd,\"pattern\":\"%s\",\"protocol\":\"%s\",\"stage\":\"%s\"}\n",
      op->opCount, comm->rank, channelId, s, opStr, peer, op->nbytes, pattern, proto, curStage ? curStage : "generic");
  }

  fclose(fps);
  return ncclSuccess;
}

extern "C" ncclResult_t ncclWriteAggregatedFlow(struct ncclComm* comm) {
    if (comm == nullptr) return ncclInvalidArgument;
    char outDir[256];
    getOutputDir(outDir, sizeof(outDir));
    ensureDir(outDir);
    char stepsPath[512], proxyPath[512], outPath[512];
    snprintf(stepsPath, sizeof(stepsPath), "%s/flow_steps_rank%d.jsonl", outDir, comm->rank);
    snprintf(proxyPath, sizeof(proxyPath), "%s/proxy_flow_rank%d.jsonl", outDir, comm->rank);
    snprintf(outPath, sizeof(outPath), "%s/flow_rank%d.json", outDir, comm->rank);

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

extern "C" ncclResult_t ncclExtractFlow(
    ncclFunc_t collType,
    size_t count,
    ncclDataType_t dataType,
    int root,
    ncclComm_t comm) {
    if (!flowExtractionEnabled) return ncclInvalidArgument;
    if (comm == NULL) return ncclInvalidArgument;

    ncclResult_t res = ncclSuccess;
    switch (collType) {
      case ncclFuncAllReduce:
        res = ncclAllReduce(NULL, NULL, count, dataType, ncclSum, comm, (cudaStream_t)0);
        break;
      case ncclFuncAllGather:
        res = ncclAllGather(NULL, NULL, count, dataType, comm, (cudaStream_t)0);
        break;
      case ncclFuncReduceScatter:
        res = ncclReduceScatter(NULL, NULL, count, dataType, ncclSum, comm, (cudaStream_t)0);
        break;
      case ncclFuncBroadcast:
        res = ncclBroadcast(NULL, NULL, count, dataType, root, comm, (cudaStream_t)0);
        break;
      case ncclFuncReduce:
        res = ncclReduce(NULL, NULL, count, dataType, ncclSum, root, comm, (cudaStream_t)0);
        break;
      default:
        return ncclInvalidArgument;
    }
    if (res != ncclSuccess) return res;
    return ncclWriteAggregatedFlow(comm);
} 
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

// 启用/禁用流信息提取
extern "C" ncclResult_t ncclSetFlowExtractionEnabled(int enable) {
    flowExtractionEnabled = enable;
    INFO(NCCL_INIT, "Flow extraction %s", enable ? "enabled" : "disabled");
    return ncclSuccess;
}

// 记录：从真实 proxyOp 记录一条流信息到按 opCount 聚合的文件
// 本函数只记录 NCCL 真实生成的 proxyOp，不做任何推测或计算
// 所有字段都直接从 NCCL 结构体读取
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
    // 记录每个 proxyOp 的摘要信息，ringPrev/ringNext 来自 NCCL 初始化的 ring 拓扑（仅供参考）
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

    // 注意：flow_steps_rank*.jsonl 的生成已移至 ncclRecordProxyPeerSteps()，
    // 确保使用真实的 peer 信息（从 SaveProxy 传入），而不是基于 Ring 拓扑的假设。
    // 这保证了所有通信模式（Ring/Tree/CollNet/NVLS/Pipeline）的准确性。
    
    return ncclSuccess;
} 

// 逐 peer 的步级记录（Tree/CollNet/NVLS/Pipeline/Ring 均可使用）
// 本函数记录来自 SaveProxy 的真实 peer 信息，peer 参数来自 NCCL 拓扑结构
// 所有通信模式的 peer 都是准确的，无任何推测
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

    FILE* fpp = fopen(proxyPath, "r");
    if (fpp == NULL) {
      return ncclSystemError;
    }
    
    FILE* fps = fopen(stepsPath, "r");
    // flow_steps 文件可能不存在（如果 SaveProxy 未被调用）
    bool hasSteps = (fps != NULL);

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
    if (fo == NULL) {
      if (fps) fclose(fps);
      fclose(fpp);
      return ncclSystemError;
    }

    fprintf(fo, "{\n");
    fprintf(fo, "  \"rank\": %d,\n", comm->rank);
    fprintf(fo, "  \"meta\": %s", meta); // meta行已包含换行，末尾不加逗号
    fprintf(fo, "  ,\n  \"steps\": [\n");

    // 逐行复制 steps（如果文件存在）
    if (hasSteps) {
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
      fclose(fps);
    }
    fprintf(fo, "\n  ]\n}\n");

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
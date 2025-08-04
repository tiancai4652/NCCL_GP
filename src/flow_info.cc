/*
 * @Author: CodeBuddy
 * @Date: 2024-05-15
 * @Description: 用于获取NCCL集合通信流信息的实现
 */

#include "flow_info.h"
#include "core.h"
#include "nccl.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "enqueue.h"
#include "transport.h"
#include "collectives.h"
#include "debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 用于跟踪流信息的全局变量
static int isFlowTrackingEnabled = 0;
static ncclCollFlowInfo_t* currentFlowInfo = NULL;
static int currentFlowCount = 0;

// 流信息跟踪回调函数
typedef void (*flowTrackCallback_t)(int srcRank, int dstRank, int channelId, size_t bytes, ncclFlowDirection_t direction);
static flowTrackCallback_t flowTrackCallback = NULL;

// 设置流信息跟踪回调
static void setFlowTrackCallback(flowTrackCallback_t callback) {
  flowTrackCallback = callback;
}

// 流信息跟踪回调实现
static void flowTrackCallbackImpl(int srcRank, int dstRank, int channelId, size_t bytes, ncclFlowDirection_t direction) {
  if (currentFlowInfo && currentFlowCount < currentFlowInfo->nFlows) {
    ncclFlowInfo_t* flow = &currentFlowInfo->flows[currentFlowCount++];
    flow->srcRank = srcRank;
    flow->dstRank = dstRank;
    flow->channelId = channelId;
    flow->bytes = bytes;
    flow->direction = direction;
    flow->stepOrder = currentFlowCount;
    
    INFO(NCCL_COLL, "Flow %d: %s from rank %d to rank %d on channel %d, bytes: %zu", 
         currentFlowCount, 
         direction == FLOW_DIRECTION_SEND ? "SEND" : "RECV",
         srcRank, dstRank, channelId, bytes);
  }
}

// 初始化流信息系统
ncclResult_t ncclFlowInfoInit(struct ncclComm* comm) {
  // 设置流信息跟踪回调
  setFlowTrackCallback(flowTrackCallbackImpl);
  isFlowTrackingEnabled = 1;
  return ncclSuccess;
}

// 获取算法类型字符串
static const char* getAlgoTypeStr(ncclFlowAlgoType_t algoType) {
  switch (algoType) {
    case FLOW_ALGO_RING: return "RING";
    case FLOW_ALGO_TREE: return "TREE";
    case FLOW_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
    case FLOW_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
    case FLOW_ALGO_NVLS: return "NVLS";
    case FLOW_ALGO_NVLS_TREE: return "NVLS_TREE";
    default: return "UNKNOWN";
  }
}

// 获取协议类型字符串
static const char* getProtoTypeStr(ncclFlowProtoType_t protoType) {
  switch (protoType) {
    case FLOW_PROTO_LL: return "LL";
    case FLOW_PROTO_LL128: return "LL128";
    case FLOW_PROTO_SIMPLE: return "SIMPLE";
    default: return "UNKNOWN";
  }
}

// 获取集合通信类型字符串
static const char* getCollTypeStr(ncclFlowCollType_t collType) {
  switch (collType) {
    case FLOW_ALLREDUCE: return "ALLREDUCE";
    case FLOW_BROADCAST: return "BROADCAST";
    case FLOW_REDUCE: return "REDUCE";
    case FLOW_ALLGATHER: return "ALLGATHER";
    case FLOW_REDUCESCATTER: return "REDUCESCATTER";
    case FLOW_SENDRECV: return "SENDRECV";
    case FLOW_ALLTOALL: return "ALLTOALL";
    default: return "UNKNOWN";
  }
}

// 模拟NCCL的算法选择逻辑
static void selectAlgoProto(struct ncclComm* comm, ncclFlowCollType_t collType, size_t count, 
                           ncclFlowAlgoType_t* algoType, ncclFlowProtoType_t* protoType) {
  // 默认值
  *algoType = FLOW_ALGO_RING;
  *protoType = FLOW_PROTO_SIMPLE;
  
  // 根据通信量和拓扑选择算法和协议
  // 这里是简化的逻辑，实际NCCL的选择更复杂
  int nvlinkPresent = ncclTopoPathAllNVLink(comm->topo);
  
  // 小数据量使用LL协议
  if (count < 32768) {
    *protoType = FLOW_PROTO_LL;
  } 
  // 中等数据量使用LL128协议
  else if (count < 131072) {
    *protoType = FLOW_PROTO_LL128;
  }
  
  // 如果有NVLink，考虑使用Tree或NVLS算法
  if (nvlinkPresent) {
    switch (collType) {
      case FLOW_ALLREDUCE:
      case FLOW_REDUCE:
      case FLOW_REDUCESCATTER:
        if (comm->nRanks > 8) {
          *algoType = FLOW_ALGO_TREE;
        }
        break;
      case FLOW_BROADCAST:
      case FLOW_ALLGATHER:
        if (comm->nRanks > 4) {
          *algoType = FLOW_ALGO_TREE;
        }
        break;
      default:
        break;
    }
  }
  
  // 如果支持NVLS，对某些操作使用NVLS
  if (comm->nvlsSupport && (collType == FLOW_ALLREDUCE || collType == FLOW_REDUCE)) {
    *algoType = FLOW_ALGO_NVLS;
  }
}

// 估计流的数量
static int estimateFlowCount(struct ncclComm* comm, ncclFlowCollType_t collType, ncclFlowAlgoType_t algoType) {
  int nRanks = comm->nRanks;
  int nChannels = comm->nChannels;
  
  switch (collType) {
    case FLOW_ALLREDUCE:
      if (algoType == FLOW_ALGO_RING) {
        return nRanks * 2 * nChannels; // 每个rank发送和接收两次
      } else if (algoType == FLOW_ALGO_TREE) {
        return nRanks * 3 * nChannels; // 树结构有更多的通信
      } else {
        return nRanks * 2 * nChannels; // 默认
      }
    case FLOW_BROADCAST:
    case FLOW_REDUCE:
      if (algoType == FLOW_ALGO_RING) {
        return nRanks * nChannels;
      } else if (algoType == FLOW_ALGO_TREE) {
        return nRanks * 2 * nChannels;
      } else {
        return nRanks * nChannels;
      }
    case FLOW_ALLGATHER:
    case FLOW_REDUCESCATTER:
      return nRanks * nChannels;
    case FLOW_SENDRECV:
      return 2 * nChannels; // 每个通道一个发送和一个接收
    case FLOW_ALLTOALL:
      return nRanks * nRanks * nChannels; // 每个rank与其他所有rank通信
    default:
      return nRanks * nChannels; // 默认
  }
}

// 生成Ring算法的流信息
static void generateRingFlows(struct ncclComm* comm, ncclFlowCollType_t collType, size_t count, 
                             ncclCollFlowInfo_t* flowInfo) {
  int nRanks = comm->nRanks;
  int nChannels = comm->nChannels;
  int rank = comm->rank;
  size_t bytes = count * ncclTypeSize(ncclFloat); // 假设为float类型
  
  currentFlowCount = 0;
  
  // 为每个通道生成环形通信模式
  for (int c = 0; c < nChannels; c++) {
    for (int i = 0; i < nRanks; i++) {
      int prevRank = (rank - 1 + nRanks) % nRanks;
      int nextRank = (rank + 1) % nRanks;
      
      // 根据集合通信类型生成不同的流模式
      switch (collType) {
        case FLOW_ALLREDUCE:
          // Reduce-Scatter阶段
          flowTrackCallback(prevRank, rank, c, bytes/nRanks, FLOW_DIRECTION_RECV);
          flowTrackCallback(rank, nextRank, c, bytes/nRanks, FLOW_DIRECTION_SEND);
          
          // All-Gather阶段
          flowTrackCallback(prevRank, rank, c, bytes/nRanks, FLOW_DIRECTION_RECV);
          flowTrackCallback(rank, nextRank, c, bytes/nRanks, FLOW_DIRECTION_SEND);
          break;
          
        case FLOW_BROADCAST:
          flowTrackCallback(prevRank, rank, c, bytes, FLOW_DIRECTION_RECV);
          flowTrackCallback(rank, nextRank, c, bytes, FLOW_DIRECTION_SEND);
          break;
          
        case FLOW_REDUCE:
          flowTrackCallback(prevRank, rank, c, bytes, FLOW_DIRECTION_RECV);
          flowTrackCallback(rank, nextRank, c, bytes, FLOW_DIRECTION_SEND);
          break;
          
        case FLOW_ALLGATHER:
          flowTrackCallback(prevRank, rank, c, bytes/nRanks, FLOW_DIRECTION_RECV);
          flowTrackCallback(rank, nextRank, c, bytes/nRanks, FLOW_DIRECTION_SEND);
          break;
          
        case FLOW_REDUCESCATTER:
          flowTrackCallback(prevRank, rank, c, bytes/nRanks, FLOW_DIRECTION_RECV);
          flowTrackCallback(rank, nextRank, c, bytes/nRanks, FLOW_DIRECTION_SEND);
          break;
          
        default:
          break;
      }
    }
  }
}

// 生成Tree算法的流信息
static void generateTreeFlows(struct ncclComm* comm, ncclFlowCollType_t collType, size_t count, 
                             ncclCollFlowInfo_t* flowInfo) {
  int nRanks = comm->nRanks;
  int nChannels = comm->nChannels;
  int rank = comm->rank;
  size_t bytes = count * ncclTypeSize(ncclFloat); // 假设为float类型
  
  currentFlowCount = 0;
  
  // 为每个通道生成树形通信模式
  for (int c = 0; c < nChannels; c++) {
    // 简化的树结构：每个节点最多有两个子节点
    int parent = (rank == 0) ? -1 : (rank - 1) / 2;
    int child1 = rank * 2 + 1;
    int child2 = rank * 2 + 2;
    
    // 检查子节点是否有效
    if (child1 >= nRanks) child1 = -1;
    if (child2 >= nRanks) child2 = -1;
    
    // 根据集合通信类型生成不同的流模式
    switch (collType) {
      case FLOW_ALLREDUCE:
        // Reduce阶段（从叶子到根）
        if (child1 != -1) {
          flowTrackCallback(child1, rank, c, bytes, FLOW_DIRECTION_RECV);
        }
        if (child2 != -1) {
          flowTrackCallback(child2, rank, c, bytes, FLOW_DIRECTION_RECV);
        }
        if (parent != -1) {
          flowTrackCallback(rank, parent, c, bytes, FLOW_DIRECTION_SEND);
        }
        
        // Broadcast阶段（从根到叶子）
        if (parent != -1) {
          flowTrackCallback(parent, rank, c, bytes, FLOW_DIRECTION_RECV);
        }
        if (child1 != -1) {
          flowTrackCallback(rank, child1, c, bytes, FLOW_DIRECTION_SEND);
        }
        if (child2 != -1) {
          flowTrackCallback(rank, child2, c, bytes, FLOW_DIRECTION_SEND);
        }
        break;
        
      case FLOW_BROADCAST:
        if (parent != -1) {
          flowTrackCallback(parent, rank, c, bytes, FLOW_DIRECTION_RECV);
        }
        if (child1 != -1) {
          flowTrackCallback(rank, child1, c, bytes, FLOW_DIRECTION_SEND);
        }
        if (child2 != -1) {
          flowTrackCallback(rank, child2, c, bytes, FLOW_DIRECTION_SEND);
        }
        break;
        
      case FLOW_REDUCE:
        if (child1 != -1) {
          flowTrackCallback(child1, rank, c, bytes, FLOW_DIRECTION_RECV);
        }
        if (child2 != -1) {
          flowTrackCallback(child2, rank, c, bytes, FLOW_DIRECTION_RECV);
        }
        if (parent != -1) {
          flowTrackCallback(rank, parent, c, bytes, FLOW_DIRECTION_SEND);
        }
        break;
        
      default:
        // 其他集合通信类型使用Ring算法
        generateRingFlows(comm, collType, count, flowInfo);
        break;
    }
  }
}

// 获取集合通信流信息
ncclResult_t ncclGetCollFlowInfo(
    struct ncclComm* comm,
    ncclFlowCollType_t collType,
    size_t count,
    ncclDataType_t dataType,
    ncclRedOp_t redOp,
    int root,
    ncclCollFlowInfo_t** flowInfo) {
  
  // 检查参数
  if (comm == NULL || flowInfo == NULL) {
    return ncclInvalidArgument;
  }
  
  // 选择算法和协议
  ncclFlowAlgoType_t algoType;
  ncclFlowProtoType_t protoType;
  selectAlgoProto(comm, collType, count, &algoType, &protoType);
  
  // 估计流的数量
  int nFlows = estimateFlowCount(comm, collType, algoType);
  
  // 分配流信息结构
  ncclCollFlowInfo_t* info = (ncclCollFlowInfo_t*)malloc(sizeof(ncclCollFlowInfo_t));
  if (info == NULL) {
    return ncclSystemError;
  }
  
  // 分配流数组
  info->flows = (ncclFlowInfo_t*)malloc(nFlows * sizeof(ncclFlowInfo_t));
  if (info->flows == NULL) {
    free(info);
    return ncclSystemError;
  }
  
  // 初始化流信息
  info->collType = collType;
  info->algoType = algoType;
  info->protoType = protoType;
  info->nRanks = comm->nRanks;
  info->nFlows = nFlows;
  
  // 设置当前流信息为全局变量
  currentFlowInfo = info;
  currentFlowCount = 0;
  
  // 根据算法生成流信息
  switch (algoType) {
    case FLOW_ALGO_RING:
      generateRingFlows(comm, collType, count, info);
      break;
    case FLOW_ALGO_TREE:
      generateTreeFlows(comm, collType, count, info);
      break;
    case FLOW_ALGO_NVLS:
    case FLOW_ALGO_NVLS_TREE:
    case FLOW_ALGO_COLLNET_DIRECT:
    case FLOW_ALGO_COLLNET_CHAIN:
      // 这些高级算法需要更复杂的实现，暂时使用Ring算法代替
      generateRingFlows(comm, collType, count, info);
      break;
    default:
      generateRingFlows(comm, collType, count, info);
      break;
  }
  
  // 更新实际流的数量
  info->nFlows = currentFlowCount;
  
  // 清除全局变量
  currentFlowInfo = NULL;
  
  // 返回流信息
  *flowInfo = info;
  
  return ncclSuccess;
}

// 释放流信息
void ncclFreeCollFlowInfo(ncclCollFlowInfo_t* flowInfo) {
  if (flowInfo) {
    if (flowInfo->flows) {
      free(flowInfo->flows);
    }
    free(flowInfo);
  }
}

// 打印流信息
void ncclPrintCollFlowInfo(ncclCollFlowInfo_t* flowInfo) {
  if (flowInfo == NULL) {
    printf("流信息为空\n");
    return;
  }
  
  printf("集合通信类型: %s\n", getCollTypeStr(flowInfo->collType));
  printf("算法类型: %s\n", getAlgoTypeStr(flowInfo->algoType));
  printf("协议类型: %s\n", getProtoTypeStr(flowInfo->protoType));
  printf("参与通信的总rank数: %d\n", flowInfo->nRanks);
  printf("流的总数量: %d\n", flowInfo->nFlows);
  
  printf("\n流信息详情:\n");
  printf("%-10s %-10s %-10s %-10s %-10s %-10s\n", 
         "步骤", "源Rank", "目标Rank", "通道ID", "字节数", "方向");
  
  for (int i = 0; i < flowInfo->nFlows; i++) {
    ncclFlowInfo_t* flow = &flowInfo->flows[i];
    printf("%-10d %-10d %-10d %-10d %-10zu %-10s\n", 
           flow->stepOrder,
           flow->srcRank,
           flow->dstRank,
           flow->channelId,
           flow->bytes,
           flow->direction == FLOW_DIRECTION_SEND ? "发送" : "接收");
  }
}
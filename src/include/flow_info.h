/*
 * @Author: CodeBuddy
 * @Date: 2024-05-15
 * @Description: 用于获取NCCL集合通信流信息的接口
 */

#ifndef NCCL_FLOW_INFO_H_
#define NCCL_FLOW_INFO_H_

#include "nccl.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"

// 集合通信类型
typedef enum {
  FLOW_ALLREDUCE = 0,
  FLOW_BROADCAST = 1,
  FLOW_REDUCE = 2,
  FLOW_ALLGATHER = 3,
  FLOW_REDUCESCATTER = 4,
  FLOW_SENDRECV = 5,
  FLOW_ALLTOALL = 6
} ncclFlowCollType_t;

// 算法类型
typedef enum {
  FLOW_ALGO_RING = 0,
  FLOW_ALGO_TREE = 1,
  FLOW_ALGO_COLLNET_DIRECT = 2,
  FLOW_ALGO_COLLNET_CHAIN = 3,
  FLOW_ALGO_NVLS = 4,
  FLOW_ALGO_NVLS_TREE = 5
} ncclFlowAlgoType_t;

// 协议类型
typedef enum {
  FLOW_PROTO_LL = 0,
  FLOW_PROTO_LL128 = 1,
  FLOW_PROTO_SIMPLE = 2
} ncclFlowProtoType_t;

// 流方向
typedef enum {
  FLOW_DIRECTION_SEND = 0,
  FLOW_DIRECTION_RECV = 1
} ncclFlowDirection_t;

// 流信息结构体
typedef struct {
  int srcRank;           // 源节点rank
  int dstRank;           // 目标节点rank
  int channelId;         // 通道ID
  size_t bytes;          // 通信字节数
  ncclFlowDirection_t direction; // 流方向
  int stepOrder;         // 在整个集合通信中的步骤顺序
} ncclFlowInfo_t;

// 集合通信流信息结构体
typedef struct {
  ncclFlowCollType_t collType;    // 集合通信类型
  ncclFlowAlgoType_t algoType;    // 算法类型
  ncclFlowProtoType_t protoType;  // 协议类型
  int nRanks;                     // 参与通信的总rank数
  int nFlows;                     // 流的总数量
  ncclFlowInfo_t* flows;          // 流信息数组
} ncclCollFlowInfo_t;

// 初始化流信息系统
ncclResult_t ncclFlowInfoInit(struct ncclComm* comm);

// 获取集合通信流信息
ncclResult_t ncclGetCollFlowInfo(
    struct ncclComm* comm,         // NCCL通信器
    ncclFlowCollType_t collType,   // 集合通信类型
    size_t count,                  // 元素数量
    ncclDataType_t dataType,       // 数据类型
    ncclRedOp_t redOp,             // 归约操作(对于需要的集合通信)
    int root,                      // 根节点(对于需要的集合通信)
    ncclCollFlowInfo_t** flowInfo  // 输出的流信息
);

// 释放流信息
void ncclFreeCollFlowInfo(ncclCollFlowInfo_t* flowInfo);

// 打印流信息
void ncclPrintCollFlowInfo(ncclCollFlowInfo_t* flowInfo);

#endif // NCCL_FLOW_INFO_H_
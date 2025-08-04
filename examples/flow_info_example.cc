/*
 * @Author: CodeBuddy
 * @Date: 2024-05-15
 * @Description: 展示如何使用NCCL流信息功能的示例程序
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "nccl.h"
#include "flow_info.h"

int main(int argc, char* argv[]) {
  // 初始化NCCL
  ncclComm_t comm;
  int nRanks = 4;  // 假设有4个rank
  int rank = 0;    // 当前rank为0
  
  // 初始化NCCL通信器
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  ncclCommInitRank(&comm, nRanks, id, rank);
  
  // 初始化流信息系统
  ncclFlowInfoInit((struct ncclComm*)comm);
  
  // 定义集合通信参数
  size_t count = 1024 * 1024;  // 1M个元素
  ncclDataType_t dataType = ncclFloat;
  ncclRedOp_t redOp = ncclSum;
  int root = 0;
  
  // 获取AllReduce的流信息
  ncclCollFlowInfo_t* allReduceInfo = NULL;
  ncclGetCollFlowInfo(
      (struct ncclComm*)comm,
      FLOW_ALLREDUCE,
      count,
      dataType,
      redOp,
      root,
      &allReduceInfo
  );
  
  // 打印AllReduce的流信息
  printf("\n===== AllReduce流信息 =====\n");
  ncclPrintCollFlowInfo(allReduceInfo);
  
  // 获取Broadcast的流信息
  ncclCollFlowInfo_t* broadcastInfo = NULL;
  ncclGetCollFlowInfo(
      (struct ncclComm*)comm,
      FLOW_BROADCAST,
      count,
      dataType,
      redOp,
      root,
      &broadcastInfo
  );
  
  // 打印Broadcast的流信息
  printf("\n===== Broadcast流信息 =====\n");
  ncclPrintCollFlowInfo(broadcastInfo);
  
  // 释放资源
  ncclFreeCollFlowInfo(allReduceInfo);
  ncclFreeCollFlowInfo(broadcastInfo);
  ncclCommDestroy(comm);
  
  return 0;
}
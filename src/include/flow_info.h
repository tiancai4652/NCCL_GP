/*************************************************************************
 * Copyright (c) 2024, Flow Info Extraction Tool
 * 
 * NCCL集合通信流信息提取工具
 * 用于提取NCCL算法选择和通信流信息，不实际执行通信
 ************************************************************************/

#ifndef NCCL_FLOW_INFO_H_
#define NCCL_FLOW_INFO_H_

#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <string>

// 前向声明，避免循环依赖
struct ncclComm;
typedef int ncclFunc_t;
typedef int ncclDataType_t;

// 流信息步骤类型
enum ncclFlowStepType {
  FLOW_STEP_SEND = 0,      // 发送数据
  FLOW_STEP_RECV = 1,      // 接收数据  
  FLOW_STEP_REDUCE = 2,    // 归约操作
  FLOW_STEP_COPY = 3,      // 数据拷贝
  FLOW_STEP_WAIT = 4       // 等待同步
};

// 单个流步骤信息
struct ncclFlowStep {
  int stepId;                    // 步骤ID
  ncclFlowStepType type;         // 步骤类型
  int srcRank;                   // 源节点rank
  int dstRank;                   // 目标节点rank
  size_t dataSize;               // 数据大小(字节)
  int channel;                   // 使用的通道
  int phase;                     // 所属阶段
  char description[256];         // 步骤描述
};

// 算法选择信息
struct ncclAlgorithmInfo {
  ncclFunc_t collective;         // 集合通信类型
  int algorithm;                 // 选择的算法(RING/TREE等)
  int protocol;                  // 选择的协议(LL/LL128/SIMPLE)
  int nChannels;                 // 使用的通道数
  int nThreads;                  // 每个块的线程数
  size_t chunkSize;              // 数据块大小
  float bandwidth;               // 预期带宽
  float latency;                 // 预期延迟
  char reason[512];              // 算法选择原因
};

// 完整的流信息
struct ncclFlowInfo {
  // 基本信息
  int rank;                      // 当前节点rank
  int nRanks;                    // 总节点数
  size_t totalBytes;             // 总数据量
  ncclDataType_t dataType;       // 数据类型
  
  // 算法信息
  ncclAlgorithmInfo algInfo;
  
  // 流步骤列表
  std::vector<ncclFlowStep> steps;
  
  // 统计信息
  int totalSteps;                // 总步骤数
  int totalPhases;               // 总阶段数
  double estimatedTime;          // 预估执行时间(ms)
};

// 全局流信息收集器
class ncclFlowCollector {
private:
  static ncclFlowCollector* instance;
  ncclFlowInfo currentFlow;
  bool enabled;
  FILE* logFile;
  
public:
  static ncclFlowCollector* getInstance();
  
  // 启用/禁用流信息收集
  void enable() { enabled = true; }
  void disable() { enabled = false; }
  bool isEnabled() const { return enabled; }
  
  // 初始化流信息收集
  void initFlow(ncclComm* comm, ncclFunc_t collective, size_t bytes, ncclDataType_t dataType);
  
  // 设置算法信息
  void setAlgorithmInfo(int algorithm, int protocol, int nChannels, int nThreads, 
                       size_t chunkSize, float bandwidth, float latency, const char* reason);
  
  // 添加流步骤
  void addFlowStep(ncclFlowStepType type, int srcRank, int dstRank, 
                   size_t dataSize, int channel, int phase, const char* description);
  
  // 简化的添加步骤接口
  void addStep(int channel, int stepId, const char* description);
  
  // 完成流信息收集并输出
  void finalizeFlow();
  
  // 设置日志文件
  void setLogFile(const char* filename);
  void closeLogFile();
  
  // 输出流信息
  void printFlowInfo();
  void saveToFile(const char* filename);
  
private:
  ncclFlowCollector();
  ~ncclFlowCollector();
};

// 便利宏定义
#define FLOW_INFO_INIT(comm, collective, bytes, dataType) \
  do { \
    if (ncclFlowCollector::getInstance()->isEnabled()) { \
      ncclFlowCollector::getInstance()->initFlow(comm, collective, bytes, dataType); \
    } \
  } while(0)

#define FLOW_INFO_SET_ALGORITHM(alg, proto, nCh, nTh, chunk, bw, lat, reason) \
  do { \
    if (ncclFlowCollector::getInstance()->isEnabled()) { \
      ncclFlowCollector::getInstance()->setAlgorithmInfo(alg, proto, nCh, nTh, chunk, bw, lat, reason); \
    } \
  } while(0)

#define FLOW_INFO_ADD_STEP(ch, stepId, desc) \
  do { \
    if (ncclFlowCollector::getInstance()->isEnabled()) { \
      ncclFlowCollector::getInstance()->addStep(ch, stepId, desc); \
    } \
  } while(0)

#define FLOW_INFO_FINALIZE() \
  do { \
    if (ncclFlowCollector::getInstance()->isEnabled()) { \
      ncclFlowCollector::getInstance()->finalizeFlow(); \
    } \
  } while(0)

#endif // NCCL_FLOW_INFO_H_
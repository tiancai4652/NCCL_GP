# NCCL流信息获取工具

## 项目概述

本项目基于NCCL_GP，实现了一个用于获取NCCL集合通信流信息的工具，无需实际执行通信操作。这对于大模型仿真工作非常有用，可以帮助分析和预测集合通信的性能和行为。

## 主要功能

- 获取NCCL为特定集合通信操作选择的算法和协议
- 获取通信流的详细信息，包括源节点、目标节点、通道ID、通信字节数、流方向和步骤顺序
- 分析NCCL的通信模式和优化策略

## 新增文件

1. `src/include/flow_info.h` - 定义了获取NCCL集合通信流信息的接口和数据结构
2. `src/flow_info.cc` - 实现了流信息获取和处理的核心功能
3. `examples/flow_info_example.cc` - 提供了一个使用示例
4. `examples/Makefile` - 用于编译示例程序

## 编译方法

### 编译NCCL_GP库

```bash
cd NCCL_GP/src
make
```

### 编译示例程序

```bash
cd NCCL_GP/examples
make
```

## 使用方法

### 运行示例程序

```bash
cd NCCL_GP/examples
./flow_info_example
```

### API使用说明

1. 初始化流信息系统：

```c
ncclResult_t ncclFlowInfoInit();
```

2. 获取集合通信流信息：

```c
ncclResult_t ncclGetCollFlowInfo(
    ncclComm_t comm,           // NCCL通信器
    void* sendbuff,            // 发送缓冲区
    void* recvbuff,            // 接收缓冲区
    size_t count,              // 元素数量
    ncclDataType_t dataType,   // 数据类型
    ncclRedOp_t op,            // 归约操作
    ncclCollType_t collType,   // 集合通信类型
    ncclFlowInfo_t** flowInfo  // 输出的流信息
);
```

3. 打印流信息：

```c
void ncclPrintCollFlowInfo(ncclFlowInfo_t* flowInfo);
```

4. 释放流信息资源：

```c
void ncclFreeCollFlowInfo(ncclFlowInfo_t* flowInfo);
```

## 集成到仿真系统

要将此工具集成到您的大模型仿真系统中，您可以：

1. 从workload文件中读取集合通信任务信息
2. 调用`ncclGetCollFlowInfo`获取流信息
3. 将流信息转换为仿真系统可理解的格式
4. 在仿真系统中模拟通信行为

## 注意事项

- 本工具基于NCCL_GP，不需要实际的NVIDIA硬件支持
- 流信息的准确性取决于NCCL_GP对NCCL行为的模拟精度
- 对于复杂的拓扑和大规模集群，可能需要进一步调整和优化
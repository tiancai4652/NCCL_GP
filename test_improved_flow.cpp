/*************************************************************************
 * NCCL流信息提取工具改进测试
 * 测试算法名称、协议名称和详细流步骤显示
 ************************************************************************/

#include <iostream>
#include <cstdio>
#include <cstring>

// 模拟NCCL结构
struct ncclComm {};

// 包含我们的流信息头文件
#include "src/include/flow_info.h"

int main() {
    printf("=== NCCL流信息提取工具改进测试 ===\n\n");
    
    // 启用流信息收集
    ncclFlowCollector::getInstance()->enable();
    printf("流信息收集已启用\n\n");
    
    // 模拟初始化流信息
    ncclComm* comm = nullptr;
    ncclFlowCollector::getInstance()->initFlow(comm, 4, 1048576, 0); // AllReduce, 1MB数据
    
    // 测试不同的算法和协议组合
    printf("测试1: RING算法 + SIMPLE协议\n");
    ncclFlowCollector::getInstance()->setAlgorithmInfo(
        1,      // RING算法
        2,      // SIMPLE协议
        4,      // 4个通道
        256,    // 256个线程
        131072, // 128KB块大小
        10.0,   // 10 GB/s带宽
        5.0,    // 5us延迟
        "基于节点数和数据大小选择RING算法，使用SIMPLE协议获得最佳性能"
    );
    
    // 添加详细的流步骤
    printf("\n添加流执行步骤...\n");
    
    // 通道0的步骤
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_SEND, 0, 1, 262144, 0, 0, "发送数据块到下一个节点"
    );
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_RECV, 3, 0, 262144, 0, 0, "接收数据块从上一个节点"
    );
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_REDUCE, 0, 0, 262144, 0, 0, "执行归约操作"
    );
    
    // 通道1的步骤
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_SEND, 0, 1, 262144, 1, 0, "发送数据块到下一个节点"
    );
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_RECV, 3, 0, 262144, 1, 0, "接收数据块从上一个节点"
    );
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_REDUCE, 0, 0, 262144, 1, 0, "执行归约操作"
    );
    
    // 通道2的步骤
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_SEND, 0, 1, 262144, 2, 1, "第二阶段：发送归约结果"
    );
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_RECV, 3, 0, 262144, 2, 1, "第二阶段：接收归约结果"
    );
    
    // 通道3的步骤
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_SEND, 0, 1, 262144, 3, 1, "第二阶段：发送归约结果"
    );
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_RECV, 3, 0, 262144, 3, 1, "第二阶段：接收归约结果"
    );
    
    // 完成流信息收集
    ncclFlowCollector::getInstance()->finalizeFlow();
    
    // 输出流信息
    ncclFlowCollector::getInstance()->printFlowInfo();
    
    // 保存到文件
    ncclFlowCollector::getInstance()->saveToFile("improved_flow_test.log");
    
    printf("\n=== 测试2: TREE算法 + LL128协议 ===\n");
    
    // 重新初始化
    ncclFlowCollector::getInstance()->initFlow(comm, 1, 2097152, 1); // Broadcast, 2MB数据
    
    ncclFlowCollector::getInstance()->setAlgorithmInfo(
        0,      // TREE算法
        1,      // LL128协议
        2,      // 2个通道
        512,    // 512个线程
        65536,  // 64KB块大小
        8.0,    // 8 GB/s带宽
        10.0,   // 10us延迟
        "对于Broadcast操作，TREE算法能够提供更好的扇出性能"
    );
    
    // 添加TREE算法的流步骤
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_SEND, 0, 1, 1048576, 0, 0, "根节点发送到左子节点"
    );
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_SEND, 0, 2, 1048576, 0, 0, "根节点发送到右子节点"
    );
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_SEND, 1, 3, 1048576, 1, 1, "左子节点转发到叶节点"
    );
    ncclFlowCollector::getInstance()->addFlowStep(
        FLOW_STEP_RECV, 0, 3, 1048576, 1, 1, "叶节点接收数据"
    );
    
    ncclFlowCollector::getInstance()->finalizeFlow();
    ncclFlowCollector::getInstance()->printFlowInfo();
    ncclFlowCollector::getInstance()->saveToFile("tree_flow_test.log");
    
    printf("\n=== 测试3: NVLS算法 + LL协议 ===\n");
    
    // 重新初始化
    ncclFlowCollector::getInstance()->initFlow(comm, 2, 4194304, 2); // AllGather, 4MB数据
    
    ncclFlowCollector::getInstance()->setAlgorithmInfo(
        4,      // NVLS算法
        0,      // LL协议
        8,      // 8个通道
        640,    // 640个线程
        32768,  // 32KB块大小
        25.0,   // 25 GB/s带宽
        2.0,    // 2us延迟
        "使用NVLS算法充分利用NVLink带宽，LL协议确保低延迟"
    );
    
    // 添加NVLS算法的流步骤
    for (int ch = 0; ch < 8; ch++) {
        ncclFlowCollector::getInstance()->addFlowStep(
            FLOW_STEP_SEND, ch % 4, (ch + 1) % 4, 524288, ch, 0, "NVLS并行传输"
        );
        ncclFlowCollector::getInstance()->addFlowStep(
            FLOW_STEP_RECV, (ch + 3) % 4, ch % 4, 524288, ch, 0, "NVLS并行接收"
        );
    }
    
    ncclFlowCollector::getInstance()->finalizeFlow();
    ncclFlowCollector::getInstance()->printFlowInfo();
    ncclFlowCollector::getInstance()->saveToFile("nvls_flow_test.log");
    
    printf("\n=== 所有测试完成 ===\n");
    printf("生成的日志文件:\n");
    printf("  - improved_flow_test.log (RING + SIMPLE)\n");
    printf("  - tree_flow_test.log (TREE + LL128)\n");
    printf("  - nvls_flow_test.log (NVLS + LL)\n");
    
    return 0;
}
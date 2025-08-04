#!/bin/bash

echo "=== NCCL流信息提取工具编译和测试脚本 ==="

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查必要的依赖
echo -e "${YELLOW}步骤1: 检查编译环境...${NC}"

# 检查编译器
if ! command -v g++ &> /dev/null; then
    echo -e "${RED}错误: 未找到g++编译器${NC}"
    exit 1
fi

# 检查CUDA（如果可用）
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}找到CUDA编译器: $(nvcc --version | grep release)${NC}"
    CUDA_AVAILABLE=1
else
    echo -e "${YELLOW}警告: 未找到CUDA编译器，将使用CPU模拟模式${NC}"
    CUDA_AVAILABLE=0
fi

echo -e "${GREEN}编译环境检查完成${NC}"
echo ""

# 编译NCCL库（如果需要）
echo -e "${YELLOW}步骤2: 编译NCCL库...${NC}"

if [ ! -f "build/lib/libnccl.so" ] && [ ! -f "build/lib/libnccl.a" ]; then
    echo "开始编译NCCL库..."
    cd src
    if make -j$(nproc) 2>/dev/null; then
        echo -e "${GREEN}NCCL库编译成功${NC}"
    else
        echo -e "${YELLOW}NCCL库编译失败，尝试使用现有库文件${NC}"
    fi
    cd ..
else
    echo -e "${GREEN}找到现有NCCL库文件${NC}"
fi

echo ""

# 编译流信息提取模块
echo -e "${YELLOW}步骤3: 编译流信息提取模块...${NC}"

# 创建简化的编译命令
INCLUDES="-Isrc/include -I/usr/local/cuda/include"
CXXFLAGS="-std=c++11 -O2 -g -Wall -fPIC"

# 编译流信息模块
echo "编译 flow_info.cc..."
if g++ $CXXFLAGS $INCLUDES -c src/flow_info.cc -o src/flow_info.o; then
    echo -e "${GREEN}flow_info.cc 编译成功${NC}"
else
    echo -e "${RED}flow_info.cc 编译失败${NC}"
    exit 1
fi

echo ""

# 创建简化的测试程序
echo -e "${YELLOW}步骤4: 创建简化测试程序...${NC}"

cat > simple_test.cc << 'EOF'
#include <stdio.h>
#include <string.h>
#include "src/include/flow_info.h"

int main(int argc, char* argv[]) {
    printf("=== NCCL流信息提取工具简化测试 ===\n");
    
    // 解析参数
    int nRanks = (argc > 1) ? atoi(argv[1]) : 4;
    size_t dataSize = (argc > 2) ? atoi(argv[2]) : 1024;
    const char* collType = (argc > 3) ? argv[3] : "allreduce";
    
    printf("测试参数:\n");
    printf("  节点数: %d\n", nRanks);
    printf("  数据大小: %zu bytes\n", dataSize);
    printf("  集合通信类型: %s\n", collType);
    printf("\n");
    
    // 启用流信息收集
    ncclFlowCollector::getInstance()->enable();
    printf("流信息收集已启用\n");
    
    // 模拟NCCL初始化和算法选择过程
    printf("\n=== 模拟NCCL算法选择过程 ===\n");
    
    // 模拟算法选择
    int algorithm = 1; // NCCL_ALGO_RING
    int protocol = 0;  // NCCL_PROTO_SIMPLE
    int nChannels = (nRanks <= 4) ? 2 : 4;
    int nThreads = 256;
    size_t chunkSize = 131072; // 128KB
    float bandwidth = 10.0; // 10 GB/s
    float latency = 5.0; // 5 us
    
    char reason[512];
    snprintf(reason, sizeof(reason), 
             "基于节点数%d和数据大小%zu选择Ring算法", nRanks, dataSize);
    
    FLOW_INFO_SET_ALGORITHM(algorithm, protocol, nChannels, nThreads, 
                           chunkSize, bandwidth, latency, reason);
    
    // 模拟流生成过程
    printf("模拟流生成过程...\n");
    for (int c = 0; c < nChannels; c++) {
        for (int step = 0; step < 3; step++) {
            char stepInfo[256];
            snprintf(stepInfo, sizeof(stepInfo), 
                     "通道%d步骤%d: 发送%zu字节到下一个节点", 
                     c, step, dataSize / nChannels);
            FLOW_INFO_ADD_STEP(c, step, stepInfo);
        }
    }
    
    // 输出结果
    printf("\n=== 流信息提取结果 ===\n");
    ncclFlowCollector::getInstance()->printFlowInfo();
    
    // 保存到文件
    char logFile[256];
    snprintf(logFile, sizeof(logFile), "nccl_flow_%s_%d_%zu.log", 
             collType, nRanks, dataSize);
    ncclFlowCollector::getInstance()->saveToFile(logFile);
    printf("\n流信息已保存到: %s\n", logFile);
    
    printf("\n=== 测试完成 ===\n");
    return 0;
}
EOF

echo -e "${GREEN}简化测试程序创建完成${NC}"

# 编译测试程序
echo "编译测试程序..."
if g++ $CXXFLAGS $INCLUDES simple_test.cc src/flow_info.o -o simple_test; then
    echo -e "${GREEN}测试程序编译成功${NC}"
else
    echo -e "${RED}测试程序编译失败${NC}"
    exit 1
fi

echo ""

# 运行测试
echo -e "${YELLOW}步骤5: 运行测试用例...${NC}"

echo "测试1: AllReduce, 4节点, 1KB数据"
./simple_test 4 1024 allreduce
echo ""

echo "测试2: AllGather, 8节点, 4KB数据"
./simple_test 8 4096 allgather
echo ""

echo "测试3: Broadcast, 2节点, 512B数据"
./simple_test 2 512 broadcast
echo ""

echo -e "${GREEN}所有测试完成！${NC}"
echo ""

# 显示生成的日志文件
echo -e "${YELLOW}生成的日志文件:${NC}"
ls -la nccl_flow_*.log 2>/dev/null || echo "未找到日志文件"

echo ""
echo -e "${GREEN}=== 编译和测试脚本执行完成 ===${NC}"
#!/bin/bash

# NCCL流信息提取工具 Linux编译和测试脚本
# 适用于Ubuntu/Debian/CentOS/RHEL等Linux发行版

set -e  # 遇到错误立即退出

echo "=== NCCL流信息提取工具 Linux编译测试 ==="
echo

# 检查编译环境
echo "步骤1: 检查编译环境..."

# 检查是否有C++编译器
if command -v g++ >/dev/null 2>&1; then
    echo "✓ 找到g++编译器: $(g++ --version | head -n1)"
elif command -v clang++ >/dev/null 2>&1; then
    echo "✓ 找到clang++编译器: $(clang++ --version | head -n1)"
    export CXX=clang++
else
    echo "✗ 错误: 未找到C++编译器"
    echo "请安装编译工具:"
    echo "  Ubuntu/Debian: sudo apt install build-essential"
    echo "  CentOS/RHEL:   sudo yum groupinstall 'Development Tools'"
    echo "  Fedora:        sudo dnf groupinstall 'Development Tools'"
    exit 1
fi

# 检查make工具
if ! command -v make >/dev/null 2>&1; then
    echo "✗ 错误: 未找到make工具"
    echo "请安装make工具"
    exit 1
fi

echo "✓ 编译环境检查完成"
echo

# 创建必要的目录
echo "步骤2: 创建构建目录..."
mkdir -p build/obj
mkdir -p build/lib
echo "✓ 构建目录创建完成"
echo

# 编译NCCL源文件
echo "步骤3: 编译NCCL源文件..."

# 设置编译参数
CXXFLAGS="-std=c++11 -O2 -fPIC -Wall"
INCLUDES="-I./src/include -I./src"

echo "编译 flow_info.cc..."
${CXX:-g++} $CXXFLAGS $INCLUDES -c src/flow_info.cc -o build/obj/flow_info.o

if [ -f "src/enqueue.cc" ]; then
    echo "编译 enqueue.cc..."
    ${CXX:-g++} $CXXFLAGS $INCLUDES -c src/enqueue.cc -o build/obj/enqueue.o 2>/dev/null || echo "⚠ enqueue.cc编译跳过(可能需要完整NCCL环境)"
fi

if [ -f "src/collectives/all_reduce.cc" ]; then
    echo "编译 all_reduce.cc..."
    ${CXX:-g++} $CXXFLAGS $INCLUDES -c src/collectives/all_reduce.cc -o build/obj/all_reduce.o 2>/dev/null || echo "⚠ all_reduce.cc编译跳过(可能需要完整NCCL环境)"
fi

echo "✓ 核心文件编译完成"
echo

# 编译测试程序
echo "步骤4: 编译测试程序..."

# 编译改进的测试程序
if [ -f "test_improved_flow.cpp" ]; then
    echo "编译改进测试程序..."
    ${CXX:-g++} $CXXFLAGS $INCLUDES test_improved_flow.cpp build/obj/flow_info.o -o flow_test_improved
    echo "✓ 改进测试程序编译完成: ./flow_test_improved"
fi

# 编译简化测试程序
if [ -f "simple_flow_test.cpp" ]; then
    echo "编译简化测试程序..."
    ${CXX:-g++} $CXXFLAGS $INCLUDES simple_flow_test.cpp build/obj/flow_info.o -o flow_test_simple 2>/dev/null || echo "⚠ 简化测试程序编译跳过"
fi

# 编译完整测试程序
if [ -f "test_flow_info.cc" ]; then
    echo "编译完整测试程序..."
    ${CXX:-g++} $CXXFLAGS $INCLUDES test_flow_info.cc build/obj/flow_info.o -o flow_test_full 2>/dev/null || echo "⚠ 完整测试程序编译跳过"
fi

echo "✓ 测试程序编译完成"
echo

# 运行测试
echo "步骤5: 运行功能测试..."

if [ -f "./flow_test_improved" ]; then
    echo "运行改进测试程序..."
    echo "----------------------------------------"
    ./flow_test_improved
    echo "----------------------------------------"
    echo "✓ 改进测试程序运行完成"
    
    # 检查生成的日志文件
    echo
    echo "生成的日志文件:"
    ls -la *.log 2>/dev/null || echo "未找到日志文件"
    echo
else
    echo "⚠ 改进测试程序未编译成功，跳过测试"
fi

# 显示编译结果
echo "步骤6: 编译结果总结..."
echo "编译完成的程序:"
[ -f "./flow_test_improved" ] && echo "  ✓ ./flow_test_improved - 改进测试程序(推荐)"
[ -f "./flow_test_simple" ] && echo "  ✓ ./flow_test_simple - 简化测试程序"
[ -f "./flow_test_full" ] && echo "  ✓ ./flow_test_full - 完整测试程序"

echo
echo "编译的目标文件:"
ls -la build/obj/*.o 2>/dev/null || echo "  无目标文件"

echo
echo "=== 编译测试完成 ==="
echo
echo "使用方法:"
echo "  运行改进测试: ./flow_test_improved"
echo "  查看日志文件: cat *.log"
echo "  清理构建文件: rm -rf build/ *.log flow_test_*"
echo
echo "如遇到问题，请查看 编译问题修复说明.md 文档"
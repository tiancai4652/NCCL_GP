# NCCL流信息提取工具编译测试脚本 (PowerShell版本)

Write-Host "=== NCCL流信息提取工具编译测试 ===" -ForegroundColor Green

# 检查编译环境
Write-Host "步骤1: 检查编译环境..." -ForegroundColor Yellow

# 检查是否有MinGW或MSVC编译器
$compilerFound = $false
$compiler = ""

# 检查MinGW g++
try {
    $gccVersion = & g++ --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "找到MinGW g++编译器" -ForegroundColor Green
        $compiler = "g++"
        $compilerFound = $true
    }
} catch {
    # g++不可用
}

# 检查MSVC cl
if (-not $compilerFound) {
    try {
        $clVersion = & cl 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "找到MSVC编译器" -ForegroundColor Green
            $compiler = "cl"
            $compilerFound = $true
        }
    } catch {
        # cl不可用
    }
}

if (-not $compilerFound) {
    Write-Host "错误: 未找到可用的C++编译器 (g++ 或 cl)" -ForegroundColor Red
    Write-Host "请安装MinGW或Visual Studio" -ForegroundColor Red
    exit 1
}

Write-Host "编译环境检查完成" -ForegroundColor Green
Write-Host ""

# 创建简化的测试程序
Write-Host "步骤2: 创建简化测试程序..." -ForegroundColor Yellow

$testCode = @"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>

// 简化的流信息结构
struct SimpleFlowInfo {
    int algorithm;
    int protocol;
    int nChannels;
    int nThreads;
    size_t chunkSize;
    float bandwidth;
    float latency;
    char reason[512];
    std::vector<std::string> steps;
};

// 简化的流信息收集器
class SimpleFlowCollector {
private:
    static SimpleFlowCollector* instance;
    SimpleFlowInfo flowInfo;
    bool enabled;
    
public:
    static SimpleFlowCollector* getInstance() {
        if (!instance) {
            instance = new SimpleFlowCollector();
        }
        return instance;
    }
    
    void enable() { enabled = true; }
    bool isEnabled() const { return enabled; }
    
    void setAlgorithmInfo(int alg, int proto, int nCh, int nTh, 
                         size_t chunk, float bw, float lat, const char* reason) {
        if (!enabled) return;
        flowInfo.algorithm = alg;
        flowInfo.protocol = proto;
        flowInfo.nChannels = nCh;
        flowInfo.nThreads = nTh;
        flowInfo.chunkSize = chunk;
        flowInfo.bandwidth = bw;
        flowInfo.latency = lat;
        if (reason) {
            strncpy(flowInfo.reason, reason, sizeof(flowInfo.reason) - 1);
            flowInfo.reason[sizeof(flowInfo.reason) - 1] = '\0';
        }
    }
    
    void addStep(int channel, int stepId, const char* desc) {
        if (!enabled) return;
        char stepStr[256];
        snprintf(stepStr, sizeof(stepStr), "通道%d步骤%d: %s", channel, stepId, desc);
        flowInfo.steps.push_back(std::string(stepStr));
    }
    
    void printFlowInfo() {
        if (!enabled) {
            printf("流信息收集未启用\n");
            return;
        }
        
        printf("\n=== NCCL流信息提取结果 ===\n");
        printf("算法选择信息:\n");
        printf("  算法: %d\n", flowInfo.algorithm);
        printf("  协议: %d\n", flowInfo.protocol);
        printf("  通道数: %d\n", flowInfo.nChannels);
        printf("  线程数: %d\n", flowInfo.nThreads);
        printf("  块大小: %zu bytes\n", flowInfo.chunkSize);
        printf("  预期带宽: %.2f GB/s\n", flowInfo.bandwidth);
        printf("  预期延迟: %.2f us\n", flowInfo.latency);
        printf("  选择原因: %s\n", flowInfo.reason);
        
        printf("\n流执行步骤:\n");
        for (size_t i = 0; i < flowInfo.steps.size(); i++) {
            printf("  %s\n", flowInfo.steps[i].c_str());
        }
        printf("========================\n");
    }
    
    void saveToFile(const char* filename) {
        if (!enabled) return;
        
        FILE* file = fopen(filename, "w");
        if (!file) {
            printf("错误: 无法创建日志文件 %s\n", filename);
            return;
        }
        
        fprintf(file, "# NCCL流信息提取日志\n");
        fprintf(file, "算法=%d\n", flowInfo.algorithm);
        fprintf(file, "协议=%d\n", flowInfo.protocol);
        fprintf(file, "通道数=%d\n", flowInfo.nChannels);
        fprintf(file, "线程数=%d\n", flowInfo.nThreads);
        fprintf(file, "块大小=%zu\n", flowInfo.chunkSize);
        fprintf(file, "预期带宽=%.2f\n", flowInfo.bandwidth);
        fprintf(file, "预期延迟=%.2f\n", flowInfo.latency);
        fprintf(file, "选择原因=%s\n", flowInfo.reason);
        
        for (size_t i = 0; i < flowInfo.steps.size(); i++) {
            fprintf(file, "步骤: %s\n", flowInfo.steps[i].c_str());
        }
        
        fclose(file);
        printf("流信息已保存到: %s\n", filename);
    }
    
private:
    SimpleFlowCollector() : enabled(false) {
        memset(&flowInfo, 0, sizeof(flowInfo));
    }
};

SimpleFlowCollector* SimpleFlowCollector::instance = nullptr;

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
    SimpleFlowCollector::getInstance()->enable();
    printf("流信息收集已启用\n");
    
    // 模拟NCCL算法选择过程
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
    
    SimpleFlowCollector::getInstance()->setAlgorithmInfo(
        algorithm, protocol, nChannels, nThreads, 
        chunkSize, bandwidth, latency, reason);
    
    // 模拟流生成过程
    printf("模拟流生成过程...\n");
    for (int c = 0; c < nChannels; c++) {
        for (int step = 0; step < 3; step++) {
            char stepDesc[256];
            snprintf(stepDesc, sizeof(stepDesc), 
                     "发送%zu字节到下一个节点", dataSize / nChannels);
            SimpleFlowCollector::getInstance()->addStep(c, step, stepDesc);
        }
    }
    
    // 输出结果
    SimpleFlowCollector::getInstance()->printFlowInfo();
    
    // 保存到文件
    char logFile[256];
    snprintf(logFile, sizeof(logFile), "nccl_flow_%s_%d_%zu.log", 
             collType, nRanks, dataSize);
    SimpleFlowCollector::getInstance()->saveToFile(logFile);
    
    printf("\n=== 测试完成 ===\n");
    return 0;
}
"@

$testCode | Out-File -FilePath "simple_test.cpp" -Encoding UTF8

Write-Host "简化测试程序创建完成" -ForegroundColor Green

# 编译测试程序
Write-Host "步骤3: 编译测试程序..." -ForegroundColor Yellow

if ($compiler -eq "g++") {
    $compileCmd = "g++ -std=c++11 -O2 -Wall simple_test.cpp -o simple_test.exe"
} else {
    $compileCmd = "cl /EHsc /std:c++11 simple_test.cpp /Fe:simple_test.exe"
}

Write-Host "执行编译命令: $compileCmd" -ForegroundColor Cyan

try {
    Invoke-Expression $compileCmd
    if ($LASTEXITCODE -eq 0) {
        Write-Host "测试程序编译成功" -ForegroundColor Green
    } else {
        Write-Host "测试程序编译失败" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "编译过程中出现错误: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 运行测试
Write-Host "步骤4: 运行测试用例..." -ForegroundColor Yellow

$testCases = @(
    @{nodes=4; size=1024; type="allreduce"},
    @{nodes=8; size=4096; type="allgather"},
    @{nodes=2; size=512; type="broadcast"}
)

foreach ($test in $testCases) {
    Write-Host "测试: $($test.type), $($test.nodes)节点, $($test.size)B数据" -ForegroundColor Cyan
    try {
        & .\simple_test.exe $test.nodes $test.size $test.type
        Write-Host ""
    } catch {
        Write-Host "测试执行失败: $_" -ForegroundColor Red
    }
}

Write-Host "所有测试完成！" -ForegroundColor Green
Write-Host ""

# 显示生成的日志文件
Write-Host "生成的日志文件:" -ForegroundColor Yellow
Get-ChildItem -Name "nccl_flow_*.log" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  $_" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "=== 编译和测试脚本执行完成 ===" -ForegroundColor Green
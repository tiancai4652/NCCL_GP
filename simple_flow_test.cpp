#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

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
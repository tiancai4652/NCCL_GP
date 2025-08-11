### NCCL_GP Flow Extractor 使用说明（README2）

本说明文档记录在 NCCL_GP 基础上新增的“流信息提取”改动、编译步骤、运行方法以及可以获得的输出信息。目标是在无需真实硬件的环境下，直接复用 NCCL 的拓扑感知、算法/协议选择与通信计划，导出可供网络仿真器消费的有序流信息。

## 1. 新增与修改概览

- 新增文件
  - `src/flow_extractor.h`：对外导出的 Flow Extractor API 与数据结构声明。
  - `src/flow_extractor.cc`：流信息提取实现，包括算法/协议选择复用、pattern/loop 推导、以及从 NCCL 真实 `ncclProxyOp` 记录通信步骤。
- 关键修改
  - `src/enqueue.cc`
    - 在 `addProxyOpIfNeeded` 中，当 proxy 确认需要入队时，调用 `ncclRecordProxyOp(...)` 记录一条真实 `ncclProxyOp`（去重后仅在“真实入队”时记录）。
    - 去除了预记录与队列头部重复记录，保证每个入队 `proxyOp` 仅记录一次。
  - `src/Makefile`：确保 `flow_extractor.cc` 编译进 `libnccl.so`（已有）。
  - `test/Makefile`：编译 `test_flow_extractor.cpp`、`test_flow_simple.cpp`（已有）。
- 新增/导出 API（位于 `src/flow_extractor.h`）
  - `ncclSetFlowExtractionEnabled(int enable)`：启用/禁用流提取。
  - `ncclGetCollectiveFlow(ncclFunc_t collType, size_t count, ncclDataType_t dataType, int root, ncclComm_t comm, ncclCollectiveFlow** flow)`：复用 NCCL 的算法/协议选择与模式推导，生成模式化的“流信息”（非真实执行，仅描述）。
  - `ncclFlowToJson(ncclCollectiveFlow* flow, char** jsonStr)`、`ncclFreeCollectiveFlow(...)`：JSON 输出与释放。
  - `ncclRecordProxyOp(const ncclInfo* info, const ncclProxyOp* proxyOp, ncclComm* comm)`：由 NCCL 入队路径调用，落盘真实 `proxyOp` 摘要与逐步展开条目。
  - `ncclWriteAggregatedFlow(ncclComm* comm)`：将本 rank 的 `proxy_flow_rank<rank>.jsonl` 与 `flow_steps_rank<rank>.jsonl` 聚合为 `flow_rank<rank>.json`。

## 2. 编译

- 快速方式
  - 在项目根目录 `NCCL_GP/` 下执行：
    ```bash
    make -j4
    ```
  - 产物：`build/lib/libnccl.so.*` 与测试可执行文件 `test/test_flow_extractor`、`test/test_flow_simple`。

- 环境变量（运行时需设置）
  - `NCCL_TOPO_FILE`：指定拓扑 XML 路径（必须），示例：`NCCL_GP/topo/nvlink_5GPU.xml`
  - `NCCL_GRAPH_DUMP_FILE`：图导出文件路径（可选）
  - `LD_LIBRARY_PATH`：包含 `NCCL_GP/build/lib`（必须）
  - `GPU_DEV_NUM`：模拟 GPU 数量（与 XML 保持一致），示例：`5`
  - `NCCL_DEBUG`：推荐 `WARN` 或 `INFO`

## 3. 运行（示例）

以 5 个设备为例，运行综合测试并输出流信息：

```bash
cd NCCL_GP/test
export LD_LIBRARY_PATH=$(pwd)/../build/lib:$LD_LIBRARY_PATH
export NCCL_TOPO_FILE=$(pwd)/../topo/nvlink_5GPU.xml
export NCCL_GRAPH_DUMP_FILE=$(pwd)/../topo/graph_dump.xml
export GPU_DEV_NUM=5
export NCCL_DEBUG=WARN
./test_flow_extractor
```

执行完成后，当前目录会生成以下文件：
- `flow_AllReduce_rank0.json` 等：通过 `ncclGetCollectiveFlow` 输出的“模式化流信息”（非真实执行，用于直观理解算法/模式/通道/步数等）。
- `proxy_flow_rank<rank>.jsonl`：每个 rank 的 `proxyOp` 摘要，每行一条，字段包括：
  - `opCount, rank, channel, nsteps, nbytes, chunkSize, sliceSteps, chunkSteps, dtype, redOp, pattern, protocol, ringPrev, ringNext`
  - 这些字段均来自 NCCL 的真实 `ncclProxyOp` 与通道拓扑，不做估算或重推导。
- `flow_steps_rank<rank>.jsonl`：每个 rank 的逐步展开流记录。按 `proxyOp` 的 `nsteps` 展开为每步两条记录（SEND/RECV），字段：
  - `opCount, rank, channel, step, op, peer, bytes, pattern, protocol, stage`
  - `peer` 与阶段（如 `reduce-scatter`/`allgather`）根据 `pattern` 与环/树邻居机械展开，保持与 NCCL 规划一致。
- `flow_rank<rank>.json`：聚合视图（包含一条 `meta` 摘要与完整 `steps` 列表），便于一次性加载。

## 4. 开发者说明（数据来源与准确性）

- 算法/协议/模式（algorithm/protocol/pattern）：由 NCCL 内部选择逻辑决定，选择过程会调用 `ncclTopoGetAlgoTime` 进行代价评估，但最终选择结果是 NCCL 的真实决策。
- 通道/拓扑（channel、ringPrev/ringNext、树结构）：直接读取自 `ncclComm->channels[...]`，与加载的 XML 拓扑一致。
- `ncclProxyOp` 字段（nsteps、chunkSize、sliceSteps、chunkSteps、nbytes、dtype、redOp、pattern、protocol 等）：由 NCCL 内部 `computeColl()` 计算并填充，我们仅记录；未做任何估算。
- 逐步展开（`flow_steps_rank*.jsonl`）：根据 `ncclProxyOp` 与通道邻居（Ring/Tree/CollNet/NVLS）机械展开，保持一致性；当前已去除“预记录”与“入队后重复记录”，仅在真实入队时记录。
- 已移除了所有“估计时间”字段，避免混淆。

## 5. 编程接口（嵌入到你自己的程序）

- 启用/禁用
  - `ncclSetFlowExtractionEnabled(int enable);`
- 获取模式化流信息（非执行，仅描述）
  - `ncclGetCollectiveFlow(ncclFunc_t collType, size_t count, ncclDataType_t dataType, int root, ncclComm_t comm, ncclCollectiveFlow** flow);`
  - `ncclFlowToJson(...)` / `ncclFreeCollectiveFlow(...)`
- 真实执行路径上的记录（无需手动调用）
  - `ncclRecordProxyOp(...)` 由 `enqueue.cc` 的 `addProxyOpIfNeeded` 在 `proxyOp` 入队时自动调用。
- 聚合输出
  - `ncclWriteAggregatedFlow(ncclComm* comm);` 生成 `flow_rank<rank>.json`

## 6. 常见问题

- 运行崩溃/段错误
  - 请确认设置了 `NCCL_TOPO_FILE` 指向有效的 XML（示例：`topo/nvlink_5GPU.xml`）。
  - `LD_LIBRARY_PATH` 必须包含 `NCCL_GP/build/lib`。
  - `GPU_DEV_NUM` 与拓扑中的 GPU 数目一致。
- 无 `proxy_flow_rank*.jsonl` 输出
  - 需要执行至少一次真实 NCCL 集合通信（测试程序已自动调用一次 AllReduce）。
- 输出字段含义
  - `pattern`/`protocol` 与 NCCL 保持一致；Ring/Tree/CollNet/NVLS 的邻居关系可从通道结构推导（我们在 JSONL 中已展开为 `ringPrev/ringNext` 或在逐步记录中直接给出 `peer`）。

## 7. 目录与文件（本次新增/修改关注）

- `src/flow_extractor.h/.cc`（新增）
- `src/enqueue.cc`（插桩：真实入队记录）
- `test/test_flow_extractor.cpp`（综合测试：静态提取 + 真实执行记录 + 聚合输出）
- 运行输出：`proxy_flow_rank*.jsonl`、`flow_steps_rank*.jsonl`、`flow_rank*.json`、`flow_*.json`

## 8. 许可与致谢

- 基于 NCCL_GP（NCCL 2.19.1 修改版）的无硬件调试思路。
- Flow Extractor 保持与 NCCL 内部决策一致，尽量减少重实现；仅做日志记录与格式化导出。 
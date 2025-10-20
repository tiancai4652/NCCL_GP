### NCCL_GP Flow Extractor 使用说明（README2）

本说明文档记录在 NCCL_GP 基础上新增的“流信息提取”改动、编译步骤、运行方法以及可以获得的输出信息。目标是在无需真实硬件的环境下，直接复用 NCCL 的拓扑感知、算法/协议选择与通信计划，导出可供网络仿真器消费的有序且真实的流信息。

重要说明：已经完全移除任何“复制/简化 NCCL 逻辑”的模式化路径。用于仿真的数据100%来源于 NCCL 的真实执行路径（proxy 入队的 ncclProxyOp）；我们只做记录与序列化。

## 1. 新增与修改概览

- 新增文件
  - `src/flow_extractor.h`：Flow Extractor API 与声明。
  - `src/flow_extractor.cc`：流信息提取实现（仅记录与聚合，无估算）。
- 关键修改
  - `src/enqueue.cc`
    - 在 `addProxyOpIfNeeded` 中，当 proxy 确认需要入队时，调用 `ncclRecordProxyOp(...)` 记录真实 `ncclProxyOp`（仅在真实入队时记录，已去重）。
  - `src/proxy.cc`
    - 在 `SaveProxy` 路径插桩，调用 `ncclRecordProxyPeerSteps(...)` 对逐 peer 的 SEND/RECV 步骤进行记录，已覆盖 Ring/RingTwice、Pipeline(From/To)、Tree(Up/Down/UpDown)、CollNet(Chain/Direct)、NVLS/NVLS_TREE 等模式的阶段标签与对端。
  - `src/Makefile`：确保 `flow_extractor.cc` 编译进 `libnccl.so`。
  - `test/Makefile`：编译 `test_flow_extractor.cpp`。
- 导出 API（`src/flow_extractor.h`）
  - `ncclSetFlowExtractionEnabled(int enable)`: 启用/禁用流提取
  - `ncclRecordProxyOp(...)`: 由 `enqueue.cc` 在真实 proxy 入队时自动调用（用户无需手动调）
  - `ncclWriteAggregatedFlow(ncclComm* comm)`: 聚合当前 rank 的记录，输出单文件视图
  - `ncclExtractFlow(collType, count, dataType, root, comm)`: 权威提取接口（直接调用 NCCL 集合通信触发完整链路后聚合输出）
  - 说明：`ncclRecordProxyPeerSteps(...)` 为内部记录函数（由 `proxy.cc` 调用），一般无需在业务代码中显式调用。

已移除：`ncclGetCollectiveFlow`/`ncclFlowToJson` 等所有“模式化/估算式”接口与实现。

## 2. 编译

- 快速方式
  ```bash
  make -j4
  ```
- 产物
  - `build/lib/libnccl.so.*`
  - 测试可执行：`test/test_flow_extractor`

## 3. 运行（示例）

以 5 个设备为例，运行一次真实 AllReduce 以触发 NCCL 真实逻辑并输出流信息：

```bash
cd NCCL_GP/test
export LD_LIBRARY_PATH=$(pwd)/../build/lib:$LD_LIBRARY_PATH
export NCCL_TOPO_FILE=$(pwd)/../topo/nvlink_5GPU.xml
export NCCL_GRAPH_DUMP_FILE=$(pwd)/../topo/graph_dump.xml   # 可选
export GPU_DEV_NUM=5
export NCCL_DEBUG=WARN
./test_flow_extractor
```

执行完成后，输出位于当前运行目录的 `output/<topo_base>/`，例如 `output/nvlink_5GPU/`。

## 4. 输出文件与含义

- `proxy_flow_rank<rank>.jsonl`（每行一个入队成功的 proxyOp 摘要）：
  - `opCount, rank, channel, nsteps, nbytes, chunkSize, sliceSteps, chunkSteps, dtype, redOp, pattern, protocol, ringPrev, ringNext`
  - 字段全部来源于 NCCL 的真实 `ncclProxyOp` 与通道拓扑。
- `flow_steps_rank<rank>.jsonl`（逐步展开，直接可用于仿真）：
  - 每步两条（SEND/RECV）；字段：`opCount, rank, channel, step, op, peer, bytes, pattern, protocol, stage`
  - `peer` 与 `stage` 已覆盖以下模式：
    - Ring、RingTwice（环前后半程标记为 `reduce-scatter`/`allgather`）
    - PipelineFrom/PipelineTo（标记为 `pipeline-from`/`pipeline-to`）
    - TreeUp/TreeDown/TreeUpDown（标记为 `tree-up`/`tree-down`）
    - CollNetChain/CollNetDirect（标记为 `collnet-chain`/`collnet-direct`）
    - NVLS、NVLS_TREE（标记为 `nvls`/`nvls-tree`）
- `flow_rank<rank>.json`（聚合视图）：
  - `rank`、`meta`（来自 proxy 摘要第一条）、`steps`（完整逐步列表）

目录层级：`output/<topo_base>/...`，`<topo_base>` 自动取自 `NCCL_TOPO_FILE` 文件名（无扩展名）。

## 5. 编程接口（最小用法）

- 启用提取
  ```c
  ncclSetFlowExtractionEnabled(1);
  ```
- 触发一次集合通信（示例：AllReduce），随后聚合输出
  ```c
  ncclGroupStart();
  for (int i = 0; i < nDev; ++i) {
    ncclAllReduce(NULL, NULL, count, ncclFloat, ncclSum, comms[i], (cudaStream_t)0);
  }
  ncclGroupEnd();
  ncclWriteAggregatedFlow(comms[0]);
  ```
- 或使用权威提取 API（单次调用完成触发+聚合）
  ```c
  ncclExtractFlow(ncclFuncAllReduce, count, ncclFloat, /*root*/0, comm);
  ```

## 6. 数据来源与准确性

- 算法/协议/模式、通道与邻居、`ncclProxyOp` 字段（nsteps/chunk/slice/nbytes/dtype/redOp/pattern/protocol）全部来自 NCCL 真实执行路径。
- 我们不做时间或数据量估算；只负责记录/展开/聚合。
- 逐步对端与阶段标签现已覆盖：Ring/RingTwice、PipelineFrom/To、TreeUp/Down/UpDown、CollNetChain/Direct、NVLS/NVLS_TREE。
- 极个别协议/拓扑若无需 proxy，可能不生成 `proxy_flow_*` 与 `flow_steps_*`；此时仍可通过再次触发其它集合通信获取其它流记录。

## 7. 常见问题

- 段错误/启动失败：确认 `NCCL_TOPO_FILE`、`LD_LIBRARY_PATH`、`GPU_DEV_NUM` 设置正确。
- 无输出文件：需要至少一次真实的 NCCL 集合通信以触发 proxy 入队（`test_flow_extractor` 已自动调用一次 AllReduce）。
- 输出路径：相对于运行目录创建 `output/<topo_base>/`；如需固定到项目根，可在根目录运行或调整实现。

## 8. 目录与文件（本次新增/修改关注）

- `src/flow_extractor.h/.cc`（新增，记录+聚合+权威提取）
- `src/enqueue.cc`（插桩：真实入队记录）
- `src/proxy.cc`（插桩：逐 peer 步级记录）
- `test/test_flow_extractor.cpp`（最小触发 + 聚合）
- 输出：`output/<topo_base>/proxy_flow_rank*.jsonl`、`flow_steps_rank*.jsonl`、`flow_rank*.json` 
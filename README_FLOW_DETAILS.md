### Flow Extractor 代码与输出详解（详细说明）

本文件详细说明“流信息提取”相关代码的作用、记录的数据、字段含义，以及生成文件的解释。用于仿真的数据全部来源于 NCCL 的真实执行路径（proxy 入队的 `ncclProxyOp`）；我们不复制/简化 NCCL 逻辑、不做估算。

## 1. 代码总览与职责边界

- `src/enqueue.cc`
  - 负责构建 NCCL 的 kernel 计划与 proxy 入队。
  - 真实计算与填充 `ncclProxyOp` 字段发生在 NCCL 内部（如 `computeColl()`）。
  - 插桩位置：`addProxyOpIfNeeded(struct ncclComm*, struct ncclKernelPlan*, struct ncclProxyOp* op)`
    - 当 `ncclProxySaveOp` 判断“需要 proxy”并成功入队后，调用 `ncclRecordProxyOp(&infoCtx, q, comm)` 进行记录。

- `src/flow_extractor.h`
  - 导出 API：
    - `ncclSetFlowExtractionEnabled(int enable)`：启用/禁用提取
    - `ncclRecordProxyOp(...)`：由 `enqueue.cc` 在真实入队时自动调用（用户无需手动调）
    - `ncclWriteAggregatedFlow(ncclComm*)`：聚合当前 rank 的记录输出单文件视图
    - `ncclExtractFlow(...)`：权威提取（内部直接调用 NCCL 集合通信，触发完整选择/规划/入队链路后聚合输出）
  - 字符串辅助：`ncclAlgorithmToString`/`ncclProtocolToString`/`ncclPatternToString`/`ncclFlowOpTypeToString`

- `src/flow_extractor.cc`
  - `ncclRecordProxyOp(...)`：
    - 写入 `output/<topo_base>/proxy_flow_rank<rank>.jsonl`（摘要，一行一个 `proxyOp`）
    - 写入 `output/<topo_base>/flow_steps_rank<rank>.jsonl`（逐步展开，每步两条 SEND/RECV）
  - `ncclWriteAggregatedFlow(...)`：
    - 读取上述两个 JSONL，生成 `output/<topo_base>/flow_rank<rank>.json`（包含 `meta` 与完整 `steps`）
  - `ncclExtractFlow(...)`：
    - 直接调用 NCCL 集合通信（AllReduce/AllGather/ReduceScatter/Broadcast/Reduce）以触发完整 NCCL 逻辑，然后写出聚合文件

已移除：任何“模式化/估算式”的接口与实现，避免复制/简化 NCCL 逻辑导致语义偏差。

## 2. 记录的数据与字段来源

- 真实来源（非估算）：
  - `algorithm/protocol/pattern`：NCCL 内部真实选择（内部通过 `ncclTopoGetAlgoTime` 做代价评估）
  - 通道/邻居：`ncclComm->channels[chan]`（ring.prev/next，tree.up/down 等）
  - `ncclProxyOp` 字段：`nsteps, nbytes, chunkSize, sliceSteps, chunkSteps, dtype, redOp, pattern, protocol, root`
- 我们只做记录与序列化：
  - 入队成功即记录（避免预记录与重复记录）
  - 逐步展开中 `peer` 来自通道结构（Ring：SEND->next，RECV<-prev；其他模式可按需扩展）

## 3. 输出文件与字段含义

- `output/<topo_base>/proxy_flow_rank<rank>.jsonl`（每行一条 `proxyOp` 摘要）
  - `opCount`：操作计数（用于关联/排序）
  - `rank`：本地 rank
  - `channel`：通道 ID
  - `nsteps`：总步数（该 `proxyOp` 的步数）
  - `nbytes`：单步传输字节数（由 NCCL 计算）
  - `chunkSize`：块大小
  - `sliceSteps` / `chunkSteps`：切片/块步数
  - `dtype`：数据类型（`ncclDataType_t` 的枚举值）
  - `redOp`：规约操作（网络视角将 `avg` 视为 `sum`）
  - `pattern` / `protocol`：模式 / 协议（字符串）
  - `ringPrev` / `ringNext`：环结构中本 rank 的上/下游邻居（其他模式可扩展对应字段）

- `output/<topo_base>/flow_steps_rank<rank>.jsonl`（逐步展开，用于仿真）
  - 每个步骤两条记录（SEND/RECV）：
    - `opCount, rank, channel, step, op`（SEND/RECV）, `peer`, `bytes`, `pattern`, `protocol`, `stage`
    - `stage`：对 RING_TWICE 区分 `reduce-scatter` / `allgather`（不改变步序与对端，仅语义标签）

- `output/<topo_base>/flow_rank<rank>.json`（聚合视图）
  - `rank`：本地 rank
  - `meta`：摘要（来自 proxy_flow 第一条）
  - `steps`：完整逐步列表（同 `flow_steps_rank*.jsonl`）

## 4. 典型使用流程（建议）

- 设置环境变量：`NCCL_TOPO_FILE`、`LD_LIBRARY_PATH`、`GPU_DEV_NUM`。
- 在你的测试或集成代码中：
  ```c
  ncclSetFlowExtractionEnabled(1);
  // 触发需要的集合通信（例如 AllReduce）
  ncclGroupStart();
  for (int i = 0; i < nDev; ++i) {
    ncclAllReduce(NULL, NULL, count, ncclFloat, ncclSum, comms[i], (cudaStream_t)0);
  }
  ncclGroupEnd();
  // 生成当前 rank 的聚合输出
  ncclWriteAggregatedFlow(comms[0]);
  ```
  或
  ```c
  ncclExtractFlow(ncclFuncAllReduce, count, ncclFloat, /*root*/0, comm);
  ```
- 到 `output/<topo_base>/` 下读取 `proxy_flow_*`、`flow_steps_*`、`flow_rank*.json`。

## 5. 扩展与注意事项

- 非 Ring 模式（Tree/CollNet/NVLS）的逐步 peer 关系：可按需要在记录阶段扩展相同逻辑（通过 `comm->channels[chan].tree/*`、`collnet*`、`nvls*` 结构获取对端）。
- 若某协议/拓扑无需 proxy，将不会产生 `proxy_flow_*` 与 `flow_steps_*`；可根据需要选择其他集合通信触发获取更多记录。
- 输出目录为 `output/<topo_base>/...`，其中 `<topo_base>` 自动来自 `NCCL_TOPO_FILE` 的文件名（去扩展名）。

如需我对 Tree/CollNet/NVLS 的逐步展开进行增强或增加排序/统计字段，请告知具体需求与格式。 
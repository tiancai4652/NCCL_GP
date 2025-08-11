### Flow Extractor 代码与输出详解（详细说明）

本文件详细说明本次“流信息提取”相关代码的作用、记录的数据、字段含义，以及生成文件的解释。目标：让你可以快速将 NCCL 的真实通信计划转换为可供网络仿真器消费的有序流信息。

## 1. 代码总览与职责边界

- `src/enqueue.cc`
  - 负责构建 NCCL 的 kernel 计划与 proxy 入队。
  - 核心：`computeColl()` 计算 `ncclProxyOp` 字段；`addProxyOpIfNeeded()` 判断是否需要 proxy 并入队。
  - 我们在 `addProxyOpIfNeeded()` 处插桩：当“确实需要”入队时，调用 `ncclRecordProxyOp(...)` 记录真实 `proxyOp`。

- `src/flow_extractor.h`（新增）
  - 对外 API 与数据结构声明。
  - 主要 API：
    - `ncclSetFlowExtractionEnabled(int enable)`：启用/禁用提取
    - `ncclGetCollectiveFlow(...)`：模式化流（不执行，只描述）
    - `ncclFlowToJson(...)` / `ncclFreeCollectiveFlow(...)`
    - `ncclRecordProxyOp(...)`：真实入队时调用，落盘记录
    - `ncclWriteAggregatedFlow(...)`：聚合输出

- `src/flow_extractor.cc`（新增）
  - 复用 NCCL 的算法/协议选择，生成“模式化流”（便于直观校验）
  - 记录真实 `proxyOp`（摘要 + 逐步展开）
  - 聚合输出（`flow_rank<rank>.json`）

## 2. 关键函数与记录字段（代码语义）

### 2.1 `computeColl(struct ncclInfo*, ..., struct ncclProxyOp* proxyOp)`（`src/enqueue.cc`）
- 职责：在 NCCL 内部为一次 collective 计算 proxy 参数。
- 真实计算的字段（部分举例）：
  - `proxyOp->nsteps = info->nstepsPerLoop * nLoops * chunkSteps`
  - `proxyOp->sliceSteps`, `proxyOp->chunkSteps`, `proxyOp->chunkSize`
  - `proxyOp->protocol = info->protocol`
  - `proxyOp->dtype = info->datatype`
  - `proxyOp->redOp`（平均 op 在网络侧视为 sum）
  - `proxyOp->pattern = info->pattern`
  - `proxyOp->nbytes = stepSize * proxyOp->sliceSteps`（单步传输字节数）

这些值全部由 NCCL 内部依据拓扑、算法、协议与 buffer 切分规则计算得到。

### 2.2 `addProxyOpIfNeeded(struct ncclComm*, struct ncclKernelPlan*, struct ncclProxyOp* op)`（`src/enqueue.cc`）
- 职责：判断是否需要 proxy；若需要则拷贝 `op` 入队（计划的 `proxyOpQueue`）。
- 插桩点：在“确认需要”入队后，调用：
  ```c
  ncclRecordProxyOp(&infoCtx, q, comm);
  ```
  记录入队的真实 `proxyOp`。
- 注意：我们已去除“预记录/队列头重复记录”；仅在“真实入队”时记录一次，避免重复。

### 2.3 `ncclGetCollectiveFlow(...)`（`src/flow_extractor.cc`）
- 职责：在不执行通信的情况下，复用 NCCL 的算法/协议选择与 pattern/loop 推导，生成“模式化流”结构 `ncclCollectiveFlow`，可 JSON 输出。
- 内部逻辑：
  - 利用 `ncclTopoGetAlgoTime` 评估算法/协议，选择最优组合（选择结果为 NCCL 的真实决策）。
  - 根据 `collType + algorithm` 推导 `pattern`（RING、RING_TWICE、TREE_UP/DOWN/UP_DOWN、NVLS/NVLS_TREE、COLLNET_* 等）。
  - 推导 loop 信息（`nstepsPerLoop/nchunksPerLoop`）。

该函数输出的是“描述性计划”，方便阅读与校验；真实执行的步序与数据量来自 `proxyOp`。

### 2.4 `ncclRecordProxyOp(const ncclInfo*, const ncclProxyOp*, ncclComm*)`（`src/flow_extractor.cc`）
- 职责：在真实 `proxyOp` 入队后，落盘两类文件：
  - `proxy_flow_rank<rank>.jsonl`：每行一条 `proxyOp` 摘要
  - `flow_steps_rank<rank>.jsonl`：将 `nsteps` 逐步展开为 2×nsteps 行（SEND+RECV）
- 字段来源：
  - 邻居来自 `comm->channels[channelId].ring.{prev,next}`（Ring）
  - `nsteps/nbytes/chunkSize/sliceSteps/chunkSteps/dtype/redOp/pattern/protocol` 直接来自 `proxyOp`
  - 阶段标注（`stage`）仅对 `RING_TWICE` 区分 `reduce-scatter` 与 `allgather`，属于语义标签，不影响对端与步序。

### 2.5 `ncclWriteAggregatedFlow(struct ncclComm*)`（`src/flow_extractor.cc`）
- 职责：聚合当前 rank 的 `proxy_flow_*.jsonl` 与 `flow_steps_*.jsonl`，输出 `flow_rank<rank>.json`（包含一条 `meta` + 完整 `steps`）。
- 说明：当前按文件自然顺序输出；如需严格排序（`opCount, channel, step`），可在后续加入最小 JSON 解析与排序。

## 3. 生成文件与字段含义

### 3.1 `proxy_flow_rank<rank>.jsonl`
- 每行一条 JSON，表示一个入队成功的 `proxyOp` 摘要：
  - `opCount`：操作计数（用于关联/排序）
  - `rank`：本地 rank
  - `channel`：通道 ID（`channelId`）
  - `nsteps`：总步数（本 `proxyOp` 所含步数）
  - `nbytes`：单步传输字节数（从 `proxyOp->nbytes`）
  - `chunkSize`：块大小（从 `proxyOp->chunkSize`）
  - `sliceSteps` / `chunkSteps`：切片/块步数（协议/算法相关）
  - `dtype`：数据类型（`ncclDataType_t` 的枚举值，以 `uint8` 存储）
  - `redOp`：规约操作（网络视角，一般 `avg` 记录为 `sum`）
  - `pattern`：模式（字符串，RING、RING_TWICE、TREE_*、NVLS_*、COLLNET_* 等）
  - `protocol`：协议（字符串，LL/LL128/SIMPLE）
  - `ringPrev`/`ringNext`：环结构中本 rank 的上/下游邻居 rank

用途：快速了解 NCCL 对本次 collective 的真实计划要素；用于后续逐步展开与仿真。

### 3.2 `flow_steps_rank<rank>.jsonl`
- 每行一条 JSON，将 `proxyOp` 的 `nsteps` 展开为逐步操作；每步两条（SEND 与 RECV）：
  - `opCount`：与摘要一致，便于关联
  - `rank`：本地 rank
  - `channel`：通道 ID
  - `step`：步序号（0..nsteps-1）
  - `op`：`SEND` 或 `RECV`
  - `peer`：对端 rank（Ring：`SEND` 到 `ringNext`，`RECV` 来自 `ringPrev`）
  - `bytes`：本步传输字节数（来自 `proxyOp->nbytes`）
  - `pattern`/`protocol`：与摘要一致
  - `stage`：仅对 `RING_TWICE` 标注 `reduce-scatter` / `allgather`；其他模式默认 `ring` 或后续扩展

用途：直接可被网络仿真器消费；每行即为一次“有方向的数据传输”。

### 3.3 `flow_rank<rank>.json`
- 聚合视图，包含：
  - `rank`：本地 rank
  - `meta`：来自 `proxy_flow_rank<rank>.jsonl` 的第一条摘要 JSON（一次代表性 `proxyOp`）
  - `steps`：完整逐步列表，内容与 `flow_steps_rank<rank>.jsonl` 相同

用途：一次性加载当前 rank 的传输计划（既有代表性摘要，也有全部逐步信息）。

### 3.4 `flow_<Collective>_rank0.json`
- 由 `ncclGetCollectiveFlow(...) + ncclFlowToJson(...)` 输出的“模式化流”，用于直观校验。
- 主要字段：
  - `collective_type`，`algorithm`，`protocol`，`pattern`
  - `my_rank`，`total_ranks`，`total_steps`，`total_channels`，`total_bytes`
  - `topology_summary`
  - `channels[].steps[]`：每步包含 `step_id / operation / src_rank / dst_rank / data_size / data_offset / channel_id / chunk_id / description`
- 说明：这是“描述性计划”，不等同于真实入队的 `proxyOp`，但两者在算法/模式/通道/步数等维度一般吻合，用于 sanity check。

## 4. 字段真实性与来源说明

- 真实来源（非估算）：
  - 算法/协议/模式：NCCL 真实选择（内部用 `ncclTopoGetAlgoTime` 做代价评估）
  - 通道与邻居：`ncclComm->channels[...]` 中的 ring/tree/collnet/nvls 结构
  - `ncclProxyOp` 字段：由 `computeColl()` 依据协议/缓冲规则计算
  - 每步 `bytes`：来自 `proxyOp->nbytes`
- 标签/格式化：
  - `stage` 仅为 `RING_TWICE` 的语义标签，不改变对端与步序
  - 我们已移除所有“估算时间”字段，避免混淆

## 5. 使用建议与扩展点

- 重建流：
  - 对于 Ring/RING_TWICE：每步 `SEND` 到 `ringNext`，`RECV` 来自 `ringPrev`；`RING_TWICE` 的前半段可视为 reduce-scatter，后半段 allgather。
  - 多通道并行：按 `channel` 分组并行调度。
- 其他模式：
  - `PIPELINE_FROM/TO`（Broadcast/Reduce）与 Tree/CollNet/NVLS：可在记录时根据 `comm->channels[c].tree/*`、`collnet*`、`nvls*` 结构补充 `peer` 关系（当前示例主要展示 Ring，其他模式易于扩展相同逻辑）。
- 限制：
  - 仅当 `ncclProxySaveOp` 判断“需要 proxy”时才会入队并记录。极个别协议/拓扑组合若无需 proxy，将不会生成 `proxy_flow_*` 与 `flow_steps_*` 记录（你仍可使用 `ncclGetCollectiveFlow` 查看模式化计划）。
  - `flow_rank<rank>.json` 当前未做严格排序（按文件顺序输出）；如需稳定排序可后续加入。

## 6. 快速定位代码段

- `src/enqueue.cc`
  - `computeColl(...)`：计算 `proxyOp` 字段
  - `addProxyOpIfNeeded(...)`：真实入队与记录调用（插桩位置）
- `src/flow_extractor.h`
  - API：`ncclSetFlowExtractionEnabled`，`ncclGetCollectiveFlow`，`ncclFlowToJson`，`ncclRecordProxyOp`，`ncclWriteAggregatedFlow`
- `src/flow_extractor.cc`
  - `ncclGetCollectiveFlow(...)`：复用 NCCL 选择逻辑输出“模式化流”
  - `ncclRecordProxyOp(...)`：落盘 `proxy_flow_*` 与 `flow_steps_*`
  - `ncclWriteAggregatedFlow(...)`：聚合输出 `flow_rank<rank>.json`

如需我为 Tree/CollNet/NVLS 等模式补充逐步 `peer` 展开或为 `flow_rank*.json` 增加排序与统计字段，请告知你的优先级与所需格式。 
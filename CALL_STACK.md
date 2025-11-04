# NCCL Flow Extractor 输出文件生成调用栈

## 概览

本文档详细记录从用户调用到生成三种输出文件的完整调用链路。

---

## 🎯 三种输出文件

| 文件类型 | 内容 | 生成函数 |
|---------|------|---------|
| `proxy_flow_rank*.jsonl` | ProxyOp 摘要信息 | `ncclRecordProxyOp` |
| `flow_steps_rank*.jsonl` | 逐步的 SEND/RECV 操作 | `ncclRecordProxyPeerSteps` |
| `flow_rank*.json` | 聚合的完整视图 | `ncclWriteAggregatedFlow` |

---

## 📊 完整调用栈（从上到下）

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 用户层 (test_flow_extractor.cpp)                             │
└─────────────────────────────────────────────────────────────────┘
   │
   │ ncclSetFlowExtractionEnabled(1)  // 启用流提取
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ ncclGroupStart()                                                 │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. NCCL API 层                                                   │
│ ncclAllReduce(sendbuf, recvbuf, count, dtype, op, comm, stream) │
└─────────────────────────────────────────────────────────────────┘
   │
   │ (src/collectives/all_reduce.cc)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. NCCL 内部调度层                                               │
│ ncclEnqueueCheck(struct ncclInfo* info)                         │
│   └─> taskAppend(comm, info)                                    │
│         └─> ncclLaunchPrepare(comm)                             │
│               └─> ncclLaunchKernelBefore()                      │
└─────────────────────────────────────────────────────────────────┘
   │
   │ (src/enqueue.cc)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Plan 构建层                                                   │
│ ncclSaveKernel(struct ncclInfo* info)                           │
│   └─> setupCollFunc(info, plan)                                │
│         └─> computeColl(info, plan, &proxyOp)                  │
│               └─> addCollToPlan(...)                            │
└─────────────────────────────────────────────────────────────────┘
   │
   │ (src/enqueue.cc:277-356)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. ProxyOp 入队层 (关键插桩点 1)                                 │
│ addProxyOpIfNeeded(comm, plan, proxyOp)                         │
│   ├─> ncclProxySaveOp(comm, op, &needed)  // 判断是否需要 proxy │
│   │                                                              │
│   └─> if (needed) {                                             │
│         ncclRecordProxyOp(&infoCtx, q, comm);  ← 🔴 插桩点 1   │
│         // 生成 proxy_flow_rank*.jsonl                          │
│      }                                                           │
└─────────────────────────────────────────────────────────────────┘
   │
   │ (src/enqueue.cc:254-273)
   │ (src/flow_extractor.cc:198-226)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 📝 输出 1: proxy_flow_rank*.jsonl                                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ {                                                            │ │
│ │   "opCount": 0,                                             │ │
│ │   "rank": 0,                                                │ │
│ │   "channel": 0,                                             │ │
│ │   "nsteps": 8,                                              │ │
│ │   "nbytes": 65536,                                          │ │
│ │   "pattern": "RING_TWICE",                                  │ │
│ │   "protocol": "LL",                                         │ │
│ │   ...                                                        │ │
│ │ }                                                            │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
   │
   │ (继续 NCCL 内部流程)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Proxy 保存层                                                  │
│ ncclProxySaveOp(comm, op, &needed)                              │
│   └─> 根据 pattern 调用不同的 SaveProxy                         │
│                                                                  │
│   switch (op->pattern) {                                        │
│     case ncclPatternRing:                                       │
│     case ncclPatternRingTwice:                                  │
│       SaveProxy(comm, channel, proxyRecv, ring->prev, op, ...)  │
│       SaveProxy(comm, channel, proxySend, ring->next, op, ...)  │
│       break;                                                     │
│                                                                  │
│     case ncclPatternTreeUp:                                     │
│     case ncclPatternTreeDown:                                   │
│       SaveProxy(comm, channel, ..., tree->up/down[], op, ...)   │
│       break;                                                     │
│                                                                  │
│     case ncclPatternCollnetChain:                               │
│       SaveProxy(comm, channel, ..., collnetChain.up, op, ...)   │
│       break;                                                     │
│     ...                                                          │
│   }                                                              │
└─────────────────────────────────────────────────────────────────┘
   │
   │ (src/proxy.cc:528-589)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Proxy 详细记录层 (关键插桩点 2)                               │
│ SaveProxy(comm, channel, type, peer, op, ...)                   │
│   ├─> ncclLocalOpAppend(comm, &connector->proxyConn, op)       │
│   │                                                              │
│   └─> ncclRecordProxyPeerSteps(comm, channelId, type, peer, op)│
│         ← 🔴 插桩点 2                                           │
│         // 生成 flow_steps_rank*.jsonl                          │
│         // peer 来自真实拓扑！                                   │
└─────────────────────────────────────────────────────────────────┘
   │
   │ (src/proxy.cc:503-524)
   │ (src/flow_extractor.cc:254-312)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 📝 输出 2: flow_steps_rank*.jsonl                                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ {"opCount":0,"rank":0,"channel":0,"step":0,                │ │
│ │  "op":"SEND","peer":1,"bytes":65536,                        │ │
│ │  "pattern":"RING_TWICE","protocol":"LL",                    │ │
│ │  "stage":"reduce-scatter"}                                  │ │
│ │                                                              │ │
│ │ {"opCount":0,"rank":0,"channel":0,"step":0,                │ │
│ │  "op":"RECV","peer":4,"bytes":65536,                        │ │
│ │  "pattern":"RING_TWICE","protocol":"LL",                    │ │
│ │  "stage":"reduce-scatter"}                                  │ │
│ │ ...                                                          │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
   │
   │ (ncclGroupEnd() 触发实际执行)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ ncclGroupEnd()                                                   │
│   └─> ... 执行实际的集合通信 ...                                │
└─────────────────────────────────────────────────────────────────┘
   │
   │ (用户手动调用聚合函数)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. 聚合层                                                        │
│ ncclWriteAggregatedFlow(comm)                                   │
│   ├─> 读取 proxy_flow_rank*.jsonl 第一行作为 meta              │
│   ├─> 读取 flow_steps_rank*.jsonl 所有行作为 steps             │
│   └─> 生成 flow_rank*.json                                     │
└─────────────────────────────────────────────────────────────────┘
   │
   │ (src/flow_extractor.cc:314-369)
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 📝 输出 3: flow_rank*.json                                       │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ {                                                            │ │
│ │   "rank": 0,                                                │ │
│ │   "meta": { /* 来自 proxy_flow 第一行 */ },                 │ │
│ │   "steps": [ /* 来自 flow_steps 所有行 */ ]                 │ │
│ │ }                                                            │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 关键调用栈详解

### 栈 1：生成 proxy_flow_rank*.jsonl

```c
test_flow_extractor.cpp::main()
  └─> ncclAllReduce(...)
        └─> ncclEnqueueCheck(info)                    // src/enqueue.cc:1581
              └─> taskAppend(comm, info)
                    └─> ncclSaveKernel(info)
                          └─> setupCollFunc(info, plan)
                                └─> computeColl(info, plan, &proxyOp)
                                      └─> addCollToPlan(...)       // src/enqueue.cc:277
                                            └─> addProxyOpIfNeeded(comm, plan, &tmp)  
                                                                    // src/enqueue.cc:353
                                                  └─> ncclProxySaveOp(comm, op, &needed)
                                                  └─> if (needed)
                                                        ncclRecordProxyOp(&infoCtx, q, comm)
                                                        // ← 🔴 生成 proxy_flow_rank*.jsonl
                                                        // src/flow_extractor.cc:198-226
```

**关键数据来源**：
- `proxyOp` 来自 `computeColl`，由 NCCL 真实计算
- 所有字段（nsteps, nbytes, pattern, protocol）都来自 NCCL 内部决策

---

### 栈 2：生成 flow_steps_rank*.jsonl

```c
test_flow_extractor.cpp::main()
  └─> ncclAllReduce(...)
        └─> ncclEnqueueCheck(info)
              └─> taskAppend(comm, info)
                    └─> ncclSaveKernel(info)
                          └─> ... (同上) ...
                                └─> addProxyOpIfNeeded(comm, plan, &tmp)
                                      └─> ncclProxySaveOp(comm, op, &needed)
                                                          // src/proxy.cc:528-589
                                            └─> switch (op->pattern)
                                                  case ncclPatternRing:
                                                  case ncclPatternRingTwice:
                                                    ├─> SaveProxy(comm, channel, proxyRecv, 
                                                    │              ring->prev, op, ...)
                                                    │     └─> ncclRecordProxyPeerSteps(
                                                    │              comm, channel->id, 
                                                    │              type=0(RECV), 
                                                    │              peer=ring->prev, op)
                                                    │         // ← 🔴 生成 RECV 记录
                                                    │
                                                    └─> SaveProxy(comm, channel, proxySend, 
                                                                  ring->next, op, ...)
                                                          └─> ncclRecordProxyPeerSteps(
                                                                   comm, channel->id, 
                                                                   type=1(SEND), 
                                                                   peer=ring->next, op)
                                                              // ← 🔴 生成 SEND 记录
                                                              // src/flow_extractor.cc:254-312
```

**关键数据来源**：
- `peer` 参数来自 NCCL 真实拓扑：
  - Ring: `ring->prev` / `ring->next`
  - Tree: `tree->up` / `tree->down[]`
  - CollNet: `collnetChain.up` / `collnetDirect.out`
  - NVLS: `nvls.out` / `nvls.tree*`

---

### 栈 3：生成 flow_rank*.json

```c
test_flow_extractor.cpp::main()
  └─> ncclWriteAggregatedFlow(comm)              // 用户手动调用
                                                   // src/flow_extractor.cc:314-369
        ├─> fopen("proxy_flow_rank*.jsonl", "r")  // 读取
        ├─> fgets(meta, ...)                       // 第一行作为 meta
        ├─> fopen("flow_steps_rank*.jsonl", "r")  // 读取
        ├─> while (fgets(line, ...))               // 所有行作为 steps
        └─> fprintf(fo, "{\n  \"rank\": %d,\n  \"meta\": %s,\n  \"steps\": [...]\n}\n")
            // ← 🔴 生成 flow_rank*.json
```

**关键数据来源**：
- 聚合前两个 jsonl 文件
- 无新数据生成，只是格式转换

---

## 📋 关键插桩点总结

### 🔴 插桩点 1：`addProxyOpIfNeeded` (enqueue.cc:264)

```c
// 记录需要 proxy 的操作（会实际入队到 proxy 队列）
struct ncclInfo infoCtx = {};
infoCtx.comm = comm;
(void)ncclRecordProxyOp(&infoCtx, q, comm);  // ← 生成 proxy_flow
```

**触发时机**：当 NCCL 决定一个操作需要 proxy 线程时  
**调用频率**：每个 channel × 每个 proxyOp  
**输出文件**：`proxy_flow_rank*.jsonl`

---

### 🔴 插桩点 2：`SaveProxy` (proxy.cc:521)

```c
// 记录逐步流（SEND/RECV），使用真实的 peer 信息（从 NCCL 内部拓扑获取）
// 此函数覆盖所有通信模式：Ring/Tree/CollNet/NVLS/Pipeline
extern ncclResult_t ncclRecordProxyPeerSteps(struct ncclComm*, int channelId, 
                                              int type, int peer, 
                                              const struct ncclProxyOp*);
(void)ncclRecordProxyPeerSteps(comm, channel->id, type, peer, op);  // ← 生成 flow_steps
```

**触发时机**：当 NCCL 为每个 peer 保存 proxy 连接时  
**调用频率**：每个 channel × 每个方向(SEND/RECV) × 每个 peer  
**输出文件**：`flow_steps_rank*.jsonl`  
**关键**：`peer` 参数来自 NCCL 真实拓扑，不是推导！

---

## 🎯 数据流向图

```
┌──────────────────┐
│  ncclAllReduce   │
│   (用户调用)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   NCCL 内部       │
│  算法选择 +       │
│  拓扑规划         │
└────────┬─────────┘
         │
         ├─────────────────────────────┐
         │                              │
         ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│ addProxyOpIfNeeded│          │ ncclProxySaveOp  │
│   (插桩点 1)      │          │    (拓扑遍历)     │
└────────┬─────────┘          └────────┬─────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│ ncclRecordProxyOp│          │   SaveProxy      │
│                  │          │   (插桩点 2)      │
└────────┬─────────┘          └────────┬─────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│ proxy_flow_rank* │          │ncclRecordProxy   │
│     .jsonl       │          │   PeerSteps      │
│   (ProxyOp 摘要) │          └────────┬─────────┘
└──────────────────┘                   │
                                       ▼
                              ┌──────────────────┐
                              │ flow_steps_rank* │
                              │     .jsonl       │
                              │   (逐步 SEND/RECV)│
                              └────────┬─────────┘
                                       │
         ┌─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ncclWriteAggregated│
│      Flow         │
│   (用户手动调用)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  flow_rank*.json │
│   (聚合视图)      │
└──────────────────┘
```

---

## 🔧 调试技巧

### 1. 追踪 ProxyOp 生成
```bash
export NCCL_DEBUG=TRACE
./test_flow_extractor 2>&1 | grep -E "addProxyOpIfNeeded|addCollToPlan"
```

### 2. 追踪 SaveProxy 调用
在 `proxy.cc:521` 添加日志：
```c
printf("SaveProxy: rank=%d channel=%d type=%s peer=%d\n", 
       comm->rank, channel->id, type==0?"RECV":"SEND", peer);
```

### 3. 验证调用次数
```bash
# 查看 proxy_flow 记录数
wc -l output/nvlink_5GPU/proxy_flow_rank0.jsonl

# 查看 flow_steps 记录数（应该是 nsteps * 2 * proxy 数）
wc -l output/nvlink_5GPU/flow_steps_rank0.jsonl
```

---

## 📊 调用频率统计（以 5 GPU AllReduce 为例）

| 函数 | 每个 Rank 调用次数 | 总调用次数 (5 GPU) |
|------|-------------------|-------------------|
| `ncclAllReduce` | 1 | 5 |
| `addProxyOpIfNeeded` | 约 6-10 次 | 30-50 |
| `ncclRecordProxyOp` | 约 6-10 次 | 30-50 |
| `SaveProxy` | 约 12-20 次 | 60-100 |
| `ncclRecordProxyPeerSteps` | 约 12-20 次 | 60-100 |
| `ncclWriteAggregatedFlow` | 1 | 1 (只需调用一次) |

**注**：具体次数取决于 channel 数量、pattern 类型等。

---

## 🎯 总结

### 关键要点

1. **两个核心插桩点**：
   - `addProxyOpIfNeeded` → 生成 `proxy_flow`
   - `SaveProxy` → 生成 `flow_steps`

2. **所有数据都来自 NCCL 真实路径**：
   - `proxyOp` 来自 NCCL 内部计算
   - `peer` 来自 NCCL 拓扑初始化

3. **三种文件的关系**：
   - `proxy_flow`: 高层次摘要
   - `flow_steps`: 细粒度步骤（可直接用于仿真）
   - `flow_rank`: 聚合视图（便于分析）

4. **100% 准确性保证**：
   - 无推导、无假设、无估算
   - 所有 peer 来自真实拓扑
   - 所有数据来自 NCCL 内部结构

---

## 📖 相关文档

- [ACCURACY_FIX.md](./ACCURACY_FIX.md) - 准确性修复详情
- [COMMENT_FIX.md](./COMMENT_FIX.md) - 注释修正说明
- [README2.md](./README2.md) - 使用指南
- [VERIFY_ACCURACY.md](../VERIFY_ACCURACY.md) - 验证指南


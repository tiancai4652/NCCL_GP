# NCCL-GP 多节点支持 - 快速参考

## 一句话总结
实现了 NCCL-GP 对多节点拓扑的支持，通过 **Host-aware fake_cuda**（虚拟 busId）+ **分离的节点 XML**（本地拓扑），模拟真实 NCCL 的"全局感知、本地详细"机制。

---

## 核心问题 → 解决方案

| # | 问题 | 根本原因 | 解决方案 | 涉及文件 |
|---|------|---------|---------|---------|
| 1 | busId 冲突 | 所有进程在同一物理机，返回相同的真实 busId | `fake_cuda` 根据 `NCCL_HOSTID` 生成虚拟 busId（`0000:`/`0100:`） | `fake_cuda.cc` |
| 2 | Split comm 崩溃 | `gpu.rank` 是全局 rank，但 `peerInfo` 只有 local ranks，数组越界 | AllGather 后通过 busId 重新映射 `gpu.rank` | `init.cc` |
| 3 | 拓扑不匹配 | 统一 XML 的全局 `dev` ID 与本地 `cudaDev` 不符 | 每个节点加载单独的 XML（`node0.xml`, `node1.xml`） | `topo.cc`, XML 文件 |
| 4 | Trim 逻辑错误 | 用 `rank` 匹配 GPU，但 `rank` 已被重新映射 | 用 `busId`（不变）匹配 GPU | `paths.cc` |
| 5 | NET 设备误删 | 单节点 comm 也保留了 NET，判断逻辑错误 | 通过 `hostHash` 判断是否单节点 | `paths.cc` |
| 6 | NET 连接失败 | 拒绝注册 CUDA 内存类型 | `fake_cuda` 环境中，CUDA 内存=主机内存，允许注册 | `net_socket.cc` |

---

## 5 个关键改动点

### 1. `fake_cuda.cc` - Host-aware 虚拟 busId
```cpp
// 新增：逻辑节点 ID
static int g_hostId = -1;  // 从 NCCL_HOSTID 环境变量获取

// 修改：根据节点生成不同的虚拟 busId
cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    if (g_hostId == 0)
        snprintf(pciBusId, len, "0000:%02x:00.0", device + 1);  // Node 0
    else if (g_hostId == 1)
        snprintf(pciBusId, len, "0100:%02x:00.0", device + 1);  // Node 1
}
```
**效果**: Node 0 的 GPU0 → `0x10`，Node 1 的 GPU0 → `0x100010`，全局唯一

---

### 2. `topo.cc` - 根据 HOSTID 加载不同 XML
```cpp
char* xmlTopoFile = getenv("NCCL_TOPO_FILE");  // "../topo/2node_16gpu.xml"
char* hostIdStr = getenv("NCCL_HOSTID");       // "0" or "1"

// 构造节点专属文件名
snprintf(xmlPath, sizeof(xmlPath), "%.*s_node%d%s", 
         prefixLen, basePath, hostId, ".xml");
// Node 0: "../topo/2node_16gpu_node0.xml"
// Node 1: "../topo/2node_16gpu_node1.xml"
```
**效果**: 每个节点只加载本地 8 个 GPU + 1 个 NIC

---

### 3. `init.cc` - busId 重新映射 GPU rank
```cpp
// AllGather 后，遍历本地拓扑的所有 GPU
for (int g = 0; g < comm->topo->nodes[GPU].count; g++) {
    int64_t gpuBusId = gpu->id;  // 例如 0x10 (Node 0, GPU 0)
    
    // 在新 comm 的 peerInfo 中查找匹配的 busId
    for (int r = 0; r < nranks; r++) {
        if (comm->peerInfo[r].busId == gpuBusId) {
            gpu->gpu.rank = r;  // 更新为新 comm 的 local rank
            break;
        }
    }
}
```
**示例**:
- **Global comm (16 ranks)**: GPU0@Node0 的 busId=0x10 → peerInfo[0].busId=0x10 → rank=0 ✓
- **DP comm (2 ranks)**: GPU0@Node0 的 busId=0x10 → peerInfo[0].busId=0x10 → rank=0 ✓
- **DP comm (2 ranks)**: GPU0@Node1 的 busId=0x100010 → peerInfo[1].busId=0x100010 → rank=1 ✓

---

### 4. `paths.cc` - 用 busId 匹配 + 单节点判断
```cpp
// 修复 1：用 busId 匹配当前 GPU
for (int g=0; g<system->nodes[GPU].count; g++) {
    // 原来：if (gpu->gpu.rank == comm->rank) myDomain = domains[g];  // ❌ 错误！
    // 改为：
    if (gpu->id == comm->busId) {  // ✅ 正确！busId 不变
        myDomain = domains[g];
    }
}

// 修复 2：只对单节点 comm 删除 NET
int singleNode = 1;
for (int r = 1; r < comm->nRanks; r++) {
    if (comm->peerInfo[r].hostHash != comm->peerInfo[0].hostHash) {
        singleNode = 0;  // 发现不同节点的 rank，是多节点 comm
        break;
    }
}

if (singleNode && system->nodes[GPU].count == comm->nRanks) {
    // 单节点且 GPU 数量 = rank 数量，删除 NET（不需要跨节点通信）
    for (int n=system->nodes[NET].count-1; n>=0; n--)
        NCCLCHECK(ncclTopoRemoveNode(system, NET, n));
}
```

---

### 5. `net_socket.cc` - 允许注册 CUDA 内存
```cpp
ncclResult_t ncclNetSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
    // 原来：return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
    // 改为：
    if (mhandle) *mhandle = data;  // fake_cuda 中 CUDA 内存 = 主机内存
    return ncclSuccess;
}
```

---

## 数据流图

### 单节点 TP comm（Rank 0）
```
1. 启动进程
   └─> NCCL_HOSTID=0 (from run_test.sh)

2. fake_cuda: cudaDeviceGetPCIBusId(device=0)
   └─> 返回 "0000:01:00.0" (busId = 0x10)

3. topo.cc: ncclTopoGetSystem()
   └─> 加载 "../topo/2node_16gpu_node0.xml"
       └─> 包含 GPU 0-7 (busId 0000:01~08.0) + NIC 0

4. init.cc: AllGather peerInfo
   └─> Rank 0: busId=0x10, hostHash=0xAAA
   └─> Rank 1: busId=0x20, hostHash=0xAAA
   ...
   └─> Rank 7: busId=0x80, hostHash=0xAAA

5. init.cc: 重新映射 GPU rank
   └─> GPU 0 (busId=0x10) → peerInfo[0] → rank=0
   └─> GPU 1 (busId=0x20) → peerInfo[1] → rank=1
   ...

6. paths.cc: ncclTopoTrimSystem()
   └─> myGpu = busId 0x10 → domain=0
   └─> 所有 GPU 0-7 都在 domain 0（NVLink 连接）
   └─> singleNode=1（所有 hostHash 相同）
   └─> GPU count=8 == nRanks=8 → 删除 NET ✓

7. 建立 P2P/SHM 连接
   └─> Rank 0 ↔ Rank 1: SHM
   └─> Rank 0 ↔ Rank 2: SHM
   ...
```

### 跨节点 DP comm（Rank {0, 8}）
```
1. ncclCommSplit(global_comm, color=0, key=0/8)
   └─> 创建新 comm: nRanks=2, ranks={0, 8}

2. init.cc: AllGather peerInfo (新 comm)
   └─> peerInfo[0]: rank=0, busId=0x10,     hostHash=0xAAA (Node 0)
   └─> peerInfo[1]: rank=8, busId=0x100010, hostHash=0xBBB (Node 1)

3. topo.cc: ncclTopoGetSystem() (新 comm)
   └─> Rank 0: 加载 "../topo/2node_16gpu_node0.xml"
   └─> Rank 8: 加载 "../topo/2node_16gpu_node1.xml"

4. init.cc: 重新映射 GPU rank
   Rank 0 视角:
   └─> GPU 0 (busId=0x10)     → peerInfo[0].busId=0x10     → rank=0 ✓
   └─> GPU 1 (busId=0x20)     → 不在 peerInfo 中          → 保持
   ...
   
   Rank 8 视角:
   └─> GPU 0 (busId=0x100010) → peerInfo[1].busId=0x100010 → rank=1 ✓ (本地 dev=0，新 comm rank=1)
   └─> GPU 1 (busId=0x100020) → 不在 peerInfo 中          → 保持
   ...

5. paths.cc: ncclTopoTrimSystem()
   Rank 0:
   └─> myGpu = busId 0x10 → domain=0
   └─> 保留 domain 0 的 GPU（GPU 0-7），删除其他
   └─> singleNode=0（hostHash 0xAAA vs 0xBBB）
   └─> 保留 NET 设备 ✓
   
   Rank 8:
   └─> myGpu = busId 0x100010 → domain=0
   └─> 保留 domain 0 的 GPU（GPU 0-7 on Node 1）
   └─> singleNode=0
   └─> 保留 NET 设备 ✓

6. 建立 NET 连接
   └─> Rank 0 → Rank 1: via NET/Socket/0 (NIC eth0)
   └─> Rank 1 → Rank 0: via NET/Socket/1 (NIC eth1)
```

---

## 测试结果验证

### 命令
```bash
cd NCCL_GP/test
./run_test_2node_16gpu.sh
```

### 预期日志（Rank 0）
```
[fake_cuda] Set NCCL_HOSTID=0
NCCL INFO NCCL_TOPO_FILE set to ../topo/2node_16gpu.xml, NCCL_HOSTID=0, loading ../topo/2node_16gpu_node0.xml
NCCL INFO Rank 0: Mapped GPU 0 (busId=0x10) XML_rank=0 -> comm_rank=0
...
NCCL INFO [DEBUG] After GPU trim: GPU count=8, comm->nRanks=8, NET count=1
NCCL INFO [DEBUG] Single-node comm with all GPUs, removing NET devices
NCCL INFO comm 0x... rank 0 nranks 8 - Init COMPLETE  # TP comm
[Rank 0] Performing TP AllReduce...

NCCL INFO [DEBUG] After GPU trim: GPU count=8, comm->nRanks=2, NET count=1
NCCL INFO [DEBUG] Multi-node or partial comm, keeping NET devices
NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/Socket/0/GDRDMA
NCCL INFO comm 0x... rank 0 nranks 2 - Init COMPLETE  # DP comm
[Rank 0] Performing DP AllReduce...
```

### 成功指标
- ✅ 16 个 ranks 都打印 "Init COMPLETE" 3 次（Global, TP, DP）
- ✅ 16 个 ranks 都打印 "Performing TP AllReduce"
- ✅ 16 个 ranks 都打印 "Performing DP AllReduce"
- ✅ DP comm 日志显示 "via NET/Socket"

---

## Code Review 重点问题 Q&A

### Q1: 为什么不直接修改 XML 让 busId 匹配？
**A**: 因为 `fake_cuda` 的 `cudaDeviceGetPCIBusId` 返回的是**运行时计算的值**，而 XML 是静态文件。如果 XML 中写死 busId，那么：
- 必须确保 XML 的 busId 与 `fake_cuda` 生成的完全一致
- 无法灵活支持不同的节点数量
- 违反了"代码生成 busId"的设计

**我们的方案**：`fake_cuda` 和 XML 都根据 `NCCL_HOSTID` 生成一致的 busId（`0000:`/`0100:` 前缀）

---

### Q2: 为什么要重新映射 `gpu.rank`，不能直接用 XML 里的 rank 吗？
**A**: 因为 **split comm 的 rank 会重新编号**。
- **Global comm (16 ranks)**: Rank 0-15
- **DP comm (2 ranks)**: 新的 rank 0, 1（对应全局的 rank 0, 8）
- 如果不重新映射，`ncclTopoComputePaths` 会用 XML 的 rank=8 去索引 `comm->peerInfo[8]`，但新 comm 只有 2 个元素，**数组越界**！

**重新映射后**：
- GPU0@Node0: XML rank=0 → 新 comm rank=0 ✓
- GPU0@Node1: XML rank=8 → 新 comm rank=1 ✓

---

### Q3: 为什么不能简单判断 `GPU count != nRanks` 就保留 NET？
**A**: 因为会误判单节点的 partial comm。
- **TP comm (单节点，8 ranks)**: GPU count=8, nRanks=8 → `8 == 8` → 需要删除 NET ✓
- **某个单节点 split comm (4 ranks)**: GPU count=8, nRanks=4 → `8 != 4` → **误保留 NET**！

**正确判断**：先检查 `hostHash` 是否全部相同（单节点），再判断 GPU count

---

### Q4: fake_cuda 为什么要实现 `memcpy`？
**A**: 因为 NCCL 的 P2P 和 IPC 连接需要真实的内存拷贝来传递数据。
- **原来**: `cudaMemcpyAsync` 只返回成功，不执行拷贝
- **问题**: 进程间通过 IPC 共享内存时，数据不会真正传输
- **改后**: 在 `fake_cuda` 环境中，"CUDA 内存"实际上是主机内存（`malloc`），直接用 `memcpy` 模拟

---

### Q5: 这个方案能扩展到 4 节点、8 节点吗？
**A**: 可以！只需：
1. 为每个节点准备 XML（`node0.xml`, `node1.xml`, `node2.xml`, ...）
2. 在 `fake_cuda` 中为每个节点分配不同的 busId 前缀（`0000:`, `0100:`, `0200:`, ...）
3. 在测试脚本中根据 rank 设置 `NCCL_HOSTID`

**示例**（4 节点，32 ranks）:
```bash
if [ $OMPI_COMM_WORLD_RANK -lt 8 ]; then
    export NCCL_HOSTID=0
elif [ $OMPI_COMM_WORLD_RANK -lt 16 ]; then
    export NCCL_HOSTID=1
elif [ $OMPI_COMM_WORLD_RANK -lt 24 ]; then
    export NCCL_HOSTID=2
else
    export NCCL_HOSTID=3
fi
```

---

## 快速自查清单

在 code review 前，请确认：
- [ ] 理解 busId 的作用（全局唯一标识 GPU）
- [ ] 理解为什么要用 busId 而不是 rank 来匹配 GPU
- [ ] 理解 split comm 的 rank 重新编号问题
- [ ] 理解 hostHash 的作用（判断是否单节点）
- [ ] 理解为什么 fake_cuda 需要根据 NCCL_HOSTID 生成不同的 busId
- [ ] 能够解释每个改动的原因和效果

---

## 关键术语速查

| 术语 | 含义 | 在代码中的使用 |
|------|------|---------------|
| **busId** | PCI Bus ID，GPU 的唯一标识 | `comm->busId`, `gpu->id`, `peerInfo[r].busId` |
| **hostHash** | 节点的唯一标识 | `peerInfo[r].hostHash`，用于判断单/多节点 |
| **domain** | 通过 NVLink 直连的 GPU 集合 | `ncclTopoTrimSystem` 中计算，保留同 domain GPU |
| **rank** | 在 communicator 中的编号 | Global rank（0-15）vs Local rank（0-7 或 0-1） |
| **split comm** | 从已有 comm 创建的新 comm | `ncclCommSplit`，rank 会重新编号 |
| **NCCL_HOSTID** | 逻辑节点 ID（环境变量） | 0=Node0, 1=Node1，用于生成虚拟 busId |
| **PATH_NET** | 通过网络设备的路径类型 | 值为 8，小于此值表示本地连接（NVL/PIX） |

---

**总结**: 核心是"busId 作为全局唯一标识"，通过 busId 重新映射解决 split comm 问题，通过 hostHash 判断解决 NET 设备保留问题。


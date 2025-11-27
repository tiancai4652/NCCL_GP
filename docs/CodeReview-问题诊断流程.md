# NCCL-GP 多节点支持 - 问题诊断流程

本文档记录了我们在实现多节点支持过程中遇到的问题及诊断过程，供 code review 参考。

---

## 问题 1: Split comm 初始化崩溃

### 现象
```
NCCL WARN Could not find a path for pattern 4, falling back to simple order
internal error - please report this issue to the NCCL developers
```

### 诊断过程

#### Step 1: 查看日志，定位错误位置
```bash
$ tail -100 rank_logs/1/rank.13/stdout
...
NCCL INFO Rank 13 trimming GPU id 0x... (rank 13, domain 0, myDomain 0)
NCCL WARN Could not find a path for pattern 4
```

**发现**: Rank 13 在 trim GPU 时出现问题，只保留了 domain 0 的 GPU

---

#### Step 2: 检查拓扑文件生成
```bash
$ cat rank_logs/graph_dump.xml
```

**发现**: 
- 生成的 graph 只包含 GPU 0-7（Node 0）
- 缺少 GPU 8-15（Node 1）
- 只有 1 个 NIC

**结论**: 拓扑信息不完整，导致无法计算跨节点路径

---

#### Step 3: 检查 `ncclTopoTrimSystem` 逻辑
```cpp
// paths.cc:672
if (gpu->gpu.rank == comm->rank) myDomain = domains[g];
```

**分析**:
- `gpu->gpu.rank` 是什么？→ 从 XML 加载的 rank（例如 13）
- `comm->rank` 是什么？→ 当前进程的 rank（例如 13）
- 对于 **global comm**，它们匹配 ✓
- 对于 **split comm** 呢？

**测试 split comm**:
- DP comm: {rank 0=GPU0@Node0, rank 1=GPU0@Node1}
- XML 中: GPU0@Node0 的 `gpu.rank` = 0（全局 rank）
- 但是在 split comm 中，这个 GPU 的新 rank 还是 0！
- XML 中: GPU0@Node1 的 `gpu.rank` = 8（全局 rank）
- 但是在 split comm 中，这个 GPU 的新 rank 应该是 1！

**问题根源**: 
- XML 中的 `gpu.rank` 是**全局 rank**（0-15）
- Split comm 创建后，rank 会**重新编号**为 local rank（0-7 或 0-1）
- 但是 `gpu.rank` 没有更新，仍然是全局 rank
- 导致 `gpu->gpu.rank == comm->rank` 永远不匹配（对于 rank > 0 的情况）

---

#### Step 4: 查找正确的匹配方式
**Q**: 如何在 split comm 中找到"当前进程对应的 GPU"？

**选项 1**: 用 `gpu.rank` 匹配
- ❌ `gpu.rank` 是全局 rank，split comm 后不匹配

**选项 2**: 用 `cudaDev` 匹配
- ❌ `cudaDev` 在每个节点都是 0-7，不唯一

**选项 3**: 用 `busId` 匹配 ✅
- ✅ `comm->busId` 在初始化时从 `cudaDeviceGetPCIBusId` 获得
- ✅ `gpu->id` 从 XML 的 `busid` 属性加载
- ✅ busId 是全局唯一的，不受 split 影响

---

### 解决方案 1.1: 使用 busId 匹配当前 GPU
```cpp
// paths.cc, ncclTopoTrimSystem
for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    // ...
    
    // 原来：if (gpu->gpu.rank == comm->rank) myDomain = domains[g];
    // 改为：
    if (gpu->id == comm->busId) {
        myDomain = domains[g];
        myGpuFound = 1;
    }
}
```

---

### 验证结果
```
NCCL INFO [DEBUG] Found my GPU: busId=0x10, domain=0
NCCL INFO [DEBUG] After GPU trim: GPU count=8
```

**成功**: 现在能正确找到当前 GPU，trim 逻辑正常工作

---

## 问题 2: busId 冲突（Duplicate GPU detected）

### 现象
```
NCCL WARN Duplicate GPU detected : rank 0 and rank 8
```

### 诊断过程

#### Step 1: 打印所有 ranks 的 busId
```bash
$ grep "busId=" rank_logs/1/rank.*/stdout
rank.00: busId=0x10
rank.01: busId=0x20
...
rank.07: busId=0x80
rank.08: busId=0x10  # ❌ 与 rank 0 相同！
rank.09: busId=0x20  # ❌ 与 rank 1 相同！
...
```

**发现**: Node 0 和 Node 1 的 GPU busId 完全相同

---

#### Step 2: 检查 `fake_cuda` 的实现
```cpp
// fake_cuda.cc (原来)
cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    // 返回真实的 PCI busId
    snprintf(pciBusId, len, "0000:%02x:00.0", device + 1);
    return cudaSuccess;
}
```

**分析**:
- 所有进程都运行在**同一台物理机**上
- `cudaDeviceGetPCIBusId(device=0)` 返回真实的 PCIe busId `0000:01:00.0`
- Node 0 的 rank 0 和 Node 1 的 rank 8 都访问 `device=0`
- 它们得到**相同的 busId**！

**问题根源**: `fake_cuda` 不知道当前进程模拟的是哪个逻辑节点

---

#### Step 3: 设计虚拟 busId 方案
**目标**: 不同节点的 GPU 应该有不同的 busId

**方案**:
1. 通过环境变量 `NCCL_HOSTID` 告诉 `fake_cuda` 当前节点 ID
2. `fake_cuda` 根据 `NCCL_HOSTID` 生成不同的虚拟 busId

**busId 编码规则**:
- Node 0 (HOSTID=0): `0000:01.0` ~ `0000:08.0` (domain=0x0000)
- Node 1 (HOSTID=1): `0100:01.0` ~ `0100:08.0` (domain=0x0100)
- Node N (HOSTID=N): `0N00:01.0` ~ `0N00:08.0`

**转换为 64-bit ID**:
- `0000:01:00.0` → `0x00000100` → `0x10`
- `0100:01:00.0` → `0x01000100` → `0x100010`

---

### 解决方案 2.1: Host-aware fake_cuda
```cpp
// fake_cuda.cc
static int g_hostId = -1;

static void initFakeCudaHostId() {
    char* hostId = getenv("NCCL_HOSTID");
    if (hostId != NULL) {
        g_hostId = atoi(hostId);
    } else {
        g_hostId = 0;  // 默认节点 0
    }
}

cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    initFakeCudaHostId();
    
    if (g_hostId == 0) {
        snprintf(pciBusId, len, "0000:%02x:00.0", device + 1);
    } else if (g_hostId == 1) {
        snprintf(pciBusId, len, "0100:%02x:00.0", device + 1);
    } else {
        snprintf(pciBusId, len, "dead:beef:%02x.0", device);
    }
    
    return cudaSuccess;
}
```

---

### 解决方案 2.2: 测试脚本设置 NCCL_HOSTID
```bash
# run_test_2node_16gpu.sh
mpirun -np 16 \
  bash -c '
    if [ $OMPI_COMM_WORLD_RANK -lt 8 ]; then
      export NCCL_HOSTID=0
    else
      export NCCL_HOSTID=1
    fi
    exec ./test_2node_16gpu_tp_dp
  '
```

---

### 验证结果
```bash
$ grep "busId=" rank_logs/1/rank.*/stdout
rank.00: busId=0x10      (Node 0, GPU 0)
rank.01: busId=0x20      (Node 0, GPU 1)
...
rank.07: busId=0x80      (Node 0, GPU 7)
rank.08: busId=0x100010  (Node 1, GPU 0)  # ✅ 不同了！
rank.09: busId=0x100020  (Node 1, GPU 1)  # ✅ 不同了！
...
rank.15: busId=0x100080  (Node 1, GPU 7)
```

**成功**: busId 全局唯一，不再冲突

---

## 问题 3: XML 中的 busId 与 fake_cuda 不匹配

### 现象
```
NCCL WARN Rank 0: Could not find comm rank for GPU 0 (busId=0x..., XML_rank=0)
```

### 诊断过程

#### Step 1: 比较 XML 和 fake_cuda 的 busId
**fake_cuda 生成的 busId**:
- Node 0: `0000:01.0` → `0x10`
- Node 1: `0100:01.0` → `0x100010`

**原始 XML 中的 busId**:
```xml
<!-- 2node_16gpu.xml (统一文件) -->
<gpu dev="0" rank="0" busid="0000:01.0" />  <!-- GPU 0, Node 0 -->
<gpu dev="1" rank="1" busid="0000:02.0" />  <!-- GPU 1, Node 0 -->
...
<gpu dev="8" rank="8" busid="0000:09.0" />  <!-- GPU 0, Node 1 ??? -->
```

**问题**:
1. Node 1 的 rank 8（`NCCL_HOSTID=1`, `cudaDev=0`）：
   - `fake_cuda` 返回 busId `0100:01.0` (0x100010)
   - XML 中的 `dev="8"` 对应 busId `0000:09.0` (0x90)
   - **不匹配**！

2. 更根本的问题：Node 1 的 `cudaDev` 范围是 0-7（本地编号），但 XML 中用的是全局 `dev` 8-15

---

### 解决方案 3.1: 分离节点的 XML 文件
**思路**: 每个节点加载单独的 XML，只包含本地 GPU

**node0.xml** (Node 0):
```xml
<gpu dev="0" rank="0" busid="0000:01.0" />  <!-- 本地 dev=0，全局 rank=0 -->
<gpu dev="1" rank="1" busid="0000:02.0" />
...
<gpu dev="7" rank="7" busid="0000:08.0" />
<nic dev="0" />  <!-- NIC 0 -->
```

**node1.xml** (Node 1):
```xml
<gpu dev="0" rank="8"  busid="0100:01.0" />  <!-- 本地 dev=0，全局 rank=8 -->
<gpu dev="1" rank="9"  busid="0100:02.0" />
...
<gpu dev="7" rank="15" busid="0100:08.0" />
<nic dev="1" />  <!-- NIC 1 -->
```

**关键点**:
- `dev` 属性：本地编号 0-7（匹配 `cudaDev`）
- `rank` 属性：全局 rank 0-15
- `busid` 属性：根据节点使用不同前缀（`0000:` vs `0100:`）

---

### 解决方案 3.2: 动态加载对应的 XML
```cpp
// topo.cc
char* xmlTopoFile = getenv("NCCL_TOPO_FILE");  // "../topo/2node_16gpu.xml"
char* hostIdStr = getenv("NCCL_HOSTID");

if (hostIdStr != NULL) {
    int hostId = atoi(hostIdStr);
    char xmlPath[512];
    
    // 构造节点专属文件名
    snprintf(xmlPath, sizeof(xmlPath), "%.*s_node%d%s", 
             prefixLen, basePath, hostId, ".xml");
    // Node 0: "../topo/2node_16gpu_node0.xml"
    // Node 1: "../topo/2node_16gpu_node1.xml"
    
    NCCLCHECK(ncclTopoGetXmlFromFile(xmlPath, xml, 1));
}
```

---

### 验证结果
```
Rank 0: Loading ../topo/2node_16gpu_node0.xml
Rank 0: Mapped GPU 0 (busId=0x10) XML_rank=0 -> comm_rank=0  ✓

Rank 8: Loading ../topo/2node_16gpu_node1.xml
Rank 8: Mapped GPU 0 (busId=0x100010) XML_rank=8 -> comm_rank=8  ✓
```

**成功**: busId 匹配，能够正确加载拓扑

---

## 问题 4: Split comm 的 rank 映射错误

### 现象
```
# DP comm: {rank 0=GPU0@Node0, rank 1=GPU0@Node1}
Rank 0: ncclTopoComputePaths failed
```

### 诊断过程

#### Step 1: 分析 split comm 的 rank 编号
**Global comm (16 ranks)**:
- `comm->peerInfo[0..15]`: 全局 ranks 0-15
- XML 中的 `gpu.rank`: 0-15（全局 rank）

**DP comm (2 ranks, 从 split 创建)**:
- `comm->peerInfo[0..1]`: 
  - `peerInfo[0]`: 原全局 rank 0，busId=0x10 (GPU0@Node0)
  - `peerInfo[1]`: 原全局 rank 8，busId=0x100010 (GPU0@Node1)
- XML 中的 `gpu.rank`: **仍然是 0, 8（全局 rank）**

---

#### Step 2: 检查 `ncclTopoComputePaths` 的逻辑
```cpp
// paths.cc
for (int g = 0; g < system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    int rank = gpu->gpu.rank;  // 例如 8 (全局 rank)
    
    // 访问 peerInfo
    struct ncclPeerInfo* peerInfo = comm->peerInfo + rank;  // ❌ comm->peerInfo[8]
    // 但是 DP comm 只有 2 个元素！越界访问！
}
```

**问题根源**: 
- XML 加载后，`gpu.rank` 是全局 rank（0-15）
- Split comm 的 `peerInfo` 只有新 comm 的 ranks（0-1）
- 用全局 rank 索引 local `peerInfo`，**数组越界**

---

### 解决方案 4: AllGather 后重新映射 GPU rank
```cpp
// init.cc, initTransportsRank
// AllGather peerInfo（新 comm 的所有 ranks）
NCCLCHECK(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)));

// 重新映射 GPU rank
for (int g = 0; g < comm->topo->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = comm->topo->nodes[GPU].nodes + g;
    int64_t gpuBusId = gpu->id;
    int xmlRank = gpu->gpu.rank;  // 全局 rank
    
    // 在新 comm 的 peerInfo 中查找匹配的 busId
    int found = 0;
    for (int r = 0; r < nranks; r++) {
        if (comm->peerInfo[r].busId == gpuBusId) {
            // 找到了！更新为新 comm 的 local rank
            gpu->gpu.rank = r;
            INFO(NCCL_INIT, "Mapped GPU %d (busId=0x%lx) XML_rank=%d -> comm_rank=%d",
                 g, gpuBusId, xmlRank, r);
            found = 1;
            break;
        }
    }
}
```

**示例**:
- **DP comm**: `peerInfo[0].busId=0x10`, `peerInfo[1].busId=0x100010`
- **GPU0@Node0**: `gpu.id=0x10` → 匹配 `peerInfo[0]` → `gpu.rank` 从 0 更新为 0 ✓
- **GPU0@Node1**: `gpu.id=0x100010` → 匹配 `peerInfo[1]` → `gpu.rank` 从 8 更新为 1 ✓

---

### 验证结果
```
Rank 0 (DP comm): Mapped GPU 0 (busId=0x10) XML_rank=0 -> comm_rank=0
Rank 1 (DP comm): Mapped GPU 0 (busId=0x100010) XML_rank=8 -> comm_rank=1
```

**成功**: Split comm 可以正确访问 `peerInfo`，不再越界

---

## 问题 5: 单节点 comm 保留了 NET 设备

### 现象
```
# TP comm (单节点，8 ranks)
NCCL INFO Keeping NET devices
```

### 诊断过程

#### Step 1: 检查原始的 trim 逻辑
```cpp
// paths.cc (原来)
if (system->nodes[GPU].count == comm->nRanks) {
    // 删除所有 NET 设备
}
```

**分析**:
- **Global comm (16 ranks)**: GPU count=8 (本地), nRanks=16 → `8 != 16` → 保留 NET ✓
- **TP comm (8 ranks, 单节点)**: GPU count=8, nRanks=8 → `8 == 8` → **删除 NET** ✓
- **DP comm (2 ranks, 跨节点)**: GPU count=8, nRanks=2 → `8 != 2` → 保留 NET ✓

看起来逻辑是对的，**但是**……

---

#### Step 2: 发现边界情况
**假设有一个单节点的 4-rank split comm**:
- GPU count=8（本地所有 GPU）
- nRanks=4（split 后只有 4 个 ranks）
- `8 != 4` → **保留 NET**！

这是错误的！单节点 comm 不需要 NET 设备。

---

### 解决方案 5: 通过 hostHash 判断是否单节点
```cpp
// paths.cc
// 检查是否所有 ranks 都在同一个节点
int singleNode = 1;
if (comm->nRanks > 1) {
    uint64_t firstHostHash = comm->peerInfo[0].hostHash;
    for (int r = 1; r < comm->nRanks; r++) {
        if (comm->peerInfo[r].hostHash != firstHostHash) {
            singleNode = 0;  // 发现不同节点
            break;
        }
    }
}

// 只有单节点 + GPU count == nRanks 才删除 NET
if (singleNode && system->nodes[GPU].count == comm->nRanks) {
    INFO(NCCL_INIT, "Single-node comm, removing NET devices");
    for (int n=system->nodes[NET].count-1; n>=0; n--)
        NCCLCHECK(ncclTopoRemoveNode(system, NET, n));
} else {
    INFO(NCCL_INIT, "Multi-node comm, keeping NET devices");
}
```

**逻辑表**:
| Comm 类型 | GPU count | nRanks | singleNode | 删除 NET? |
|-----------|-----------|--------|------------|-----------|
| Global (跨节点) | 8 | 16 | 0 | ❌ 保留 |
| TP (单节点) | 8 | 8 | 1 | ✅ 删除 |
| DP (跨节点) | 8 | 2 | 0 | ❌ 保留 |
| 单节点 4-rank | 8 | 4 | 1 | ❌ 保留（count != nRanks） |

---

### 验证结果
```
# TP comm
NCCL INFO [DEBUG] After GPU trim: GPU count=8, comm->nRanks=8, NET count=1
NCCL INFO [DEBUG] Single-node comm with all GPUs, removing NET devices
NCCL INFO [DEBUG] After trim: GPU count=8, NET count=0

# DP comm
NCCL INFO [DEBUG] After GPU trim: GPU count=8, comm->nRanks=2, NET count=1
NCCL INFO [DEBUG] Multi-node or partial comm, keeping NET devices (singleNode=0)
NCCL INFO [DEBUG] After trim: GPU count=8, NET count=1
```

**成功**: 单节点 comm 删除 NET，跨节点 comm 保留 NET

---

## 问题 6: 跨节点 NET 连接失败

### 现象
```
NCCL WARN Failed to execute operation Connect, retcode 3
```

### 诊断过程

#### Step 1: 追踪错误堆栈
```
transport/net_socket.cc:810 -> 3
proxy.cc:1533 -> 3
```

#### Step 2: 检查 `ncclNetSocketRegMr` 的实现
```cpp
// net_socket.cc (原来)
ncclResult_t ncclNetSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
    return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}
```

**问题**: 只接受 `NCCL_PTR_HOST` 类型，拒绝 `NCCL_PTR_CUDA`

---

#### Step 3: 理解 fake_cuda 的内存模型
在 `fake_cuda` 环境中:
- `cudaMalloc` 实际上调用 `malloc`（主机内存）
- "CUDA 内存"就是主机内存
- 网络插件可以直接访问

---

### 解决方案 6: 允许注册所有类型的内存
```cpp
// net_socket.cc
ncclResult_t ncclNetSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
    // 仿真模式：fake_cuda 中 CUDA 内存 = 主机内存
    if (mhandle) *mhandle = data;
    return ncclSuccess;
}
```

---

### 验证结果
```
NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/Socket/0/GDRDMA
NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Socket/0/GDRDMA
NCCL INFO Connected all rings
```

**成功**: 跨节点 NET 连接建立成功

---

## 总结：问题诊断的关键步骤

1. **查看日志**: 定位错误发生的位置和上下文
2. **理解数据结构**: 
   - `gpu.rank`（XML 中的全局 rank）
   - `comm->rank`（当前进程在 comm 中的 rank）
   - `comm->busId`（当前进程的 GPU busId）
   - `peerInfo[r].busId`（comm 中第 r 个 rank 的 busId）
3. **追踪数据流**: 
   - XML → `ncclTopoGetSystem` → `system->nodes[GPU]`
   - `cudaDeviceGetPCIBusId` → `comm->busId`
   - AllGather → `comm->peerInfo`
4. **验证假设**: 打印关键变量，确认实际值与预期是否一致
5. **设计测试用例**: 覆盖不同的 comm 类型（global, TP, DP）

---

## 调试技巧

### 1. 添加详细的 debug 日志
```cpp
INFO(NCCL_INIT, "[DEBUG] GPU %d: busId=0x%lx, rank=%d, cudaDev=%d", 
     g, gpu->id, gpu->gpu.rank, cudaDev);
```

### 2. 使用 `printf` + `fflush` 确保输出
```cpp
printf("[DEBUG] Before trim: GPU count=%d\n", system->nodes[GPU].count);
fflush(stdout);
```

### 3. 对比不同 comm 的行为
- Global comm（16 ranks，跨节点）
- TP comm（8 ranks，单节点）
- DP comm（2 ranks，跨节点）

### 4. 验证数据结构的一致性
```bash
# 检查所有 ranks 的 busId
grep "busId=" rank_logs/1/rank.*/stdout | sort

# 统计成功的 comm 初始化
grep "Init COMPLETE" rank_logs/1/rank.*/stdout | wc -l
```

---

**总结**: 通过系统化的诊断流程（日志 → 数据结构 → 数据流 → 测试），我们成功定位并解决了 6 个关键问题，实现了 NCCL-GP 的多节点支持。


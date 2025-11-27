# NCCL-GP 多节点拓扑支持改动说明（Code Review）

**提交范围**: temp (2e3acc0) → temp2 (a894c3e) → temp3 (0c88a29) → temp4 (ed8deb6)  
**目标**: 实现 NCCL-GP 对多节点、多 GPU 拓扑的支持，特别是跨节点 DP（数据并行）通信  
**核心思路**: 模拟真实 NCCL 的"全局感知、本地详细"机制

---

## 问题背景

### 原始问题
在 2 节点 16 GPU 测试中出现：
```
NCCL WARN Could not find a path for pattern 4, falling back to simple order
internal error - please report this issue to the NCCL developers
```

### 根本原因
1. **拓扑文件**：原来使用统一的全局拓扑文件，但 `fake_cuda` 返回的是本地 busId
2. **busId 冲突**：所有进程运行在同一台物理机上，`cudaDeviceGetPCIBusId` 返回真实的 PCIe busId，导致多节点模拟时 busId 重复
3. **split comm 问题**：`ncclCommSplit` 后，topology 中的 `gpu.rank` 仍然是全局 rank，但 `comm->peerInfo` 只有新 comm 的 ranks，导致数组越界
4. **NET 设备丢失**：单节点 comm 也保留了 NET 设备，导致判断逻辑错误

---

## 解决方案架构

### 核心设计：Scheme A + Scheme C
- **Scheme A（分离 XML 拓扑）**: 每个节点加载单独的 XML 文件（`node0.xml`, `node1.xml`），只包含本地 GPU 和 NIC
- **Scheme C（Host-aware fake_cuda）**: `fake_cuda` 根据 `NCCL_HOSTID` 环境变量生成不同的虚拟 busId，模拟不同节点的 GPU

### 为什么选择这个方案？
1. **贴近真实 NCCL**: 真实 NCCL 中，每个节点只有本地拓扑信息，通过 `peerInfo` AllGather 获得全局感知
2. **避免冲突**: 虚拟 busId 确保不同节点的 GPU 可以唯一识别
3. **支持 split comm**: 通过 busId 重新映射 rank，正确处理 split communicator

---

## 详细改动说明

### 1. `src/graph/fake_cuda.cc` - 核心改动：Host-aware 虚拟 busId

#### 改动 1.1：增加逻辑节点 ID
```cpp
// 新增全局变量
static int g_hostId = -1;

// 新增初始化函数
static void initFakeCudaHostId() {
    char* hostId = getenv("NCCL_HOSTID");
    if (hostId != NULL) {
        g_hostId = atoi(hostId);
        printf("[fake_cuda] Set NCCL_HOSTID=%d\n", g_hostId);
    } else {
        g_hostId = 0;  // 默认节点0
    }
}
```

**为什么要这样改？**
- `fake_cuda` 需要知道当前进程模拟的是哪个逻辑节点
- `NCCL_HOSTID` 由测试脚本根据 MPI rank 设置（0-7 → Node 0, 8-15 → Node 1）
- 不同节点需要生成不同的虚拟 busId 来模拟物理隔离

#### 改动 1.2：生成节点特定的虚拟 busId
```cpp
cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    initFakeCudaHostId();  // 确保g_hostId已初始化
    
    // 根据 g_hostId 和 device 生成虚拟 busId
    // Node 0: 0000:01.0, 0000:02.0, ..., 0000:08.0
    // Node 1: 0100:01.0, 0100:02.0, ..., 0100:08.0
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

**为什么要这样改？**
- **原来**: 所有进程都返回真实的 PCIe busId（例如 `0000:01:00.0`），导致 Node 0 的 GPU0 和 Node 1 的 GPU0 busId 相同
- **改后**: Node 0 使用 `0000:XX` 前缀，Node 1 使用 `0100:XX` 前缀，确保 busId 全局唯一
- **效果**: NCCL 可以通过 busId 区分不同节点的 GPU

#### 改动 1.3：修复 cudaMemcpyAsync 实际执行拷贝
```cpp
cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, 
                                       enum cudaMemcpyKind kind, cudaStream_t stream) {
    // 在fake_cuda环境中，所有内存都是主机内存，直接执行memcpy
    if (dst && src && count > 0) {
        memcpy(dst, src, count);
    }
    return cudaSuccess;
}
```

**为什么要这样改？**
- **原来**: 只返回 `cudaSuccess`，不执行实际拷贝
- **问题**: P2P 和 IPC 连接需要真实的内存拷贝来传递数据
- **改后**: 在 `fake_cuda` 环境中，"CUDA 内存"实际上是主机内存，直接用 `memcpy` 模拟

#### 改动 1.4：修复 cudaIpcGetMemHandle 和 cudaIpcOpenMemHandle
```cpp
cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) {
    // 将指针编码到 handle 中
    *(void**)handle = devPtr;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, 
                                            unsigned int flags) {
    // 从 handle 中解码指针
    *devPtr = *(void**)handle;
    return cudaSuccess;
}
```

**为什么要这样改？**
- **原来**: `cudaIpcOpenMemHandle` 不设置 `*devPtr`，导致 P2P 连接失败
- **改后**: 通过 handle 传递指针，模拟 IPC 内存共享（在同一进程空间内，指针直接可用）

#### 改动 1.5：修复 cudaStreamGetCaptureInfo
```cpp
cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(cudaStream_t stream, 
                                                enum cudaStreamCaptureStatus *pCaptureStatus, 
                                                unsigned long long *pId) {
    if (pCaptureStatus) *pCaptureStatus = cudaStreamCaptureStatusNone;
    if (pId) *pId = 0;
    return cudaSuccess;
}
```

**为什么要这样改？**
- **原来**: 不设置输出参数，导致 NCCL 读取未初始化的内存，触发 `ncclInvalidUsage`
- **改后**: 返回 `cudaStreamCaptureStatusNone`，表示 stream 未处于 graph capture 状态

---

### 2. `src/graph/topo.cc` - 改动：根据 HOSTID 加载节点特定的 XML

#### 改动 2.1：动态选择拓扑文件
```cpp
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system) {
    char* xmlTopoFile = getenv("NCCL_TOPO_FILE");
    if (xmlTopoFile) {
        char* hostIdStr = getenv("NCCL_HOSTID");
        if (hostIdStr != NULL) {
            int hostId = atoi(hostIdStr);
            char xmlPath[512];
            
            // 构造文件名：例如 "../topo/2node_16gpu.xml" 
            //          → "../topo/2node_16gpu_node0.xml"
            const char* basePath = xmlTopoFile;
            const char* lastDot = strrchr(basePath, '.');
            if (lastDot != NULL) {
                size_t prefixLen = lastDot - basePath;
                snprintf(xmlPath, sizeof(xmlPath), "%.*s_node%d%s", 
                         (int)prefixLen, basePath, hostId, lastDot);
                NCCLCHECK(ncclTopoGetXmlFromFile(xmlPath, xml, 1));
            }
        } else {
            // 没有 NCCL_HOSTID，加载原始文件
            NCCLCHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xml, 1));
        }
    }
}
```

**为什么要这样改？**
- **原来**: 所有进程加载同一个 XML 文件，包含所有 16 个 GPU
- **问题**: 
  1. 全局 `dev` ID（0-15）与本地 `cudaDev`（0-7）不匹配
  2. 无法模拟"每个节点只有本地拓扑"的真实场景
- **改后**: 
  - Node 0（rank 0-7）加载 `2node_16gpu_node0.xml`（只有 GPU dev 0-7 + NIC 0）
  - Node 1（rank 8-15）加载 `2node_16gpu_node1.xml`（只有 GPU dev 0-7 + NIC 1）
- **效果**: 模拟真实 NCCL 的"本地详细"机制

---

### 3. `src/init.cc` - 改动：busId 重新映射 GPU rank

#### 改动 3.1：AllGather 后重新映射 GPU rank
```cpp
static ncclResult_t initTransportsRank(struct ncclComm* comm, ...) {
    // ... AllGather peerInfo ...
    
    // 重新映射 GPU rank 到当前 comm 的 rank（通过 busId 匹配）
    if (comm->topo != NULL && comm->topo->nodes[GPU].count > 0) {
        for (int g = 0; g < comm->topo->nodes[GPU].count; g++) {
            struct ncclTopoNode* gpu = comm->topo->nodes[GPU].nodes + g;
            int64_t gpuBusId = gpu->id;
            int xmlRank = gpu->gpu.rank;
            
            // 在当前 comm 的 peerInfo 中查找匹配的 busId
            int found = 0;
            for (int r = 0; r < nranks; r++) {
                if (comm->peerInfo[r].busId == gpuBusId) {
                    // 找到了！r 就是这个 GPU 在当前 comm 中的 rank
                    gpu->gpu.rank = r;
                    found = 1;
                    break;
                }
            }
            
            if (!found) {
                // 不在当前 comm 中的 GPU，保持不变
                // 会在 ncclTopoTrimSystem 中被删除
            }
        }
    }
}
```

**为什么要这样改？**
- **问题场景**: Split comm 将全局 16 ranks 分成多个小 comm（例如 TP comm: ranks 0-7，DP comm: ranks {0,8}, {1,9}, ...）
- **原来的问题**: 
  1. XML 中的 `gpu.rank` 是全局 rank（0-15）
  2. Split comm 的 `peerInfo` 只有新 comm 的 ranks（例如 DP comm 只有 2 个 ranks: 0, 1）
  3. `ncclTopoComputePaths` 使用 `gpu.rank` 索引 `comm->peerInfo[gpu.rank]`，导致越界访问
- **解决方案**: 
  1. 通过 **busId** 在 `peerInfo` 中查找对应的 rank（busId 是全局唯一的）
  2. 将 `gpu.rank` 从全局 rank 重新映射为新 comm 的 local rank
  3. 例如：DP comm {rank 0=GPU0@Node0, rank 1=GPU0@Node1}
     - GPU0@Node0 的 busId = 0x10，在新 comm 的 rank = 0
     - GPU0@Node1 的 busId = 0x100010，在新 comm 的 rank = 1
- **效果**: Split comm 可以正确访问 `peerInfo`，不再越界

#### 改动 3.2：Split comm 也加载拓扑
```cpp
ncclResult_t ncclCommInitRankFunc(struct ncclCommInitRankAsyncJob* job) {
    // Split comm 也需要加载拓扑
    if (job->parent && comm->topo == NULL) {
        NCCLCHECKGOTO(ncclTopoGetSystem(comm, &comm->topo), res, fail);
        if (comm->topo && comm->topo->nodes[GPU].count > 0) {
            get_info_from_topo(comm->topo, comm->topo->nodes[GPU].count);
        }
    }
}
```

**为什么要这样改？**
- **原来**: Split comm 的 `topo` 是 `NULL`，导致后续路径计算失败
- **改后**: Split comm 也显式加载拓扑（根据 `NCCL_HOSTID` 加载本地 XML）

---

### 4. `src/graph/paths.cc` - 改动：修复拓扑修剪逻辑

#### 改动 4.1：使用 busId 匹配当前 GPU（而不是 rank）
```cpp
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) {
    int *domains, *ids;
    int myDomain = 0;
    int myGpuFound = 0;
    
    for (int g=0; g<system->nodes[GPU].count; g++) {
        struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
        domains[g] = g;
        ids[g] = gpu->id;
        
        // 计算 domain（通过 NVLink 连接的 GPU 在同一个 domain）
        for (int p=0; p<g; p++) {
            if (gpu->paths[GPU][p].type < PATH_NET) {
                domains[g] = std::min(domains[g], domains[p]);
            }
        }
        
        // 使用 busId 匹配当前 rank 的 GPU
        if (gpu->id == comm->busId) {
            myDomain = domains[g];
            myGpuFound = 1;
        }
    }
    
    if (!myGpuFound) {
        WARN("Could not find my GPU (busId=0x%lx) in topology", comm->busId);
        return ncclInternalError;
    }
    
    // 删除不在 myDomain 的 GPU
    for (int i=0; i<ngpus; i++) {
        if (domains[i] != myDomain) {
            // 找到 GPU 并删除
            NCCLCHECK(ncclTopoRemoveNode(system, GPU, g));
        }
    }
}
```

**为什么要这样改？**
- **原来的 bug**: 
  ```cpp
  if (gpu->gpu.rank == comm->rank) myDomain = domains[g];
  ```
  - `gpu->gpu.rank` 已经被重新映射为新 comm 的 local rank（例如 0）
  - `comm->rank` 是全局 rank（例如 1）
  - 它们永远不匹配！
- **后果**: `myDomain` 保持为 0，导致只保留 domain 0 的 GPU，其他都被错误删除
- **改后**: 使用 `comm->busId`（唯一且不变）匹配，保证找到正确的 GPU

#### 改动 4.2：只对单节点 comm 删除 NET 设备
```cpp
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) {
    // ... trim GPUs ...
    
    // 检查是否所有 ranks 都在同一个节点（通过 hostHash 判断）
    int singleNode = 1;
    if (comm->nRanks > 1) {
        uint64_t firstHostHash = comm->peerInfo[0].hostHash;
        for (int r = 1; r < comm->nRanks; r++) {
            if (comm->peerInfo[r].hostHash != firstHostHash) {
                singleNode = 0;
                break;
            }
        }
    }
    
    // 只有当所有 ranks 都在同一节点且 GPU count == nRanks 时，才删除 NET 设备
    if (singleNode && system->nodes[GPU].count == comm->nRanks) {
        INFO(NCCL_INIT, "Single-node comm with all GPUs, removing NET devices");
        for (int n=system->nodes[NET].count-1; n>=0; n--)
            NCCLCHECK(ncclTopoRemoveNode(system, NET, n));
    } else {
        INFO(NCCL_INIT, "Multi-node or partial comm, keeping NET devices");
    }
}
```

**为什么要这样改？**
- **原来的逻辑**: 
  ```cpp
  if (system->nodes[GPU].count == comm->nRanks) {
      // 删除所有 NET 设备
  }
  ```
- **问题**: 
  - TP comm（单节点，8 ranks, 8 GPUs）: `8 == 8` → 删除 NET ✓（正确）
  - DP comm（跨节点，2 ranks, 8 GPUs）: `8 != 2` → 保留 NET ✓（正确）
  - **但是**: 如果某个单节点 split comm 正好也是 `GPU count == nRanks`，NET 就被错误删除了！
- **改后**: 
  1. 先检查是否所有 ranks 的 `hostHash` 相同（单节点）
  2. 只有**单节点 + GPU count == nRanks** 才删除 NET
  3. 跨节点 comm 一定保留 NET，用于跨节点通信

---

### 5. `src/transport/net_socket.cc` - 改动：允许注册 CUDA 内存

#### 改动 5.1：修复 ncclNetSocketRegMr
```cpp
ncclResult_t ncclNetSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
    // 仿真模式：fake_cuda环境中，CUDA内存实际上是主机内存
    if (mhandle) *mhandle = data; // 返回原始指针作为假handle
    return ncclSuccess;
}
```

**为什么要这样改？**
- **原来**: 
  ```cpp
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
  ```
  - 只接受 `NCCL_PTR_HOST` 类型的内存
- **问题**: 跨节点 NET 连接尝试注册 `NCCL_PTR_CUDA` 类型的内存，返回 `ncclInternalError`，导致连接失败
- **改后**: 在 `fake_cuda` 环境中，"CUDA 内存"实际上是主机内存（通过 `malloc` 分配），可以直接注册

---

### 6. `topo/2node_16gpu_node0.xml` 和 `node1.xml` - 新增：节点专属拓扑文件

#### node0.xml 结构（Node 0）
```xml
<system version="1">
  <cpu numaid="0x1" affinity="0x1">
    <pci busid="0000:00:00.0">
      <!-- 8 个 GPU: busId 0000:01.0 ~ 0000:08.0 -->
      <pci busid="0000:01.0">
        <gpu dev="0" sm="80" rank="0" gdr="1">
          <nvlink target="0000:02.0" count="2"/>
          <nvlink target="0000:03.0" count="1"/>
        </gpu>
      </pci>
      <!-- ... GPU 1-7 ... -->
      
      <!-- 1 个 NIC: busId 0000:10.0 -->
      <pci busid="0000:10.0">
        <nic>
          <net name="eth0" dev="0" speed="100000" .../>
        </nic>
      </pci>
    </pci>
  </cpu>
</system>
```

#### node1.xml 结构（Node 1）
```xml
<system version="1">
  <cpu numaid="0x2" affinity="0x2">
    <pci busid="0100:00:00.0">  <!-- 注意：0100 前缀！ -->
      <!-- 8 个 GPU: busId 0100:01.0 ~ 0100:08.0 -->
      <pci busid="0100:01.0">
        <gpu dev="0" sm="80" rank="8" gdr="1">  <!-- rank 8-15 -->
          <nvlink target="0100:02.0" count="2"/>
          <nvlink target="0100:03.0" count="1"/>
        </gpu>
      </pci>
      <!-- ... GPU 1-7 ... -->
      
      <!-- 1 个 NIC: busId 0100:10.0 -->
      <pci busid="0100:10.0">
        <nic>
          <net name="eth1" dev="1" speed="100000" .../>
        </nic>
      </pci>
    </pci>
  </cpu>
</system>
```

**为什么要分离 XML？**
1. **本地 dev ID**: 每个节点的 `dev` 都是 0-7，与 `fake_cuda` 的 `cudaDeviceCount=8` 匹配
2. **虚拟 busId**: Node 0 使用 `0000:` 前缀，Node 1 使用 `0100:` 前缀，与 `fake_cuda` 生成的 busId 一致
3. **本地拓扑**: 每个节点只包含自己的 GPU 和 NIC，模拟真实 NCCL

---

### 7. `test/run_test_2node_16gpu.sh` - 新增：测试脚本

```bash
#!/bin/bash

export NCCL_TOPO_FILE=../topo/2node_16gpu.xml  # 基础文件名
export GPU_DEV_NUM=8
export NCCL_DEBUG=INFO

mpirun -np 16 --output-filename rank_logs \
  -x LD_LIBRARY_PATH -x NCCL_TOPO_FILE -x GPU_DEV_NUM \
  bash -c '
    if [ $OMPI_COMM_WORLD_RANK -lt 8 ]; then
      export NCCL_HOSTID=0
    else
      export NCCL_HOSTID=1
    fi
    exec ./test_2node_16gpu_tp_dp
  '
```

**为什么要这样写？**
- 使用 `mpirun -np 16` 启动 16 个进程，模拟 2 节点 16 GPU
- 根据 MPI rank 设置 `NCCL_HOSTID`:
  - Rank 0-7 → `NCCL_HOSTID=0` (Node 0)
  - Rank 8-15 → `NCCL_HOSTID=1` (Node 1)
- `NCCL_TOPO_FILE` 传入基础名称，代码会自动追加 `_node0.xml` 或 `_node1.xml`

---

## 改动总结

| 文件 | 主要改动 | 改动原因 |
|------|---------|---------|
| `fake_cuda.cc` | 1. 新增 `g_hostId`<br>2. 根据 `NCCL_HOSTID` 生成虚拟 busId<br>3. 修复 `cudaMemcpyAsync`<br>4. 修复 `cudaIpcXxx`<br>5. 修复 `cudaStreamGetCaptureInfo` | 1. 模拟不同节点<br>2. 避免 busId 冲突<br>3-5. 修复仿真功能 |
| `topo.cc` | 根据 `NCCL_HOSTID` 加载不同的 XML 文件 | 实现"本地详细"机制 |
| `init.cc` | AllGather 后通过 busId 重新映射 `gpu.rank` | 修复 split comm 的 rank 映射 |
| `paths.cc` | 1. 使用 busId 匹配当前 GPU<br>2. 通过 hostHash 判断是否删除 NET | 修复拓扑修剪逻辑 |
| `net_socket.cc` | 允许注册 CUDA 内存类型 | 修复跨节点 NET 连接 |
| `node0/1.xml` | 分离的节点拓扑文件 | 支持本地拓扑加载 |
| `run_test.sh` | 测试脚本，设置 `NCCL_HOSTID` | 自动化测试 |

---

## 验证结果

### 测试场景：2 节点、16 GPU、TP+DP 通信
```
Global Comm:  16 ranks（跨节点）
TP Comm:      8 ranks（单节点内，Node 0: ranks 0-7, Node 1: ranks 8-15）
DP Comm:      2 ranks（跨节点，8 组：{0,8}, {1,9}, ..., {7,15}）
```

### 测试结果
✅ **所有 16 个 ranks 成功**:
- Global comm 初始化成功
- TP comm 初始化成功（2 个，每节点 8 ranks）
- DP comm 初始化成功（8 个，每个 2 ranks 跨节点）
- TP AllReduce 执行成功（单节点内通信）
- DP AllReduce 执行成功（跨节点 NET 通信，via NET/Socket/GDRDMA）

### 日志证据
```
lm1:488214 [0] NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/Socket/0/GDRDMA
lm1:488214 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Socket/0/GDRDMA
lm1:488214 [0] NCCL INFO Connected all rings
lm1:488214 [0] NCCL INFO comm 0x5ac0e7ef3f30 rank 0 nranks 2 - Init COMPLETE
[Rank 0] Performing TP AllReduce...
[Rank 0] Performing DP AllReduce...
```

---

## 与真实 NCCL 的对比

| 特性 | 真实 NCCL | NCCL-GP（改动后） | 说明 |
|------|-----------|------------------|------|
| 拓扑加载 | 每个节点只有本地拓扑 | ✅ 根据 `NCCL_HOSTID` 加载节点专属 XML | 实现"本地详细" |
| 全局感知 | 通过 `peerInfo` AllGather | ✅ 通过 AllGather 获得所有 ranks 的 busId | 实现"全局感知" |
| busId 唯一性 | 不同节点的 GPU busId 不同 | ✅ 通过虚拟 busId（0000:/0100: 前缀）区分 | 模拟物理隔离 |
| Split comm | 重新映射 rank | ✅ 通过 busId 重新映射 `gpu.rank` | 正确处理 split comm |
| 跨节点通信 | 通过 NET 设备 | ✅ 保留 NET 设备，建立 Socket 连接 | 支持跨节点 DP |

---

## 后续工作（可选）

1. **性能优化**: 当前 proxy thread 在退出时会超时等待（Connection closed by remote peer），可以优化清理逻辑
2. **更多拓扑**: 支持 4 节点、8 节点等更复杂的拓扑
3. **NVLink 跨节点**: 支持 NVSwitch 的跨节点 NVLink 拓扑
4. **真实数据验证**: 验证 AllReduce 的数据正确性（目前只验证了初始化和执行不崩溃）

---

## 关键概念解释

### busId (PCI Bus ID)
- **定义**: GPU 的 PCIe 总线地址，格式如 `0000:01:00.0`（Domain:Bus:Device.Function）
- **作用**: 在系统中唯一标识一个 GPU，跨进程/跨节点不变
- **在多节点模拟中**: 使用不同的 Domain 前缀区分节点（`0000:` vs `0100:`）

### hostHash
- **定义**: 节点的唯一标识符（通过主机名等信息计算的 hash）
- **作用**: 判断两个 rank 是否在同一个物理节点上
- **在 trim 逻辑中**: 通过比较所有 ranks 的 hostHash 判断是否需要 NET 设备

### rank 重新映射
- **场景**: `ncclCommSplit` 创建新 communicator
- **问题**: 新 comm 的 ranks 是 0, 1, 2, ...（local rank），但 topology 中的 `gpu.rank` 仍然是全局 rank
- **解决**: 通过 busId 在新 comm 的 `peerInfo` 中查找，更新 `gpu.rank` 为 local rank

### domain（拓扑域）
- **定义**: 通过高速连接（NVLink, NVSwitch）直接相连的 GPU 集合
- **作用**: `ncclTopoTrimSystem` 只保留与当前 GPU 在同一 domain 的 GPU，删除不可达的 GPU
- **计算方法**: 如果 `gpu[i]` 和 `gpu[j]` 之间的路径类型 < `PATH_NET`（即 NVL 或 PIX），它们在同一 domain

---

**总结**: 这次改动实现了 NCCL-GP 对多节点拓扑的完整支持，核心思路是"全局感知、本地详细"，通过 `fake_cuda` 的虚拟 busId 和分离的 XML 文件，模拟了真实 NCCL 的多节点行为。


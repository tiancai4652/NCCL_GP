# NCCL-GP 多节点支持 - 逐行改动说明（目录）

本系列文档提供了所有代码改动的逐行详细说明，适用于深入的代码审查。

---

## 📋 文档列表

### 主文档
- **本文档（目录）**: 改动概览和导航

### 详细文档（按文件分类）
1. **CodeReview-逐行改动-fake_cuda.md** - `src/graph/fake_cuda.cc`
   - 78 行改动（+69 / -9）
   - 核心：Host-aware 虚拟 busId 生成

2. **CodeReview-逐行改动-paths.md** - `src/graph/paths.cc`
   - 40 行改动（+37 / -3）
   - 核心：busId 匹配 GPU，hostHash 判断单/多节点

3. **CodeReview-逐行改动-topo.md** - `src/graph/topo.cc`
   - 30 行改动（+21 / -9）
   - 核心：根据 NCCL_HOSTID 加载节点专属 XML

4. **CodeReview-逐行改动-init.md** - `src/init.cc`
   - 68 行改动（+58 / -10）
   - 核心：AllGather 后通过 busId 重新映射 GPU rank

5. **CodeReview-逐行改动-其他文件.md** - 其他辅助文件
   - `src/misc/socket.cc` - 5 行
   - `src/transport/net_socket.cc` - 5 行
   - `test/run_test_2node_16gpu.sh` - 测试脚本
   - `test/test_2node_16gpu_tp_dp.cpp` - 测试程序

6. **CodeReview-逐行改动-拓扑文件.md** - XML 拓扑文件
   - `topo/2node_16gpu_node0.xml` - Node 0 拓扑
   - `topo/2node_16gpu_node1.xml` - Node 1 拓扑

---

## 📊 改动统计总览

| 文件 | 改动行数 | 新增 | 删除 | 关键功能 |
|------|---------|------|------|----------|
| `src/graph/fake_cuda.cc` | 78 | 69 | 9 | Host-aware busId 生成 |
| `src/graph/paths.cc` | 40 | 37 | 3 | GPU 匹配和 NET 设备判断 |
| `src/graph/topo.cc` | 30 | 21 | 9 | 动态加载节点 XML |
| `src/init.cc` | 68 | 58 | 10 | GPU rank 重新映射 |
| `src/misc/socket.cc` | 5 | 5 | 0 | Debug 日志 |
| `src/transport/net_socket.cc` | 5 | 4 | 1 | 允许 CUDA 内存注册 |
| `test/run_test_2node_16gpu.sh` | 22 | 22 | 0 | 测试脚本（新建） |
| `test/test_2node_16gpu_tp_dp.cpp` | 21 | 15 | 6 | 测试程序改进 |
| `topo/2node_16gpu_node0.xml` | 70 | 70 | 0 | Node 0 拓扑（新建） |
| `topo/2node_16gpu_node1.xml` | 92 | 46 | 46 | Node 1 拓扑（修改） |
| **总计** | **431** | **367** | **84** | - |

---

## 🎯 核心改动分类

### 类别 1: 虚拟 busId 生成（解决 busId 冲突）
**相关文件**: `src/graph/fake_cuda.cc`

**问题**: 所有进程在同一物理机上，返回相同的 busId  
**解决**: 根据 `NCCL_HOSTID` 生成不同的虚拟 busId

**改动要点**:
- 新增 `g_hostId` 全局变量
- 新增 `initFakeCudaHostId()` 初始化函数
- 修改 `cudaDeviceGetPCIBusId()` 根据节点生成 busId
- 修复 `cudaMemcpyAsync()`, `cudaIpcGetMemHandle()` 等函数

---

### 类别 2: GPU rank 重新映射（解决 split comm 崩溃）
**相关文件**: `src/init.cc`

**问题**: Split comm 后，XML 中的 `gpu.rank` 仍是全局 rank，但 `peerInfo` 只有 local ranks  
**解决**: AllGather 后通过 busId 在 `peerInfo` 中查找，更新 `gpu.rank`

**改动要点**:
- AllGather peerInfo 后新增重新映射逻辑
- 遍历本地拓扑的所有 GPU
- 通过 busId 在 peerInfo 中查找对应的 rank
- 更新 `gpu->gpu.rank` 为新 comm 的 local rank

---

### 类别 3: 动态加载拓扑文件（解决 XML 不匹配）
**相关文件**: `src/graph/topo.cc`

**问题**: 统一的全局 XML 与本地 `cudaDev` 不匹配  
**解决**: 根据 `NCCL_HOSTID` 加载节点专属的 XML 文件

**改动要点**:
- 读取 `NCCL_HOSTID` 环境变量
- 构造节点专属文件名（`_node0.xml`, `_node1.xml`）
- 加载对应的 XML 文件

---

### 类别 4: 修复拓扑修剪逻辑（解决 trim 错误）
**相关文件**: `src/graph/paths.cc`

**问题 1**: 用 `gpu.rank` 匹配当前 GPU，但 rank 已被重新映射  
**解决 1**: 改用 `comm->busId` 匹配（busId 不变）

**问题 2**: 单节点 comm 也保留了 NET 设备  
**解决 2**: 通过 `hostHash` 判断是否单节点

**改动要点**:
- 用 `gpu->id == comm->busId` 替代 `gpu->gpu.rank == comm->rank`
- 新增 `myGpuFound` 标志和错误检查
- 遍历 `peerInfo` 检查 `hostHash` 是否全部相同
- 只对单节点且 GPU count == nRanks 的 comm 删除 NET

---

### 类别 5: 网络传输支持（解决 NET 连接失败）
**相关文件**: `src/transport/net_socket.cc`

**问题**: 网络插件拒绝注册 CUDA 内存类型  
**解决**: 在 `fake_cuda` 环境中，CUDA 内存就是主机内存，允许注册

**改动要点**:
- 修改 `ncclNetSocketRegMr()` 返回 `ncclSuccess`
- 返回假的 `mhandle`（实际就是原始指针）

---

### 类别 6: 拓扑文件重构（支持分离拓扑）
**相关文件**: `topo/2node_16gpu_node0.xml`, `topo/2node_16gpu_node1.xml`

**设计原则**:
- 每个节点只包含本地 GPU 和 NIC
- `dev` 属性：本地编号 0-7
- `rank` 属性：全局 rank（node0: 0-7, node1: 8-15）
- `busid` 属性：根据节点使用不同前缀（`0000:` vs `0100:`）

---

## 📖 如何阅读详细文档

### 文档结构
每个详细文档都按以下结构组织：

1. **文件概述**
   - 文件路径
   - 改动统计
   - 主要功能

2. **改动块（Hunk）列表**
   - 每个改动块的位置
   - 上下文代码

3. **逐行注释**
   - 每一行改动的含义
   - 为什么要这样改
   - 与其他改动的关联

4. **关键数据结构**
   - 涉及的数据结构定义
   - 字段含义和作用

5. **测试验证**
   - 如何验证这个改动
   - 预期的日志输出

---

## 🔍 快速导航

### 按问题类型查找
- **busId 冲突** → `CodeReview-逐行改动-fake_cuda.md`
- **Split comm 崩溃** → `CodeReview-逐行改动-init.md`
- **拓扑加载** → `CodeReview-逐行改动-topo.md`
- **Trim 逻辑** → `CodeReview-逐行改动-paths.md`
- **NET 连接** → `CodeReview-逐行改动-其他文件.md`

### 按数据流查找
1. **启动阶段**: `fake_cuda.md` (初始化 HOSTID)
2. **拓扑加载**: `topo.md` (加载 XML) → `init.md` (AllGather)
3. **Rank 映射**: `init.md` (重新映射 GPU rank)
4. **路径计算**: `paths.md` (trim 和路径计算)
5. **连接建立**: `其他文件.md` (NET 和 P2P 连接)

---

## 💡 代码审查建议

### 审查重点
1. **数据一致性**
   - `comm->busId` vs `gpu->id` vs `peerInfo[r].busId`
   - 确保 busId 在所有地方保持一致

2. **边界条件**
   - Split comm 的 rank 重新编号
   - 单节点 vs 多节点的判断
   - GPU count vs nRanks 的各种组合

3. **错误处理**
   - `myGpuFound` 未找到的情况
   - XML 文件不存在的情况
   - busId 匹配失败的情况

4. **性能影响**
   - AllGather 后的 O(n*m) 循环（n=GPUs, m=ranks）
   - 每次初始化都会执行，但通常 n 和 m 都不大（< 100）

---

## 📝 术语对照表

| 术语 | 英文 | 含义 | 在代码中的位置 |
|------|------|------|---------------|
| 总线 ID | busId | PCI Bus ID，GPU 的唯一标识 | `comm->busId`, `gpu->id` |
| 主机哈希 | hostHash | 节点的唯一标识 | `peerInfo[r].hostHash` |
| 域 | domain | 通过 NVLink 直连的 GPU 集合 | `paths.cc:ncclTopoTrimSystem` |
| 全局 rank | global rank | 在原始 comm 中的编号（0-15） | XML 中的 `rank` 属性 |
| 本地 rank | local rank | 在新 comm 中的编号（0-7 或 0-1） | `comm->rank`, `gpu->gpu.rank` |
| 逻辑节点 ID | HOSTID | 环境变量，区分模拟的节点 | `NCCL_HOSTID=0` or `1` |
| 拆分通信器 | split comm | 从已有 comm 创建的新 comm | `ncclCommSplit()` |
| 全收集 | AllGather | 所有 ranks 交换数据 | `bootstrapAllGather()` |
| 网络设备 | NET device | 用于跨节点通信的 NIC | `system->nodes[NET]` |
| 点对点 | P2P | 进程间直接通信（GPU Direct） | `ncclTransportP2pConnect()` |

---

## 🎯 审查检查清单

在深入阅读详细文档前，请确认：
- [ ] 已理解问题背景（为什么需要多节点支持）
- [ ] 已理解核心概念（busId, hostHash, rank 重新映射）
- [ ] 已阅读快速参考文档（了解整体架构）
- [ ] 准备好测试环境（可以运行和验证）
- [ ] 了解 NCCL 的基本概念（communicator, rank, topology）

---

## 📚 推荐阅读顺序

### 第一遍：理解问题和方案
1. 阅读"CodeReview-快速参考.md"（10 分钟）
2. 阅读"CodeReview-问题诊断流程.md"（15 分钟）
3. 浏览本目录文档，了解改动分布（5 分钟）

### 第二遍：深入代码细节
1. `CodeReview-逐行改动-fake_cuda.md`（核心：busId 生成）
2. `CodeReview-逐行改动-topo.md`（加载拓扑）
3. `CodeReview-逐行改动-init.md`（rank 重新映射）
4. `CodeReview-逐行改动-paths.md`（trim 逻辑）
5. `CodeReview-逐行改动-其他文件.md`（辅助功能）
6. `CodeReview-逐行改动-拓扑文件.md`（XML 结构）

### 第三遍：验证和测试
1. 运行测试：`cd test && ./run_test_2node_16gpu.sh`
2. 查看日志：检查 busId、rank 映射、NET 连接
3. 对比预期：确认所有 16 ranks 成功

---

**下一步**: 请根据你关心的文件，查看对应的详细文档。每个文档都包含完整的 diff 和逐行注释。


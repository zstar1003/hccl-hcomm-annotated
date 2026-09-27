# 检视规范：架构合规

检视 `src/` 变更时逐条对照。权威来源：根 `AGENTS.md` 第 3 节、`docs/zh/architecture/architecture-brief.md`（「3 软件分层逻辑」+ 末尾「软件架构约束说明」）。改 `src/`、`include/`、`pkg_inc/` 前必须先读 architecture-brief。

## 分层依赖方向（硬性）

依赖方向自上而下：`coll_comm_ops`（hccl）→ `coll_communicator_mgr` → `base_comm`。

- 禁止下层 `#include` 上层头文件或调用上层接口（`base_comm` ↛ `coll_communicator_mgr`；两者 ↛ `coll_comm_ops`）
- 新增类/函数先定层级，只调用同层或更下层

## 控制面/数据面分离

- 资源管理、拓扑查询（控制面）与数据搬运、同步（数据面）接口独立演进、互不耦合
- 不得在数据面原语中引入控制面强耦合；控制面不得依赖具体数据面算子实现

## HCCL 与 HCOMM 解耦

- HCOMM 不得 `#include` HCCL 私有头；不得引入对 `cann/hccl` 的编译期硬依赖
- 跨仓调用走符号表 + `dlsym`

## legacy 不持续演进

- 禁止在 `src/legacy/` 新增功能/算子/接口
- legacy 仅 bug 修复与兼容维护；新特性落 `base_comm/` 或 `coll_communicator_mgr/`

## 目录与结构对齐

- 目录命名对齐 architecture-brief 3.2 目标结构（如 `base_comm/resource` 非 `resources`）
- `src/` 重命名/移动时同步检查：`CMakeLists.txt`、`#include` 相对路径、`classify_rule.yaml`、`blacklist.txt`、`cmake/`、`build.sh`
- 重命名 PR 检查是否混入 brief 未明确要求的无关修改

## 高风险区（变更时提高检视强度）

| 区域 | 风险 |
|------|------|
| transport/RDMA（`transport_*`） | QP 管理、内存注册、RDMA 操作 |
| socket/网络（`*_network.cc`、`*_socket.h`） | 连接生命周期、端口、超时 |
| endpoint/channel 初始化（`*_endpoint.cc`、`*_channel.cc`） | 资源获取顺序、部分失败清理 |
| 内存注册（`*_mem.cc`、`rma_buffer_mgr.h`） | MR 生命周期、buffer 重叠、key 管理 |
| 全局状态（`*_res_handler.cc`、`*_res_registry.cc`） | 全局 map 线程安全、handle 生命周期 |

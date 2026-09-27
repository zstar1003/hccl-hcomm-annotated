# HCOMM Agent 规则

本文件是 HCOMM 仓的 AI Agent 治理主入口，面向各类支持 AGENTS.md 标准的 AI 编程工具。子目录如存在自己的 `AGENTS.md`，以更近的文件补充本文件；冲突时先遵守用户当前明确要求，再遵守更近的子目录规则。

> 本文件只给**必须知道的硬约束与入口**，详细内容通过链接渐进式披露。改动代码前，请先读第 3 节架构约束。

## 1. 仓库定位

HCOMM（Huawei Communication）是 HCCL 的通信基础库，提供通信域与通信资源管理能力，是 CANN 集合通信栈的底层基座。HCCL = HCCL 算子库（`cann/hccl`）+ HCOMM 通信基础库（本仓 `cann/hcomm`），两仓通过 `dlsym` 动态加载解耦。详见 [`README.md`](./README.md) 与 [架构简介](./docs/zh/architecture/architecture-brief.md)。

## 2. 目录结构

```text
src/
├── base_comm/                 # 基础通信层（L3，最底层，不可反向依赖上层）
├── coll_communicator_mgr/     # 集合通信域管理 HCCM（L2，依赖 base_comm）
└── legacy/                    # ⚠️ 历史兼容目录，不承接新特性
include/      # 对外头文件（hccl/、ccu/、hcomm_primitives.h 等）
pkg_inc/      # 包间接口头文件（hccl/、hcomm/，不对外承诺稳定）
test/         # ut / st / stub
docs/         # 资料文档
build.sh      # 一键编译脚本
```

目标目录结构与各层职责详见 [架构简介 3.2 节](./docs/zh/architecture/architecture-brief.md)。

## 3. 软件架构与架构约束（核心）

> **架构权威来源**：[`docs/zh/architecture/architecture-brief.md`](./docs/zh/architecture/architecture-brief.md)。改动 `src/`、`include/`、`pkg_inc/` 前必须先读其「3 软件分层逻辑」与末尾「软件架构约束说明」。

### 分层

| 软件层次 | 仓位置 |
|----|--------|
| HCCL 集合通信算子 | `cann/hccl` |
| HCOMM 集合通信域管理（HCCM） | 本仓 `src/coll_communicator_mgr` |
| HCOMM 基础通信 | 本仓 `src/base_comm` |

依赖方向自上而下：`coll_comm_ops`（hccl）→ `coll_communicator_mgr` → `base_comm`。

### 架构约束（硬性，不可违反）⭐

| 约束 | AI Agent 行为要求 |
|------|------------------|
| **分层依赖方向**：上层依赖下层，下层不能反向依赖上层（`base_comm` ↛ `coll_communicator_mgr`；两者 ↛ `coll_comm_ops`） | 禁止下层 `#include` 上层头文件或调用上层接口；新增类/函数先定层级，只调用同层或更下层 |
| **控制面/数据面分离**：资源管理、拓扑查询（控制面）与数据搬运/同步（数据面）接口独立演进、互不耦合 | 不得在数据面原语中引入控制面强耦合；控制面不得依赖具体数据面算子实现 |
| **HCCL 与 HCOMM 解耦**：HCCL 通过 `dlsym` 动态加载 HCOMM 接口，两仓独立编译、独立版本演进 | HCOMM 不得 `#include` HCCL 私有头；不得引入对 `cann/hccl` 的编译期硬依赖；跨仓调用走符号表 + `dlsym` |
| **legacy 不持续演进**：`legacy/` 仅历史兼容，不承接新特性 | **禁止在 `src/legacy/` 新增功能/算子/接口**；legacy 仅 bug 修复与兼容维护；新特性落 `base_comm/` 或 `coll_communicator_mgr/` |

### 对外 API 分层

| 层次 | 头文件 | 面向 |
|------|------|------|
| L2-comm | `include/hccl/hccl_comm.h` | AI 框架层（通信域创建） |
| L2-res-rank_graph | `include/hccl/hccl_res.h`、`include/hccl/hccl_channel.h`、`include/hccl/hccl_rank_graph.h` | 算子开发者（拓扑查询、资源获取） |
| L3-prim | `include/hcomm_primitives.h` | 算子/通信库开发者（数据搬运 + 同步） |
| L3-res | `include/hcomm_res.h`、`include/hcomm_res_defs.h`、`include/hcomm_channel.h` | 通信库开发者（设备/通道/内存资源） |
| CCU | `include/ccu/`（`ccu_primitives.hpp`、`ccu_res.h`、`ccu_launch.h`） | CCU 算子开发者 |

`include/` 变更需向后兼容；`pkg_inc/` 仅供 HCOMM↔HCCL、GE 等包间使用，不对外稳定。完整 API 分层关系见 [架构简介 3.3 节](./docs/zh/architecture/architecture-brief.md)。

## 4. 构建与测试

```bash
bash build.sh --pkg            # 编译 host 包（默认）
bash build.sh --pkg --full     # 编译 host + device 全量包
bash build.sh -u               # 编译并运行 UT（= --ut）
bash build.sh -s               # 编译并运行 ST
bash build.sh --ut --noexec    # 编译 UT 但不执行用例（--noexec 须配合 --ut/--st）
bash build.sh -j64             # 并行编译
```

输出 `build_out/cann-hcomm_<version>_linux-<arch>.run`。完整选项、环境准备与上板测试见 [`docs/zh/build/build.md`](./docs/zh/build/build.md)。推送前优先本地验证 `--pkg` + `--ut --noexec` + `--st --noexec`。

## 5. 编码规范

- 命名：类/函数 PascalCase；成员变量 `camelCase_`（小驼峰+后缀下划线）；常量与宏 `UPPER_SNAKE_CASE`。
- 风格：遵循根目录 `.clang-format`（120 列、4 空格、指针右对齐、K&R 大括号）；C++14。
- 静态告警：代码须通过 CANN 静态检查要求（CI codecheck 阶段校验），编译无告警。
- pre-commit：clang-format v18.1.8 + OAT 合规检查；新增源文件须带 CANN-2.0 许可头。

参考：[CANN 编码规范](https://gitcode.com/cann/community/tree/master/contributor/coding-standards)、[CANN CI 指南](https://gitcode.com/cann/community/blob/master/contributor/repository/ci-guide.md)、[pre-commit 指导](./docs/zh/build/pre-commit-guide.md)、`.clang-format`、`OAT.xml`。

## 6. 贡献流程

- 简单问题：Issue → 认领 → PR → Committer 检视 → `/lgtm` + `/approve` 合入。
- 新功能：Requirement Issue → SIG 决策 → `docs/zh/rfcs/` RFC 评审 → 实现（含 UT+ST）→ 检视合入。
- 所有 PR 必须关联 Issue，描述按 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` 填写。
- 从代码获取到 PR 合入的贡献链路操作（代码同步、Issue 查重创建、PR 提交触发 CI、CI 轮询与失败修复、检视意见处置）：使用仓内贡献流程 skill（[`.agents/skills/hcomm-contribute/`](./.agents/skills/hcomm-contribute/)）；本地构建与测试命令以本文件第 4 节与 [`docs/zh/build/build.md`](./docs/zh/build/build.md) 为准。

详见 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。

## 7. Agent 工作原则

- 优先小而可审查的变更；除非用户明确要求，避免大范围重构。
- 编辑前先定位文件，用 3-6 条说明计划。
- 不确定 API、配置、路径或事实时，先搜索仓库或查证，不要臆造。
- 改动 `src/` 前先对照第 3 节架构约束：是否违反分层依赖？是否在 `legacy/` 新增特性？是否破坏控制面/数据面分离？是否引入对 `cann/hccl` 的编译期硬依赖？
- 严禁把密钥、token、密码、私钥、`.env` 值或凭据写入代码、日志或回复。
- 除非用户要求，不新增遥测、分析上报或额外网络调用。
- 行为变更应在项目已有测试体系下补充或更新测试；优先跑最快相关验证。
- 涉及 `src/` 目录重命名/移动时，同步检查 `CMakeLists.txt`、测试 include 路径、`#include` 相对路径、`classify_rule.yaml`、`blacklist.txt`，并清理 build 目录后重新验证。
- 破坏性命令、`git commit`、`git push` 必须得到用户明确许可。
- 默认用中文解释；输出保持简洁、具体、可复制。

---

开发提交前自检与检视他人 PR：使用仓内检视 skill（[`.agents/skills/hcomm-review/`](./.agents/skills/hcomm-review/)），编码时遵循其 `references/` 下的检视规范（编码安全、对外 API、架构合规、PR 完备性）。

*架构约束以 [`docs/zh/architecture/architecture-brief.md`](./docs/zh/architecture/architecture-brief.md) 为权威来源；贡献流程以 [`CONTRIBUTING.md`](./CONTRIBUTING.md) 为准。个人临时偏好放 `AGENTS.local.md`（已 gitignore）。*

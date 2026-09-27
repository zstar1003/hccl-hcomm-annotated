# HVRM Insight 使用指南

## 1. 概述

HVRM Insight 是一个分布式算子可视化分析工具，用于对 HCCL 集合通信算子执行过程中的图结构、内存快照和校验报错信息进行统一展示和联动分析。工具包含三个页面：

| 页面 | 标签 | 用途 |
|------|------|------|
| **Dashboard** | 总览 | 浏览数据集、选择 Rank、查看统计信息，是使用工具的起点 |
| **MemView** | 关联 | 内存时间线与 DAG 任务图联动展示，是排查问题的核心分析页面 |
| **Analytic** | 报错 | 集中展示校验错误，支持一键跳转到问题现场 |

---

## 2. 安装与启动

### 2.1 运行前准备

HVRM Insight 依赖 Checker 插件输出的数据文件。使用前，请确保已完整运行过 Checker，并在 Checker 的配置文件中开启以下两个数据输出开关：

Checker配置文件默认位于`/path/to/hccl_vm_install/plugin/checker/manifest.json`。

```json
"setting": {
    "enable_insight_dump": true,
    "enable_memory_snapshot_dump": true
}
```

- **`enable_insight_dump`**：开启后，Checker 运行时会输出 Insight 所需的图结构和校验结果文件。
- **`enable_memory_snapshot_dump`**：开启后，Checker 运行时会输出各 Rank 的内存快照文件。

这两个开关默认为 `false`。如未开启，Insight 工具将无法读取到完整的分析数据。

### 2.2 数据目录结构

工具从 `data/` 目录读取数据集，每个数据集是一个独立的子目录：

```text
data/
└── <dataset_name>/
    ├── manifest.json        # 数据集元信息（必须）
    ├── graph/               # 图结构文件（DAG）
    ├── memory/              # 内存快照文件
    └── validation/
        └── issues.msgpack   # 校验报错记录（可选）
```

缺少可选目录时，对应页面会显示空状态。

### 2.3 方式一：通过 hccl-vm 插件（推荐）

```bash
# 安装并启动后端服务
hccl-vm plugin install insight
# 安装成功后，终端会输出访问地址（默认 `http://localhost:8000`），在浏览器中打开即可使用。

# 再次查看访问地址
hccl-vm plugin run @insight

# 卸载插件（会同时停止后端服务）
hccl-vm plugin uninstall @insight
```

### 2.4 方式二：本地直接启动

```bash
# 进入 insight 插件目录
python3 serve.py
```

后端默认监听 `http://localhost:8000`，同时提供前端页面和数据 API。

### 2.5 验证启动成功

浏览器打开地址后，如果能看到 Dashboard 页面的数据集列表，说明启动成功。如果页面空白，请检查 `data/` 目录是否存在且包含 `manifest.json` 文件。

---

## 3. 场景演示

### 3.1 场景一：选择数据集并查看概览（Dashboard）

浏览所有数据集，点击选择目标数据集，查看算子信息和统计数据，然后进入关联分析页面。

![Dashboard 操作演示](insight_image/dashboard.gif)

### 3.2 场景二：联动分析内存与任务图（MemView）

通过内存时间线和 DAG 任务图的联动，定位某个 Step 的内存操作、查看节点详情、追溯数据来源。

![MemView 操作演示](insight_image/memView.gif)

### 3.3 场景三：查看报错并跳转到问题现场（Analytic）

在 Analytic 页面或 MemView 左侧报错列表中查看错误详情，然后一键跳转到关联的 DAG 节点或内存上下文。

![错误定位操作演示](insight_image/error.gif)

---

## 4.页面详解

### 4.1 Dashboard 页面

Dashboard 是使用工具的起点，页面从左到右分为三个区域：

- **左侧**：算子摘要和 Rank 树。Rank 树支持勾选 / 取消勾选，选择结果会同步到所有页面。
- **中间**：数据集列表。点击某一行选中数据集，左侧和右侧同步刷新。
- **右侧**：选中数据集的详情面板，展示统计信息和跳转按钮。

![Dashboard 总览](insight_image/dashboard_overview.png)

选中数据集后，右侧展示详细统计：

![Dashboard 已选数据集](insight_image/dashboard_selected.png)

详情面板底部提供两个跳转入口：

- **`进入关联视图`**：跳转到 MemView，带入当前数据集和 Rank 选择。
- **`查看诊断`**：跳转到 Analytic，查看该数据集的所有校验错误。

### 4.2 MemView 页面

MemView 将内存时间线和 DAG 任务图放在同一视图中联动展示，是排查问题的核心页面。

![MemView 总览](insight_image/memview_overview.png)

**页面布局：**

- **左侧**：Rank 选择树、报错列表（Issues）、搜索面板。
- **上半区**：内存时间线 —— 按 Step 展示各 Rank 的 buffer 状态演化。
- **下半区**：DAG 任务图 —— 按 Rank / Stream 泳道展示任务节点和依赖关系。
- **右侧**：详情栏 —— 展示当前选中 Step 或节点的详细信息。

![DAG 节点视图](insight_image/memview_dag_node.png)

#### 4.3 核心联动机制：Step、Rank、Task 三者联动

- 在时间线上选中一个 Step → DAG 同步高亮对应节点。
- 在 DAG 中点击一个节点 → 右侧切到节点详情。
- 从右侧点击跳转按钮 → 时间线同步定位到对应 Step。

**右侧详情栏**有两种模式，根据是否选中 DAG 节点自动切换：

- **Step 详情模式**：展示当前 Step 的内存操作列表（Task Memory Ops）和 Buffer 切片详情（Buffer Layout）。Buffer Layout 中的每个切片可展开查看数据来源语义，支持点击跳转。
- **节点详情模式**：展示节点属性、父/子节点关系、跨阶段节点映射、以及该节点关联的内存操作。

**其他功能：**

- **底部播放器**：支持按 Step 顺序浏览，可拖拽滑块快速跳转，播放速度为 1 秒 / Step。
- **Focus 视图**：选中 Step 后 DAG 自动进入局部视图，只显示当前 Step 相关节点及其上下文，适合节点密集时缩小范围。
- **子图浏览**：节点包含子图时，可在右侧点击 `打开子图` 进入查看更细粒度的任务结构。
- **Stage 切换**：通过顶部下拉框切换图阶段（如 `input_graph` / `input_task_queues`）
- **搜索**：支持按 taskId / taskType / notifyId 搜索 DAG 节点，结果点击即定位。

### 4.4 Analytic 页面

Analytic 集中展示校验过程中发现的所有错误，页面分为三个区域：

![Analytic 报错详情](insight_image/analytic_issue_detail.png)

- **左侧**：数据集与 Rank 选择。
- **中间**：报错列表，每条错误展示标题、严重程度标签（颜色区分）、关键字段摘要和原始错误码。
- **右侧**：选中错误的详情面板，包括错误基本信息、涉及的 Rank、关联节点信息、Slice 详情（精确定位数据不一致的内存位置）、补充信息和原始 JSON。

详情面板顶部提供跳转按钮：

- **`查看 memView`**：跳转到 MemView，定位到错误关联的内存 Step。
- **`查看 dagView`**：跳转到 MemView，直接定位到错误关联的 DAG 节点。

跳转时自动带入数据集、Rank 和节点位置信息，到达 MemView 后无需重新选择即可开始排查。

---

## 5. 常见使用流程

### 5.1 从 Dashboard 开始分析

1. 在数据集列表中点击目标数据集。
2. 在左侧 Rank 树中勾选需要查看的 Rank。
3. 在右侧确认数据集的文件数量和错误统计。
4. 点击 `进入关联视图` 进入 MemView。

### 5.2 在 MemView 中定位内存问题

1. 在时间线上点击目标 Step（或使用底部播放器跳转）
2. 在右侧 `Task Memory Ops` 查看该 Step 的内存操作。
3. 在 `Buffer Layout` 中选择 Rank 和 Buffer Type，展开 slice 查看数据来源。
4. 点击 `定位 Task` 或语义卡片跳转到对应 DAG 节点。

### 5.3 从 DAG 节点反查内存

1. 在 DAG 中点击目标节点。
2. 在右侧查看 `节点 Memory Ops`。
3. 点击 `跳转到 Step` 回到时间线，或点击父子节点继续沿依赖链排查。

### 5.4 从报错快速定位

1. 在 Analytic 页面或 MemView 左侧报错列表中点击一条错误。
2. 查看详情中的关联节点和 Slice 信息。
3. 点击 `查看 memView` 或 `查看 dagView` 跳转到问题现场。

---

## 6. 使用技巧

- 先在 Dashboard 确认数据集和 Rank 范围，再进入 MemView 分析。
- 看内存问题时，优先用 `Task Memory Ops` 和 `Buffer Layout` 配合排查。
- 看节点依赖关系时，优先在 DAG 中选中节点，再用右侧父子关系和节点映射跳转。
- DAG 节点太密集时，先用 Focus 视图缩小范围，再逐步展开。
- MemView 左侧的报错列表可以在不离开当前页面的情况下快速浏览并定位错误节点。

---

## 7. 术语说明

| 术语 | 说明 |
|------|------|
| **Rank** | 单个运算单元（如 NPU），每个 Rank 独立执行计算 |
| **DAG** | 有向无环图，描述任务节点之间的依赖关系 |
| **Step** | 时间线中的一个时间步，对应一次内存状态快照 |
| **Buffer** | 内存中的一块数据区域，用于存放计算或通信的数据 |
| **Stage** | 图的一个阶段，如 `input_graph` 或 `input_task_queues` |
| **Stream** | 任务执行流，同一个 Stream 内的任务按顺序执行 |
| **Issue** | 校验过程中发现的错误或异常记录 |

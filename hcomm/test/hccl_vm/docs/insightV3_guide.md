# HVRM Insight V3 使用指南

## 1. 概述

HVRM Insight V3 当前主要支持 `DAGView` 任务图查看。本文档围绕 DAGView 介绍数据准备、页面访问、数据集选择、DAG 图浏览和节点检索等常用操作。

其他页面及高级联动能力暂未实现。

---

## 2. 前置准备

使用 Insight V3 前，请确认 Checker 已完成运行，并已生成 Insight 数据。

如需配置 Checker 数据输出，请确认 Checker 配置文件中已开启 Insight dump：

```json
# 配置文件位于 /pathto/hccl_vm_install/plugin/checker/manifest.json

{
  ...
  "setting": {              // Checker插件配置项
      ...
      "enable_insight_dump": true,         // 是否启用可视化数据输出（默认为关闭状态）
      "enable_memory_snapshot_dump": false  // 是否启用可视化内存快照数据输出（默认关闭，仅支持老Checker，需要先开启可视化数据输出"enable_insight_dump"）
  }
}
```

DAGView 至少需要以下数据文件：

```text
<dataset_name>/
├── manifest.json
└── graph/
    ├── graph.msgpack
    └── layout.msgpack
```

其中，`manifest.json` 用于读取数据集元信息，`graph.msgpack` 和 `layout.msgpack` 用于绘制 DAG 任务图。

---

## 3. 编译与安装 Insight 插件

Insight V3 当前前端源码位于：

```text
{hccl_vm目录}/src/plugin/insight/frontend_v3
```

在安装 Insight 插件前，请先完成前端编译，再执行 HCCL VM 的构建安装流程。

注意：编译 Insight 可视化前端需要依赖 `Node.js` 和 `npm`。建议使用 `Node.js 20.19.0` 或以上 与 `npm 10.x` 或以上。

推荐步骤如下：

1. 进入前端目录。

   ```bash
   cd {hccl_vm目录}/src/plugin/insight/frontend_v3
   ```

2. 安装前端依赖。

   ```bash
   npm install
   ```

3. 编译前端产物。

   ```bash
   npm run build
   ```

   前端编译完成后，会生成 `src/plugin/insight/dist/` 目录。

4. 回到 HCCL_VM 根目录，执行 `build.sh` 编译并安装 hccl-vm。

   ```bash
   cd {HCCL_VM目录}
   bash build.sh --package-path <ASCEND_CANN_PATH> --hcomm-path <HCOMM_CODE_PATH>
   ```

   如需打包安装目录，可附加 `--pkg`。若当前构建场景要求 AICPU / AIV / FULL 模式，可按项目常规流程追加 `--aicpu`、`--aiv` 或 `--full` 参数。

---

## 4. Insight 插件配置与启停

Insight 插件配置文件位于：

```text
hccl_vm_install/plugin/visualization/insight/manifest.json
```

当前默认配置如下：

```json
{
    "name": "insight",
    "version": "1.0.0",
    "entry": "python3 server.py",
    "dependency": {
        "min_core_version": "1.0.0"
    },
    "setting": {
        "dist_path": "./dist",
        "data_path": "../../checker/data/insight",
        "topo_config_path": "../../../../../asset/cluster_model/config/cluster",
        "port": 8080
    }
}
```

常用字段说明如下：

| 字段 | 说明 |
|------|------|
| `entry` | Insight 插件启动命令，当前通过 `python3 server.py` 启动服务 |
| `setting.dist_path` | 前端静态资源目录，默认读取 `./dist` |
| `setting.data_path` | Insight 数据目录，默认指向 Checker 输出的 `data/insight` |
| `setting.topo_config_path` | 集群拓扑配置目录 |
| `setting.port` | Insight 服务端口，默认 `8080` |

若需要修改端口或数据目录，可直接编辑该 `manifest.json` 中的 `setting` 字段。

> 警告：Insight 默认通过 `python3 server.py` 绑定在 `127.0.0.1`。如需监听指定 IP，而不是默认的 `localhost(127.0.0.1)`，请进入 `hccl_vm_install/plugin/visualization/insight` 目录，手动执行：
>
> ```bash
> python3 server.py --host <ip> --port <port>
> ```
>
> 例如：
>
> ```bash
> cd hccl_vm_install/plugin/visualization/insight
> python3 server.py --host 0.0.0.0 --port 8080
> ```

插件安装与卸载命令如下：

```bash
# 安装并启动 Insight 插件
hccl-vm plugin install @insight

# 卸载 Insight 插件
hccl-vm plugin uninstall @insight
```

---

## 5. 打开 Insight

若 Insight 服务已启动，请直接在浏览器中打开服务地址，例如：

```text
http://localhost:8080
```

若需要通过插件启动 Insight，可执行：

```bash
hccl-vm plugin install @insight
```

插件启动后，终端会输出访问地址。请在浏览器中打开该地址进入 Insight 页面。

![启动 Insight 并打开页面](insight_image/V3_image/launch.gif)

---

## 6. 选择数据集

打开 Insight 后，默认进入 `总览` 页面。请按以下步骤选择需要分析的数据集：

1. 在中间的数据集表格中查看数据集列表。
2. 点击目标数据集。
3. 在左侧 Rank 树中确认需要查看的 Rank。
4. 如 DAG 图规模较大，可先减少 Rank 选择范围。
5. 点击右侧 `进入关联视图`。

![选择数据集并进入关联视图](insight_image/V3_image/dagView_V3.gif)

建议首次分析时先选择少量 Rank，确认分析方向后再逐步扩大 Rank 范围。

---

## 7. 进入 DAGView

进入 `关联` 页面后，重点查看下半部分的 `DAGView 任务视图`。

页面主要区域如下：

| 区域 | 功能 |
|------|------|
| 左侧面板 | 选择 Rank、搜索节点 |
| 中间下方 | 展示 DAG 任务图 |
| 右侧详情栏 | 查看当前节点详情 |

若上方内存视图为空，不影响下方 DAGView 的查看和使用。

---

## 8. 查看 DAG 任务图

DAG 任务图由泳道、节点和箭头组成：

| 元素 | 说明 |
|------|------|
| 泳道 | 表示一个执行队列，通常显示为 `Rank / Stream / Queue` |
| 节点 | 表示一个任务，例如数据搬运、Reduce、Record、Wait 等 |
| 箭头 | 表示任务之间的依赖关系 |
| Loop 虚线框 | 表示一个 Loop 区域，会用虚线框圈住该 Loop 内相关节点，并在框外显示 `Loop` 标记 |

常见节点类型如下：

| 节点类型 | 含义 |
|----------|------|
| `TRANS_MEM` | 数据搬运任务 |
| `REDUCE` | Reduce 任务 |
| `RECORD` | Notify record 任务 |
| `WAIT` | Notify wait 任务 |
| `CCU_GRAPH` | CCU 图任务 |
| `AIV_GRAPH` | AIV 图任务 |

推荐查看方式：

1. 按箭头方向查看任务依赖顺序。
2. 点击关注的任务节点。
3. 在右侧详情栏查看该节点所属的 Rank、Stream、Queue 及节点详细信息。
4. 继续查看该节点的父节点和子节点，追踪上下游依赖。

关于 Loop 展示，建议额外关注以下信息：

1. 若 DAG 中存在 Loop，页面会自动在任务图上叠加 Loop 虚线框。
2. 虚线框会覆盖该 Loop 的起点、终点及循环体内部节点，便于快速识别循环边界。
3. 点击 Loop 胶囊后，可直接定位到对应的 Loop Start 节点。
4. 选中 Loop 内部节点后，右侧详情栏会展示该节点所属的 Loop 信息和嵌套链路。

![在 DAGView 中浏览任务图并查看节点详情](insight_image/V3_image/dagView_V3_2.gif)

---

## 9. 画布操作

DAGView 画布支持以下操作：

| 操作 | 说明 |
|------|------|
| 移动画布 | 拖拽空白区域 |
| 放大 / 缩小 | 使用鼠标滚轮，或点击右下角 `+` / `-` |
| 重置缩放 | 点击右下角百分比按钮 |
| 选中节点 | 点击目标节点 |

当图规模较大时，建议先缩小 Rank 选择范围，再放大局部区域查看依赖。

---

## 10. 查看节点详情

点击 DAG 节点后，右侧详情栏会展示节点信息。常用信息如下：

| 区块 | 说明 |
|------|------|
| 节点概览 | 节点 ID、任务类型、Rank、Stream、Queue |
| Loop 信息 | 当前节点所属 Loop、Loop 次数、指令范围、Loop 边界等 |
| 父节点 | 当前节点依赖的上游节点 |
| 子节点 | 依赖当前节点的下游节点 |
| 节点语义 | Notify、Task 元数据、Memory Slices 等 |
| 原始 JSON | 当前节点的完整原始信息 |

排查依赖关系时，建议优先查看 `父节点` 和 `子节点`；排查循环结构时，可重点查看 `Loop 信息`；排查数据搬运相关问题时，可重点查看 `Memory Slices`。

其中 `Memory Slices` 区块有以下展示规则：

1. 普通内存 Slice 会显示 `rank / type / offset / size`。
2. 若 Slice 属于 `MS_CCU`，Insight V3 会优先显示 `MSID`，而不是底层抽象 offset。
3. 对于 batch 类任务，若存在多个 `MS_CCU` Slice，会自动合并为一个汇总卡片。
4. 汇总卡片会显示类似 `共计使用8个MSID` 的说明。
5. 展开 `MSID 明细` 后，可逐项查看每个 `MSID` 对应的 `id` 和 `size`。

---

## 11. 搜索节点

左侧 `搜索` 面板可用于快速定位 DAG 节点。

支持的搜索字段如下：

| 字段 | 使用场景 |
|------|----------|
| `taskId` | 已知节点 ID 时使用 |
| `taskType` | 按任务类型查找，例如 `TRANS_MEM` |
| `notifyId` | 按 notify id 查找 |

操作步骤：

1. 在左侧搜索面板选择搜索字段。
2. 输入完整关键词。
3. 点击搜索结果。
4. DAGView 会自动定位并选中对应节点。

当前搜索为精确匹配。若未找到结果，请确认关键词是否完整，以及目标 Rank 是否已被勾选。

---

## 12. 推荐分析流程

建议按以下流程进行 DAGView 分析：

1. 在 `总览` 页面选择数据集。
2. 在左侧 Rank 树中选择少量 Rank。
3. 点击 `进入关联视图`。
4. 在 DAGView 中查看整体节点和依赖箭头。
5. 点击关注节点并查看右侧详情。
6. 通过父节点和子节点追踪上下游依赖。
7. 使用搜索功能快速定位已知节点。

---

## 13. 常见问题

### 13.1 页面没有数据

请确认 Insight 服务地址是否正确，并确认 Checker 已生成 Insight 数据。

### 13.2 进入关联页后 DAG 为空

请确认数据集中存在以下文件：

```text
graph/graph.msgpack
graph/layout.msgpack
```

### 13.3 DAG 图过于密集

建议先减少左侧 Rank 勾选数量，只查看部分 Rank，再放大局部区域进行分析。

### 13.4 搜索不到节点

请确认搜索关键词完整，并确认目标节点所在 Rank 已被勾选。

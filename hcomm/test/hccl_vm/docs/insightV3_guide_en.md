# HVRM Insight V3 User Guide

## 1. Overview

HVRM Insight V3 supports the `DAGView` task graph viewer. This document covers data preparation, page access, dataset selection, DAG graph browsing, and node search.

Other pages and advanced linkage features are not yet available.

---

## 2. Prerequisites

Before using Insight V3, confirm that the Checker has completed execution and has generated Insight data.

To configure Checker data output, confirm that the Checker configuration file has enabled Insight dump:

```json
# The configuration file is located at /pathto/hccl_vm_install/plugin/checker/manifest.json

{
  ...
  "setting": {              // Checker plugin configuration items
      ...
      "enable_insight_dump": true,         // Enable or disable visualization data output (disabled by default)
      "enable_memory_snapshot_dump": false  // Enable or disable visualization memory snapshot data output (disabled by default, only supported by legacy Checker, requires enabling visualization data output "enable_insight_dump" first)
  }
}
```

DAGView requires at least the following data files:

```text
<dataset_name>/
├── manifest.json
└── graph/
    ├── graph.msgpack
    └── layout.msgpack
```

The `manifest.json` file provides dataset metadata. The `graph.msgpack` and `layout.msgpack` files render the DAG task graph.

---

## 3. Compile and Install the Insight Plugin

The Insight V3 frontend source code is located at:

```text
{hccl_vm directory}/src/plugin/insight/frontend_v3
```

Before installing the Insight plugin, compile the frontend first, then run the HCCL VM build and installation process.

Note: Compiling the Insight visualization frontend requires `Node.js` and `npm`. Use `Node.js 20.19.0` or later and `npm 10.x` or later.

The recommended steps are as follows:

1. Navigate to the frontend directory.

   ```bash
   cd {hccl_vm directory}/src/plugin/insight/frontend_v3
   ```

2. Install frontend dependencies.

   ```bash
   npm install
   ```

3. Build the frontend output.

   ```bash
   npm run build
   ```

   After the frontend compilation completes, the system generates the `src/plugin/insight/dist/` directory.

4. Return to the HCCL_VM root directory, and run `build.sh` to compile and install hccl-vm.

   ```bash
   cd {HCCL_VM directory}
   bash build.sh --package-path <ASCEND_CANN_PATH> --hcomm-path <HCOMM_CODE_PATH>
   ```

   To package the installation directory, append `--pkg`. If the current build scenario requires AICPU / AIV / FULL mode, append the `--aicpu`, `--aiv`, or `--full` parameter following the standard project procedure.

---

## 4. Insight Plugin Configuration and Start/Stop

The Insight plugin configuration file is located at:

```text
hccl_vm_install/plugin/visualization/insight/manifest.json
```

The default configuration is as follows:

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

The common fields are described below:

| Field | Description |
|------|------|
| `entry` | The Insight plugin startup command. The system starts the service using `python3 server.py`. |
| `setting.dist_path` | The frontend static resource directory. The default value is `./dist`. |
| `setting.data_path` | The Insight data directory. The default points to the Checker output directory `data/insight`. |
| `setting.topo_config_path` | The cluster topology configuration directory. |
| `setting.port` | The Insight service port. The default value is `8080`. |

To modify the port or data directory, edit the `setting` field in `manifest.json` directly.

> Warning: Insight binds to `127.0.0.1` by default using `python3 server.py`. To listen on a specific IP address instead of the default `localhost(127.0.0.1)`, navigate to the `hccl_vm_install/plugin/visualization/insight` directory and run the following command manually:
>
> ```bash
> python3 server.py --host <ip> --port <port>
> ```
>
> For example:
>
> ```bash
> cd hccl_vm_install/plugin/visualization/insight
> python3 server.py --host 0.0.0.0 --port 8080
> ```

The plugin installation and uninstall commands are as follows:

```bash
# Install and start the Insight plugin
hccl-vm plugin install @insight

# Uninstall the Insight plugin
hccl-vm plugin uninstall @insight
```

---

## 5. Open Insight

If the Insight service is running, open the service address in the browser. For example:

```text
http://localhost:8080
```

To start Insight through the plugin, run the following command:

```bash
hccl-vm plugin install @insight
```

After the plugin starts, the terminal outputs the access address. Open that address in the browser to access the Insight page.

![Start Insight and open the page](insight_image/V3_image/launch.gif)

---

## 6. Select a Dataset

After opening Insight, the system displays the `Overview` page by default. Follow these steps to select a dataset for analysis:

1. View the dataset list in the middle dataset table.
2. Click the target dataset.
3. Confirm the target Rank in the left Rank tree.
4. If the DAG graph is large, reduce the Rank selection range first.
5. Click `Enter Correlation View` on the right side.

![Select a dataset and enter the correlation view](insight_image/V3_image/dagView_V3.gif)

Select a small number of Ranks during the first analysis. Confirm the analysis direction, then expand the Rank range gradually.

---

## 7. Enter DAGView

After entering the `Correlation` page, focus on the `DAGView Task View` in the lower section.

The main page areas are as follows:

| Area | Function |
|------|------|
| Left panel | Select Ranks and search for nodes |
| Center bottom | Display the DAG task graph |
| Right detail panel | View the current node details |

If the upper memory view is empty, the DAGView below remains fully functional.

---

## 8. View the DAG Task Graph

The DAG task graph consists of swim lanes, nodes, and arrows:

| Element | Description |
|------|------|
| Swim lane | Represents an execution queue, displayed as `Rank / Stream / Queue` |
| Node | Represents a task, for example data transfer, Reduce, Record, or Wait |
| Arrow | Represents a dependency between tasks |
| Loop dashed box | Represents a Loop region. A dashed box encloses the related nodes within the Loop, and a `Loop` label appears outside the box. |

The common node types are as follows:

| Node Type | Meaning |
|----------|------|
| `TRANS_MEM` | Data transfer task |
| `REDUCE` | Reduce task |
| `RECORD` | Notify record task |
| `WAIT` | Notify wait task |
| `CCU_GRAPH` | CCU graph task |
| `AIV_GRAPH` | AIV graph task |

The recommended viewing approach is as follows:

1. Follow the arrow direction to review the task dependency order.
2. Click the task node of interest.
3. View the Rank, Stream, Queue, and detailed node information in the right detail panel.
4. Examine the parent nodes and child nodes to trace upstream and downstream dependencies.

For Loop display, pay attention to the following information:

1. If the DAG contains a Loop, the page automatically overlays a Loop dashed box on the task graph.
2. The dashed box covers the Loop start point, end point, and internal loop body nodes for quick identification of loop boundaries.
3. Click the Loop capsule to navigate directly to the corresponding Loop Start node.
4. After selecting a node inside a Loop, the right detail panel displays the Loop information and the nesting chain for that node.

![Browse the task graph and view node details in DAGView](insight_image/V3_image/dagView_V3_2.gif)

---

## 9. Canvas Operations

The DAGView canvas supports the following operations:

| Operation | Description |
|------|------|
| Pan canvas | Drag an empty area |
| Zoom in / Zoom out | Use the mouse scroll wheel, or click `+` / `-` in the bottom-right corner |
| Reset zoom | Click the percentage button in the bottom-right corner |
| Select node | Click the target node |

When the graph is large, reduce the Rank selection range first, then zoom into specific areas to examine dependencies.

---

## 10. View Node Details

After clicking a DAG node, the right detail panel displays node information. The common sections are as follows:

| Section | Description |
|------|------|
| Node overview | Node ID, task type, Rank, Stream, Queue |
| Loop information | The Loop that contains the current node, Loop count, instruction range, Loop boundaries |
| Parent nodes | Upstream nodes that the current node depends on |
| Child nodes | Downstream nodes that depend on the current node |
| Node semantics | Notify, Task metadata, Memory Slices |
| Raw JSON | The complete raw information of the current node |

When troubleshooting dependency relationships, check `Parent nodes` and `Child nodes` first. When troubleshooting loop structures, focus on `Loop information`. When troubleshooting data transfer issues, focus on `Memory Slices`.

The `Memory Slices` section follows these display rules:

1. A regular memory Slice displays `rank / type / offset / size`.
2. If the Slice belongs to `MS_CCU`, Insight V3 displays the `MSID` instead of the underlying abstract offset.
3. For batch tasks, if multiple `MS_CCU` Slices exist, the system merges them into a single summary card automatically.
4. The summary card displays a description such as `Total 8 MSIDs used`.
5. After expanding `MSID Details`, inspect the `id` and `size` of each `MSID` individually.

---

## 11. Search for Nodes

The left `Search` panel locates DAG nodes quickly.

The supported search fields are as follows:

| Field | Use Case |
|------|----------|
| `taskId` | Use when the node ID is known |
| `taskType` | Search by task type, for example `TRANS_MEM` |
| `notifyId` | Search by notify id |

Steps:

1. Select a search field in the left search panel.
2. Enter the complete keyword.
3. Click the search result.
4. DAGView locates and selects the corresponding node automatically.

The current search uses exact matching. If the system finds no results, confirm that the keyword is complete and that the target Rank is selected.

---

## 12. Recommended Analysis Workflow

Follow this workflow for DAGView analysis:

1. Select a dataset on the `Overview` page.
2. Select a small number of Ranks in the left Rank tree.
3. Click `Enter Correlation View`.
4. View the overall nodes and dependency arrows in DAGView.
5. Click the node of interest and view the details on the right.
6. Trace upstream and downstream dependencies through parent nodes and child nodes.
7. Use the search function to locate known nodes quickly.

---

## 13. Frequently Asked Questions

### 13.1 The Page Shows No Data

Confirm that the Insight service address is correct, and confirm that the Checker has generated Insight data.

### 13.2 The DAG Is Empty After Entering the Correlation Page

Confirm that the dataset contains the following files:

```text
graph/graph.msgpack
graph/layout.msgpack
```

### 13.3 The DAG Graph Is Too Dense

Reduce the number of selected Ranks on the left side first. View a subset of Ranks, then zoom into specific areas for analysis.

### 13.4 The Search Returns No Node

Confirm that the search keyword is complete, and confirm that the Rank containing the target node is selected.

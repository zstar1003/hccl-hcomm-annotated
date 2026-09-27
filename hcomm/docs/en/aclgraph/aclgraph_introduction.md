# AclGraph Introduction

## Reading Path

```text
§1  What is ACL Graph
        |
        v
§2  Capture Lifecycle
        |
        v
§3  Task Update
        |
        v
§4  Known Failure Modes
        |
        v
§5  Advanced: Multi-Stream Topology
        |
        v
Appendix A  Key API Quick Reference
Appendix B  torch_npu Source Code Integration
```

> ### Simplified Capture Flow (Read this diagram first for the big picture)
>
>```mermaid
> sequenceDiagram
>     participant U as User
>     participant C as Runtime
>     participant G as Graph DAG
>     participant R as Device
>     U->>C: aclmdlRICaptureBegin
>     U->>G: Submit operators
>     G-->>C: Buffer (not executed)
>     U->>C: aclmdlRICaptureEnd
>     C->>G: Validate capture
>     Note over R: Replay phase (can repeat)
>     U->>G: aclmdlRIExecuteAsync
>     G->>R: Dispatch entire DAG at once
>     R-->>U: Sync complete
> ```
>
> **Compared with Eager mode**: In Eager mode, "1 capture" = "1 operator" = "1 dispatch overhead"; ACL Graph compresses the entire flow into **1 capture + N 1-syscall replays**.

## 1. What is ACL Graph

### 1.1 Problem: Host Dispatch Bottleneck in Eager Mode

PyTorch defaults to Eager mode: each operator is dispatched and executed immediately. Every operator travels from the Host-side Python API to the Host-side C++ dispatch, then to the Device-side kernel execution—before each kernel execution on the Device side, the Host-side dispatch logic must complete.

Therefore, when individual operators have small computation volume or Host performance is suboptimal, Device idle time easily arises: after each kernel finishes, it must wait for the next kernel dispatch to complete.

### 1.2 Existing Solution: CUDA Graph

To optimize Host scheduling performance, CUDA provides a graph mode solution called CUDA Graph—a Device-side scheduling strategy that **eliminates the Host dispatch process for operators**. The PyTorch official documentation provides detailed descriptions.

The core idea of CUDA Graph is to turn "one dispatch, one execution" into "one dispatch, multiple executions"—amortizing the repeated Host scheduling overhead over a single capture process.

### 1.3 ACL Graph: The NPU Equivalent

ACL Graph is a "**capture-replay**" mechanism provided by the CANN Runtime, equivalent to CUDA Graph on the NPU side: between two calls to `aclmdlRICaptureBegin` / `aclmdlRICaptureEnd`, all tasks submitted on the specified stream are not executed immediately but are buffered as a **model runtime instance** (`aclmdlRI`); subsequently, calling `aclmdlRIExecuteAsync` can replay the same batch of tasks **multiple times** as a whole.

This document is positioned from the ACL Graph API perspective—readers who want to enable it from the PyTorch model layer directly can use upper-layer frameworks such as torch_npu / torchair (not covered here).

### 1.4 Applicability and Limitations

**Applicable**:

- Small-operator-intensive inference (DDP/FSDP gradient bucketing, LLM decode phase, recommendation system inference, dynamic batch inference)
- Communication-intensive training (HCCL collective communication repeatedly executed on streams)
- Scenarios where the same set of tasks is repeatedly executed across multiple inference rounds

**Not applicable**:

- Scenarios where task topology changes every time
- CPU synchronization-intensive workloads
- Debugging/prototyping phase (capture is invisible, increasing debugging difficulty)

## 2. Capture Lifecycle

### 2.1 Getting Started: 4-Step API Flow

The core ACL Graph APIs (`aclmdlRICaptureBegin` / `aclmdlRICaptureEnd` / `aclmdlRIExecuteAsync` / `aclmdlRIDestroy`), capture modes (`ACL_MODEL_RI_CAPTURE_MODE_GLOBAL` / `RELAXED`), and extended APIs (task group / update group / mode switching) — their function signatures, parameter descriptions, and type definitions are in the cann/runtime repository header file [`include/external/acl/acl_rt.h`](https://gitcode.com/cann/runtime/blob/master/include/external/acl/acl_rt.h).

> **Complete example**: The cann/runtime repository [`example/2_advanced_features/model_ri/0_simple_model/main.cpp`](https://gitcode.com/cann/runtime/blob/master/example/2_advanced_features/model_ri/0_simple_model/main.cpp) demonstrates the complete 4-step lifecycle (CaptureBegin → submit operators → CaptureEnd → ExecuteAsync x N → Destroy), including GLOBAL/RELAXED mode switching, async memcpy into the graph, and `aclmdlRIDebugJsonPrint` debug output.

**The following sections discuss only constraints related to HCCL collective communication coordination; basic ACL Graph usage is not repeated.**

### 2.2 Preparation: Before Capture

Two things:

1. **Drain HCCL watchdog** — The `ProcessGroupHCCL` background watchdog thread calls `aclrtEventQuery` to check collective completion. If it calls query during capture, the "event query" would also be written into the capture context. The `NPUGraph::capture_begin` function ([NPUGraph.cpp L240-340](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUGraph.cpp#L240)) completes mempool binding and stream checking before calling `AclmdlRICaptureBegin`, ensuring the capture context is ready.
2. **Route allocator to private pool** — Tensors allocated during capture have their lifecycle bound to `model_ri_`, and are released together with the mempool upon destruction.

### 2.3 Constraints During Capture (HCCL Perspective)

Basic ACL Graph constraints (same-stream capture, cross-stream event/notify introduction, synchronization API restrictions during capture, etc.) are demonstrated in the cann/runtime example [`0_simple_model/main.cpp`](https://gitcode.com/cann/runtime/blob/master/example/2_advanced_features/model_ri/0_simple_model/main.cpp). The following only supplements constraints that require special attention in HCCL coordination scenarios.

#### Cross-Stream Capture Must "Join Then Return to Main Stream"

To include tasks from other streams in the capture, you must introduce them via event/notify, and **ultimately must return to the main stream via event/notify**. Otherwise, capture_end validation will report an error.

![Cross-Stream Capture Basic Flow (main stream -> Stream2/3 -> must return to main stream)](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/figure/en-us_image_0000002531349376.png)

> This flowchart illustrates: call `Begin` on the main stream, Record Event on Stream2/3 to join, Wait Event on the main stream to pull tasks back, and finally call `End` on the main stream. After returning to the main stream and before ending capture, **no more tasks may be submitted on Stream2/3**—otherwise an error will be triggered due to "unassociated tasks".

#### Synchronization Primitive Selection

When establishing dependencies between streams during capture, select primitives based on semantics:

| Scenario | Use notify | Use event |
| -------- | ---------- | ---------- |
| Cross-Device synchronization | Recommended | Also works, but notify is lighter by design |
| Same-Device multi-stream sync | Equivalent to event | Equivalent to notify |
| One-to-many notification needed | Not supported | Supported |
| Timestamp needed | Not supported | Supported |
| Post-Wait state | **Auto-reset** (one-shot) | **No auto-reset** (reusable) |

> Design implication: notify's "one-shot auto-reset" characteristic matches HCCL's internal stateful machine of "notify from stream -> wait from stream -> notify dependent"; event's "many-to-many reusable" characteristic suits business/communication semantic switching points. The basic APIs for Event and Notify creation, waiting, and querying are in the Ascend official documentation: [Event Management](https://www.hiascend.com/en/document/detail/en/canncommercial/80RC1/devguide/aclrt/aclrt_0005.html) and [Notify Management](https://www.hiascend.com/en/document/detail/en/canncommercial/80RC1/devguide/aclrt/aclrt_0006.html).

### 2.4 After Capture: Replay and Destruction

`capture_end` internally completes `aclmdlRIBuildModel` (instantiating the DAG into an executable graph object). After that, calling `aclmdlRIExecuteAsync(model_ri, stream)` on any stream replays the entire graph—the only real CPU-side cost is a single syscall, independent of operator count.

**Destruction order constraint** (when working with `ProcessGroupHCCL`):

1. `NPUGraph::reset()` releases the ACL mempool
2. `destroy_process_group()` goes through HCCL watchdog join
3. `HcclCommDestroy` destroys the communication domain

Reversing the order will cause use-after-free because HCCL-side work metadata points to already-freed buffers.

## 3. Task Update

After tasks have been captured and buffered into `aclmdlRI`, if you need to update the tasks themselves or parameter information, CANN 9.0.0 provides two methods. The cann/runtime repository provides complete examples; the following only gives selection guidance.

### 3.1 In-Place Update

Applicable scenario: **a small number of tasks need updating** (e.g., varlen attention's seq_len changes every forward).

> **Complete example**: cann/runtime repository [`example/2_advanced_features/model_ri/1_model_update/main.cpp`](https://gitcode.com/cann/runtime/blob/master/example/2_advanced_features/model_ri/1_model_update/main.cpp).

![Recapture Flow (large number of tasks)](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/figure/en-us_image_0000002562269291.png)

### 3.2 Recapture

Applicable scenario: **a large number of tasks need updating** (e.g., a model has multiple different input Shapes).

> **Complete example**: cann/runtime repository [`example/2_advanced_features/model_ri/2_model_switch/`](https://gitcode.com/cann/runtime/tree/master/example/2_advanced_features/model_ri/2_model_switch) (Stream binding/jumping/switching), [`3_cond_model/`](https://gitcode.com/cann/runtime/tree/master/example/2_advanced_features/model_ri/3_cond_model) (IF/WHILE/SWITCH conditional operations).

![In-Place Update Flow (update then execute)](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/figure/en-us_image_0000002562429279.png)

### 3.3 Selection Criteria

| Dimension | In-Place Update | Recapture |
| --------- | --------- | ---------- |
| Task count | Small | Large |
| API complexity | More complex (requires task group/update group) | Simple (re-call Begin/End) |
| Hardware resources | Reuse same `aclmdlRI` | Maintain one `aclmdlRI` per Shape |
| Main risk | Task group boundary timing errors | `aclmdlRI` count exceeding hardware resource limits |
| Typical scenario | varlen attention, dynamic batch | Multi-Shape model training |

The concurrent execution scenario is a **special case** of in-place update: when you need to "update tasks first, then execute the captured model", introduce an Update stream + Event Wait ordering guarantee.

![Concurrent Execution Task Update Flow](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/acldevg/figure/en-us_image_0000002562429281.png)

## 4. Known Failure Modes

Categorized by "user-perceived impact", each category gives trigger conditions and mitigations.

### 4.1 Startup Phase Failures

| Failure Mode | Trigger Condition | Mitigation |
| --------- | --------- | ------ |
| `TASK_QUEUE_ENABLE=2` error | Environment variable incompatible with ACL Graph | Set to `0` or `1` |
| pin_memory_expandable_segments error | `NPUExpandableHostAllocator` conflicts with graph private pool | Revert to `NPUCachingAllocator` |
| Old CANN version doesn't recognize communication kernel dispatch stream | Communication kernel dispatch stream not included in `model_ri_` | Upgrade CANN to 8.5+ |

### 4.2 Capture Phase Failures

| Failure Mode | Trigger Condition | Mitigation |
| --------- | --------- | ------ |
| `aclrtEventQuery` pollutes DAG | watchdog / user code calls query during capture | `NPUGraph::capture_begin` entry check + mempool pre-binding drain |
| `aclrtMemcpy` rejected in GLOBAL mode | Business has synchronous memory functions | Call `aclmdlRICaptureThreadExchangeMode` to downgrade to RELAXED |
| Cross-stream tasks don't return to main stream | Cross-stream capture not ended on main stream | Strictly follow §2.3 capture topology design |
| Default stream operations | `with torch.npu.stream(s_default)` nested during capture | Explicitly switch to non-default stream |

### 4.3 Release Phase Failures

| Failure Mode | Trigger Condition | Mitigation |
| --------- | --------- | ------ |
| `destroy_process_group` hangs | Destruction order reversed (HCCL communication domain before `aclmdlRI`) | Strictly follow §2.4 three-step order |
| UB overflow | Graph retains all intermediate tensors at once, exceeding NPU UB size | Monitor `peak_mempool_bytes`; use graph partition + fallback if needed (see issue #102) |
| Memory limit exceeded | Too many `aclmdlRI` instances (recapture method 2) | Monitor model count; switch to in-place update |

## 5. Advanced: Multi-Stream Topology

![Multi-Stream Topology Overview](./figures/multi_streams_in_aclgraph.png)

### 5.1 HCCL Internal Multi-Stream Splitting

Runtime is not aware of communication semantics—it only receives tasks on streams. HCCL, as a "communication layer" above ACL, splits a single collective into multiple internal streams following the "control plane/data plane separation" and "parallel pipeline" hardware design principles.

> ACLGraph capture principles and cross-stream capture usage are in the Ascend official documentation: [ACL-Graph Development Guide](https://www.hiascend.com/en/document/detail/en/canncommercial/80RC1/devguide/aclgraph/aclgraph_0000.html) and [Cross-Stream Capture](https://www.hiascend.com/en/document/detail/en/canncommercial/80RC1/devguide/aclgraph/aclgraph_0002.html).

| Stream | Role | Device | Design Purpose |
| ------ | ------ | ------ | --------- |
| **Compute stream** `@pytorch` | Business matmul / bias / relu | NPU device | Parallel with communication streams, hiding communication latency |
| **Communication main stream** `@pytorch` | `dist.all_gather(...)` entry | NPU device | Business/communication semantic switching point (user-visible stream) |
| **Communication kernel dispatch stream** `@HCCL_HOST` | HCCL AICPU kernel | **AICPU coprocessor** | Offloads "communication descriptors" from host to coprocessor, non-blocking host CPU |
| **Communication execution stream** `@HCCL_DEVICE` (main) | Communication task (Ring/Tree/HalvingDoubling kernel) | NPU device | Actual data-moving device kernel |
| **Communication secondary stream** `@HCCL_DEVICE` | Another segment of communication task for master-slave parallelism | NPU device | "Master-slave" segmented parallelism for large collectives |

### 5.2 `model_ri_` Actually Contains Only 3 Layers

The "communication tasks actually moving data" on the communication execution stream / secondary stream are the **data plane**—if the data plane were also solidified into `model_ri_`, the graph object would be too large and would lose the flexibility of "re-selecting algorithms on each replay".

The design adopted by ACL Graph under AICPU expansion:

> `model_ri_` actually contains only the compute stream, communication main stream, and communication kernel dispatch stream (control plane). Communication tasks on the communication execution stream / secondary stream are **expanded in real-time** during AICPU kernel execution—during replay, Runtime calls `AclmdlRIExecuteAsync(model_ri, stream)` to trigger the AICPU kernel on the communication kernel dispatch stream, and AICPU then expands the communication tasks on the communication execution stream / secondary stream in real-time.

Benefits:

- Graph object size greatly reduced (only control flow is captured)
- Data plane algorithms can be re-decided on each replay (Ring/Bruck switching, topology change adaptation)
- Cross-NPU deployment: optimal algorithm selected based on actual current topology during replay

### 5.3 Runtime Behavior Annotations

Runtime and HCCL have 4 coordination points during capture:

| # | Behavior | Implementation |
| --- | ------ | ------ |
| 1 | Communication main stream joins `model_ri_` | Runtime automatically includes based on event relationships |
| 2 | Communication kernel dispatch stream joins `model_ri_` | HCCL actively pulls in (via `currentStreamCaptureStatusMayInitCtx` to detect capture status) |
| 3 | Communication execution stream / secondary stream not joined | AICPU expands in real-time during replay |
| 4 | `capture_end` removes all streams | One-time unload from `model_ri_`, no residue |

> **ACL Graph source implementation**: The cann/runtime repository [`src/runtime/feature/aclgraph/`](https://gitcode.com/cann/runtime/tree/master/src/runtime/feature/aclgraph) contains core implementations including capture/stream_capture/model/event_capture.

## Appendix A: Key API Quick Reference

All ACL Graph API function signatures, parameter descriptions, and type definitions are in the cann/runtime repository header file [`include/external/acl/acl_rt.h`](https://gitcode.com/cann/runtime/blob/master/include/external/acl/acl_rt.h).

Complete examples are in the cann/runtime repository [`example/2_advanced_features/model_ri/`](https://gitcode.com/cann/runtime/tree/master/example/2_advanced_features/model_ri) directory (4 examples: basic capture, task update, stream switching, conditional operations).

## Appendix B: torch_npu Source Code Integration

> This appendix is for internal developers, introducing how torch_npu wraps ACL Graph into PyTorch high-level APIs. The following source code is from the [Ascend/pytorch](https://github.com/Ascend/pytorch) public repository.

### B.1 C++ Core: `c10_npu::NPUGraph`

Source file: [`torch_npu/csrc/core/npu/NPUGraph.cpp`](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUGraph.cpp). The `capture_begin` call chain and `replay` implementation are in source lines L240-340 (mempool binding -> `AclmdlRICaptureBegin` -> `AclmdlRICaptureGetInfo`).

### B.2 Python Entry: `torch.npu.graph` / `torch.npu.NPUGraph`

Source files: [`torch_npu/csrc/npu/Graph.cpp`](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/npu/Graph.cpp) (PyBind11 binding) and [`torch_npu/csrc/npu/Graph.h`](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/npu/Graph.h) (declaration).

```python
g = torch.npu.NPUGraph()                              # Create
with torch.npu.graph(g):                              # Enter capture
    out = model(static_input)                          # Business code
g.replay()                                             # Execute multiple times
```

### B.3 OpHandler Framework: Operator-Level Preprocessing

Source directory: [`torch_npu/npu/_npugraph_handlers/`](https://github.com/Ascend/pytorch/tree/master/torch_npu/npu/_npugraph_handlers).

### B.4 Collective Communication Integration: `ProcessGroupHCCL`

Source file: [`torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp). Key coordination points:

- Header `#include "torch_npu/csrc/core/npu/NPUGraph.h"` ([L43](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp#L43), compile-time binding)
- `kWatchdogThreadSleepMillis = 1000` ([L441](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp#L441), HCCL-side watchdog polling interval)
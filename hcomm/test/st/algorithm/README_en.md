# Algorithm Analyzer Tool User Guide

## Introduction

The HCCL algorithm analyzer is used to simulate the execution of HCCL algorithms in an offline environment, verifying algorithm logic and memory operations. The HCCL algorithm analyzer provides efficient and fast batch execution of test tasks to meet developer needs.

## Principles

![](./figures/principle_en.png)

**Key points:**

1. The algorithm analyzer stubs the platform and framework layers to obtain the task sequence for all ranks during algorithm execution.
2. The task information of all ranks is organized into a **Directed Acyclic Graph (DAG)**.
3. Validations are performed based on **graph algorithms**, such as memory read/write conflict detection and semantic validation. 1) Memory conflict detection analyzes whether there are potential read/write conflicts based on synchronization within the graph. 2) Semantic validation simulates the execution of the task graph, records **data movement information**, and checks whether the **data movement information** in the UserOutput memory meets the operator requirements after simulation.

## Environment Setup

Refer to the environment setup and source code download in [Source Build](../../../docs/en/build/build.md) to prepare the prerequisites for compiling the algorithm analyzer.

## Test Case Writing

### LLT Test Case Overview

An algorithm checker test case consists of 5 steps, as shown in the 5 boxes in the figure below. The following sections describe how to write each step to accommodate different operator requirements, as well as how to use the checker tool for issue location when problems arise.

![](./figures/compile_testcase1_en.png)

### LLT Test Case Step Details

#### Topology Generation

- TopoMeta structure introduction

  ![](./figures/compile_testcase2.png)

  checker: TopoMeta is used to represent a topology structure. TopoMeta is a three-layer vector structure.

  PhyDeviceId: Represents the physical ID of an NPU.

  ServerMeta: Composed of PhyDeviceId, representing the number of cards on a server and their corresponding PhyDeviceIds.

  SuperPodMeta: Composed of ServerMeta, representing the server composition of a super node.

  TopoMeta: Represents the overall cluster topology.

- TopoMeta generation methods

  There are two ways to generate TopoMeta:

  1. Specify the number of super nodes, the number of servers, and the number of cards per server, then use the `GenTopoMeta` function provided by the `RankTable_For_LLT` class to generate it. Suitable for symmetric topology scenarios.

     ![](./figures/compile_testcase3_en.png)

  2. Fully customize super nodes, servers, and card counts for asymmetric topology scenarios. As shown below, the TopoMeta has one super node with two servers inside — one server has 2 cards and the other has 3 cards.

     ![](./figures/RE_1_1.png)

- Rank table generation

  Once TopoMeta is available, use the `GenRankTable` function provided by the `RankTable_For_LLT` class to generate the rank table.

#### Environment Variable Configuration

- Setting environment variables

  Environment variables affect the logic flow in the code. Use the `setenv` function to configure the required conditions before test case execution.

- Cleaning environment variables

  Since environment variables are process-level and an LLT task runs in the same process, environment variable usage may affect other test cases. To clean up environment variables, the `TearDown` function in the test suite currently calls the environment variable cleanup function.

  ![](./figures/RE_2_2_en.png)

  The current cleanup function handles the following environment variables. If new environment variables are added in the future, they need to be added to this function.

  ![](./figures/compile_testcase6.png)

#### Log Level Configuration

- Checker log level (default: ERROR)

   1. Call the `setCheckerLogWarn()` interface to set the log level

         Set the checker log level to WARNING:

         ![](./figures/RE_3_1.png)

   2. Set the log level via environment variable

         Enable WARNING level logging:

         ```bash
         export CHECK_LOG_LEVEL=2
         ```

         Enable ERROR level logging:

         ```bash
         export CHECK_LOG_LEVEL=3
         ```

- HCCL log level (default: ERROR)

   The current environment variable method is deprecated. Use the following approach to set the log level and print logs.

   ![](./figures/compile_testcase8.png)

   Enable DEBUG level logging:

   ```bash
   export ASCEND_LOG_LEVEL=0
   ```

   Enable INFO level logging:

   ```bash
   export ASCEND_LOG_LEVEL=1
   ```

   Enable WARNING level logging:

   ```bash
   export ASCEND_LOG_LEVEL=2
   ```

   Enable ERROR level logging:

   ```bash
   export ASCEND_LOG_LEVEL=3
   ```

#### Operator Parameter Configuration

`TestOpParam` is used to configure test parameters. The table below describes the main parameters used.

| Parameter   | Required or Optional | Description                                                   | Remarks                                         |
| ----------- | -------------------- | ------------------------------------------------------------- | ----------------------------------------------- |
| opType      | Required             | Specifies the operator type to be tested                      | batchsendrecv is currently under development     |
| tag         | Required             | The tag of the operator execution entry                       | Can be set arbitrarily                          |
| algName     | Optional             | Specifies the algorithm name to execute                       | If specified, skips algorithm selection; otherwise auto-selects |
| opMode      | Required             | Specifies single operator mode or graph mode                  |                                                 |
| reduceType  | Optional             | Required when a reduce type is involved                       |                                                 |
| devtype     | Required             | The hardware type for execution                               | Supports 310P3 V / 310P3 Duo / 910A / 910B / 910C |
| is310P3V    | Optional             | Must be set to true when running on 310P3 V hardware           |                                                 |
| count       | Required             | Number of data elements                                       |                                                 |
| dataType    | Required             | Data type                                                     |                                                 |

#### Running the Checker

Pass the `TestOpParam`, `rankTable`, `TopoMeta`, and other parameters generated in the previous steps to the `Check` function of the `Checker` object for execution.

#### Verifying Check Results

Check that the return value of `Check` is `HcclResult::HCCL_SUCCESS`.

### LLT Test Case Filtering

When there are many test cases and you only need to execute a specific one, modify the test case name in `main.cc`.

![](./figures/RE_4_en.png)

## Test Execution

Run the following commands from the source code root directory to compile and execute algorithm analyzer test cases:

```bash
# Compile all test suite cases and execute automatically
bash build.sh --st

# Compile individual test suite cases and execute automatically
bash build.sh --open_hccl_test
bash build.sh --executor_hccl_test
bash build.sh --executor_reduce_hccl_test
bash build.sh --executor_pipeline_hccl_test

# Manually execute test cases
./build/test/st/algorithm/testcase/testcase/open_hccl_test
./build/test/st/algorithm/testcase/testcase/executor_hccl_test
./build/test/st/algorithm/testcase/testcase/executor_reduce_hccl_test
./build/test/st/algorithm/testcase/testcase/executor_pipeline_hccl_test
```

## Result Examples

### Result Analysis

The test case execution results are shown below:

![](./figures/result1.png)

The meaning of each field is as follows:

`[run]`: Indicates the test case being executed

`[OK]`: Indicates successful execution and verification passed

`[FAIL]`: Indicates execution failure. Analyze the specific cause based on the printed logs.

## Issue Location

### Memory Conflict Detection

#### Symptoms

When a region of memory between two synchronization signals is concurrently written by multiple tasks, or is read while being written, a memory conflict occurs. In real runtime environments, this typically manifests as random accuracy issues.

Under the current Mesh structure, false positives may occur when Reduce operators are present. This is because, in a Mesh structure, a memory block may be simultaneously written by other cards within one synchronization interval. In this case, the hardware can ensure the atomicity of the Reduce operation and no accuracy issues occur in actual execution. However, from the checker's perspective, multiple reads and writes to the same memory occur between two synchronizations, so it is flagged as an error.

Except for the scenario above, the following error output indicates a risk of memory conflicts in the task scheduling:

```text
[1]there is memory use conflict in two SliceMemoryStatus
[2]one is startAddr is 0, size  is 3200, status is WRITE.
[3]another is startAddr is 0, size  is 3200, status is WRITE.
[4]failed to check memory BufferType::OUTPUT_CCL
[5]memory conflict between node [rankId:1, queueId:0, index:1] and node [rankId:2, queueId:0, index:1]
[6]check rank memory conflict failed for rank 0
```

- Lines 2 and 3 indicate the start address, size, and read/write status of the two conflicting memory blocks.

    The status can be READ or WRITE. READ means the memory block is being read; WRITE means it is being written. Being read and being written are abstract memory operation semantics, not limited to write tasks and read tasks.

    Memory blocks that may have READ status include the src of a localcopy task, the src of a read task, and the src of a write task. Memory blocks that may have WRITE status include the dst of a localcopy task, the dst of a read task, and the dst of a write task.

- Line 4 indicates the type of the conflicting memory block.
- Line 5 indicates which two tasks caused the memory conflict.
- Line 6 indicates the rank ID where the memory conflict occurred.

The above error log indicates that two tasks are simultaneously performing write operations in the range 0 to 3200 of the `OUTPUT_CCL` type.

#### Debugging Method

1. Enable task printing before calling the `Check` function.

    ```text
    checker.EnableTaskPrint();
    ```

2. Based on the error log, locate the two tasks that caused the memory conflict and check the synchronization arrangement before and after these two tasks.

    The error log in the [Symptoms](#symptoms) section indicates that two tasks are simultaneously performing write operations in the range 0 to 3200 of the `OUTPUT_CCL` type.

### Semantic Validation Failure

#### Basic Concepts

The algorithm analyzer uses relative addresses to represent memory, consisting of three fields: memory type, offset address, and size, represented by the `DataSlice` structure:

```c
class DataSlice {
public:
    //  Some method functions

private:
    BufferType type;
    u64        offset;
    u64        size;
}
```

The supported memory types include Input, Output, CCL_Input, CCL_Output, Scratch, etc.

During collective communication algorithm execution, complex data movement and reduction operations are involved. The algorithm analyzer uses **BufferSemantic** to record **data movement relationships**, which includes a destination memory expression and multiple source memory expressions. The destination memory is represented by the member variables `startAddr` and `Size`. The source memory is represented by the `SrcBufDes` structure, which is defined as follows:

```c
struct BufferSemantic {
    u64                         startAddr;
    mutable u64                 size;       // Size, shared between source and destination memory
    mutable bool                isReduce;   // Whether a reduce operation was performed; when multiple srcBufs exist, it must be a reduce scenario
    mutable HcclReduce0p        reduceType; // Type of reduce operation
    mutable std::set<SrcBufDes> srcBufs;    // The rank or ranks from which this data originates
};

struct SrcBufDes {
    RankId      rankId;   // Rank ID of the data source
    BufferType  bufType;  // Memory type of the data source
    mutable u64 srcAddr;  // Offset address relative to the source memory type
};
```

#### Semantic Calculation Example

The following example illustrates what semantic calculation is.

1. Initial state: There are two ranks, Rank0 and Rank1, with two memory types: Input and Output.

    ![](./figures/allgather.png)

2. Action of state 1: Copy the data block from rank0's Input at offset address 20, size 30, to rank0's Output at offset address 35. Result: A semantic block is generated on rank0's Output, recording the movement information.

    ![](./figures/allgather-0.png)

3. Action of state 2: Copy the data block from rank1's Input at offset address 70, size 15, to rank0's Output at offset address 50. Result: The destination memory overlaps with an existing semantic block, so the existing semantic block needs to be split, producing two semantic blocks.

    ![](./figures/allgather-1.png)

#### Result Validation

During semantic analysis execution, many semantic blocks are generated (recording many data movement relationships). After execution, the semantic blocks in the Output memory are checked against expectations.

The following uses a 2-rank AllGather example to illustrate the normal and abnormal scenarios for the semantic blocks in Rank0's Output memory. Assume the input data size is 100 bytes.

- **Correct scenario:**

    ![](./figures/allgather-2.png)

- **Incorrect scenario:**

    ![](./figures/allgather-3.png)

#### Debugging Approach

The semantic validation phase can detect two types of errors:

- Missing data.
- Incorrect data source.

For reduction scenarios, similar issues can occur, such as missing participating ranks or different data offset addresses among participating ranks. Typically, the semantic check provides hints when reporting errors. Analysis should be performed using the hints along with the task sequence printed by the algorithm analyzer.

# EI0012 Execution_Error_SDMA

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: remote rank information, basic information, task information, communicator information.

```text
SDMA memory copy task exception occurred. Remote rank: %s. Base information: %s. Task information: %s. Communicator information: %s.
```

Error example:

```text
SDMA memory copy task exception occurred. Remote rank: [4800]. Base information: [streamID:[44], taskID[2], taskType[Memcpy], tag[AllReduce_group_name_0], AlgType(level 0-1-2):[ring-ring-NHR].]. Task information: [src:[0x320000], dst:[0x12c088120000], size:[0x320000], notify id:[0xffffffffffffffff], link type:[OnChip], remote rank:[local]]. Communicator information: [group:[group_name_0], user define information[Unspecified], rankSize[16], rankId[13]].
```

## Possible Cause

1. Network connection exception occurred during the SDMA task execution.

2. The peer process exits abnormally.

3. The input or output memory address is not allocated, the actual allocated size is smaller than the input data size, or the memory is freed before the operator execution is complete.

## Solution

1. Check whether the network link is abnormal during the execution.

2. Check whether a process in the cluster exits before an error is reported. If yes, locate the cause of the process exit.

3. Check whether the input/output memory size is correct and whether the memory or communicator is released prematurely.

# EI0011 Resource_Error_Insufficient_Device_Memory

## Symptom

The following is error format. The placeholder %s indicates the memory size.

```text
Failed to allocate %s bytes of NPU memory.
```

Error example:

```text
Failed to allocate 262144~3145728 bytes of NPU memory.
```

## Possible Cause

Allocation failure due to insufficient NPU memory.

## Solution

Stop unnecessary processes and ensure the required memory is available.

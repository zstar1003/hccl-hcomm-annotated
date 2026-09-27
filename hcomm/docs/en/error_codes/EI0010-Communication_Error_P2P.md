# EI0010 Communication_Error_P2P

## Symptom

The following is error format. The placeholder %s indicates the error cause.

```text
P2P communication failed. Reason: %s
```

Error example:

```text
P2P communication failed. Reason: Device ID 0 in module 0 and device ID 9 in module 1 are not on the same plane.
```

## Solution

Ensure that the NPU card is normal and enter environment variable `export HCCL_INTRA_ROCE_ENABLE=1`.

# EI0007 Resource_Error

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: memory location, memory size.

```text
Failed to allocate resource %s with info %s. Reason: Resources are exhausted.
```

Error example:

```text
Failed to allocate resource HostMemory with info size:8928 bytes. Reason: Resources are exhausted.
```

## Solution

Insufficient resources. Please adjust the code as prompted in the error message.

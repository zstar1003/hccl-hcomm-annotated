# EI0020 Communication_Error_Bind_IP_Port

## Symptom

The following is error format. The placeholder %s indicates the error cause.

```text
Failed to enable listening for the NPU network adapter socket. Reason: %s
```

Error example:

```text
Failed to enable listening for the NPU network adapter socket. Reason: The IP address 192.1.3.198 add port 16666 have already been bound.
```

## Solution

Check whether the single-card multi-process scenario is used. If yes, configure the port number using the environment variable HCCL_NPU_SOCKET_PORT_RANGE.

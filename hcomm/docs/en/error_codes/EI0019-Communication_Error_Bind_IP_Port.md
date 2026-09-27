# EI0019 Communication_Error_Bind_IP_Port

## Symptom

The following is error format. The placeholder %s indicates the error cause.

```text
Failed to enable listening for the host network adapter socket. Reason: %s
```

Error example:

```text
Failed to enable listening for the host network adapter socket. Reason: The IP address 10.23.146.197 add port 50001 have already been bound.
```

## Solution

1. Check whether this port has been occupied by another process. If yes, you can make adjustment using the environment variable HCCL_IF_BASE_PORT and use `sysctl -w net.ipv4.ip_local_reserved_ports=****-****` to adjust the scope of reserved ports.

2. Check whether the service process is started multiple times on a device during this service.

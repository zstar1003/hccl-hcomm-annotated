# EI0015 Communication_Error_Ranktable_Detect

## Symptom

The following is error format. The placeholder %s indicates the error cause.

```text
Failed to collect cluster information of the communicator based on rootInfo detection. Reason: %s.
```

Error example:

```text
Failed to collect cluster information of the communicator based on rootInfo detection. Reason: No rank in the communicator can connect to the root node within the timeout period. List of unconnected ranks: "[1,]".
```

## Solution

1. Check whether all ranks in the communicator have delivered the communicator creation interface.

2. Check the connectivity between the host networks of all nodes and the server node.

3. Check whether the HCCL_SOCKET_IFNAME environment variable of all nodes is correctly configured.

4. Increase the timeout by configuring the HCCL_CONNECT_TIMEOUT environment variable.

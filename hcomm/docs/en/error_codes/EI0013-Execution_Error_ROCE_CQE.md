# EI0013 Execution_Error_ROCE_CQE

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: local server ID, local device ID, local device IP, peer server ID, peer device ID, peer device IP.

```text
An error CQE occurred during operator execution. Local information: server %s, device ID %s, device IP %s. Peer information: server %s, device ID %s, device IP %s.
```

Error example:

```text
An error CQE occurred during operator execution. Local information: server 10.78.106.107, device ID 0, device IP 192.168.200.100. Peer information: server 10.78.106.111, device ID 0, device IP 192.168.200.101.
```

## Possible Cause

1. The network between two devices is abnormal. For example, the network port is intermittently disconnected.

2. The peer process exits abnormally in advance. As a result, the local end cannot receive the response from the peer end.

## Solution

1. Check whether the network devices between the two ends are abnormal.

2. Check whether the peer process exits first. If yes, check the cause of the process exit.

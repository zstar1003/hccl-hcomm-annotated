# EI0009 Communication_Error_Initialize_Transport

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: Device ID, error cause.

```text
Device %s transport init error. Reason: %s.
```

Error example:

```text
Device 0 transport init error. Reason: The network port is down.
```

## Solution

Use the following hccn_tool commands to check whether the port link is down. \(The scope of i represents the number of NPUs of each node. 8 is used as an example.\)

1. Check whether the optical module is in position:

   ```bash
   for i in {0..7}; do hccn_tool -i $i -optical -g; done | grep present
   ```

2. Check whether the IP address is configured:

   ```bash
   for i in {0..7}; do hccn_tool -i $i -ip -g; done
   ```

3. Check whether the switch is connected:

   ```bash
   for i in {0..7}; do hccn_tool -i $i -lldp -g; done
   ```

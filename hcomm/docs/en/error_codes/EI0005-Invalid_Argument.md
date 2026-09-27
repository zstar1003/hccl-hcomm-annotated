# EI0005 Invalid_Argument

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: operator name, parameter name, local parameter value, remote parameter value.

```text
The parameters of operator %s are inconsistent between ranks, parameter %s is %s on the local rank and %s on the remote rank.
```

Error example:

```text
The parameters of operator HcomAllReduce are inconsistent between ranks, parameter count is 2176 on the local rank and 4224 on the remote rank.
```

## Solution

Please modify the parameter value as prompted in the error message.

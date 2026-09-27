# EI0003 Invalid_Argument_Collective_Communication_Operator

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: operator name, parameter value, parameter name, expected value.

```text
Failed to verify parameters of operator %s. Value %s for parameter %s is invalid. The expected value is %s.
```

Error example:

```text
Failed to verify parameters of operator CheckRankID. Value 1sfa for parameter rank_id is invalid. The expected value is a non-negative decimal number.
```

## Solution

Try again with a valid argument.

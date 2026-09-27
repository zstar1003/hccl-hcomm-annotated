# EI0001 Config_Error_Invalid_Environment_Variable

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: environment variable value, environment variable name, expected value.

```text
Value %s for environment variable %s is invalid. Expected value: %s.
```

Error example:

```text
Value 2147483648 for environment variable HCCL_EXEC_TIMEOUT is invalid. Expected value: a number greater than or equal to 0s and less than or equal to 2147483647s.
```

## Solution

The environment variable configuration is invalid. Please reconfigure the environment variable as prompted in the error message.

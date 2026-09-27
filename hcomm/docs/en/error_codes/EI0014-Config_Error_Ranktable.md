# EI0014 Config_Error_Ranktable

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: configuration value, configuration item name, expected value.

```text
Value %s for ranktable variable %s is invalid. Expected value: %s.
```

Error example:

```text
Value [192.168.200.1000] for ranktable variable [IP] is invalid. Expected value: is a valid IP address.
```

## Possible Cause

Failed to verify the content of the ranktable file, possibly due to inconsistency between the file content and the actual device information.

## Solution

Try again with a valid cluster configuration in the ranktable file. Ensure that the configuration matches the operating environment.

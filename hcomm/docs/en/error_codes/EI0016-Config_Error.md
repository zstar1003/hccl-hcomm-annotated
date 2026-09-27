# EI0016 Config_Error

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: configuration value, configuration item name, expected value.

```text
Value %s for config %s is invalid. Expected value: %s.
```

Error example:

```text
Value Disable for config "tls" is invalid. Expected value: "All ranks are consistent. Current status: rankList for enabled tls:[80.48.25.34/0]; rankList for disabled tls:[80.48.25.34/1,2,3,4,5,6,7]; rankList for query failure tls:N/A".
```

## Solution

Check and adjust the configuration item value as prompted in the error message. For details, see the documentation on the official website.

# EI0004 File_Operation_Error_Parse

## Symptom

The following is error format. The meanings of the placeholders %s in sequence are: file name, error cause.

```text
Failed to parse the ranktable file %s. Reason: %s
```

Error example:

```text
Failed to parse the ranktable file /home/HCCL_TEST/sess_run/rankta/intra_rank_table_2rank_v1.json. Reason: The rankTable file path does not exist, the permission is insufficient, or the JSON format is incorrect.
```

## Solution

Please provide the correct ranktable file as prompted in the Reason.

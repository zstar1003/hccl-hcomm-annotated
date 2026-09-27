# EI0004 File_Operation_Error_Parse

## 错误信息

报错格式如下，占位符%s的含义依次为文件名、报错原因：

```text
Failed to parse the ranktable file %s. Reason: %s
```

报错示例如下：

```text
Failed to parse the ranktable file /home/HCCL_TEST/sess_run/rankta/intra_rank_table_2rank_v1.json. Reason: The rankTable file path does not exist, the permission is insufficient, or the JSON format is incorrect.
```

## 解决方法

需按照Reason中的提示提供正确的rank table文件。

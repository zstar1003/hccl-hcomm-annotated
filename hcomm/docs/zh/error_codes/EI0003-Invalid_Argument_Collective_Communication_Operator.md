# EI0003 Invalid_Argument_Collective_Communication_Operator

## 错误信息

报错格式如下，占位符%s的含义依次为算子名、参数值、参数名、期望值：

```text
Failed to verify parameters of operator %s. Value %s for parameter %s is invalid. The expected value is %s.
```

报错示例如下：

```text
Failed to verify parameters of operator CheckRankID. Value 1sfa for parameter rank_id is invalid. The expected value is a non-negative decimal number.
```

## 解决方法

请使用有效的参数重试。

# EI0001 Config_Error_Invalid_Environment_Variable

## 错误信息

报错格式如下，占位符%s的含义依次为环境变量值、环境变量名、报错原因：

```text
Value %s for environment variable %s is invalid. Expected value: %s.
```

报错示例如下：

```text
Value 2147483648 for environment variable HCCL_EXEC_TIMEOUT is invalid. Expected value: a number greater than or equal to 0s and less than or equal to 2147483647s.
```

## 解决方法

环境变量配置无效，请根据错误提示重新配置环境变量。

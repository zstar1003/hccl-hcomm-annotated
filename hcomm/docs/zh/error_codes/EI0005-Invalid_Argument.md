# EI0005 Invalid_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为算子名、参数名、本端参数值、对端参数值：

```text
The parameters of operator %s are inconsistent between ranks, parameter %s is %s on the local rank and %s on the remote rank.
```

报错示例如下：

```text
The parameters of operator HcomAllReduce are inconsistent between ranks, parameter count is 2176 on the local rank and 4224 on the remote rank.
```

## 解决方法

请根据报错提示调整参数值。

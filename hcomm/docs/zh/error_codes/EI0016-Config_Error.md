# EI0016 Config_Error

## 错误信息

报错格式如下，占位符%s的含义依次为配置项值、配置项、期望值：

```text
Value %s for config %s is invalid. Expected value: %s.
```

报错示例如下：

```text
Value Disable for config "tls" is invalid. Expected value: "All ranks are consistent. Current status: rankList for enabled tls:[80.48.25.34/0]; rankList for disabled tls:[80.48.25.34/1,2,3,4,5,6,7]; rankList for query failure tls:N/A".
```

## 解决方法

根据报错提示检查并调整配置项的值，详细描述请参考官方网站上的文档。

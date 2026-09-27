# EI0014 Config_Error_Ranktable

## 错误信息

报错格式如下，占位符%s的含义依次为取值、配置项、报错原因：

```text
Value %s for ranktable variable %s is invalid. Expected value: %s.
```

报错示例如下：

```text
Value [192.168.200.1000] for ranktable variable [IP] is invalid. Expected value: is a valid IP address.
```

## 可能原因

无法验证ranktable文件的内容，可能是由于文件内容与实际设备信息不一致所致。

## 解决方法

请尝试使用ranktable文件中有效的集群配置。确保该配置与运行环境相匹配。

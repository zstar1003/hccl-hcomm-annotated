# EI0008 Package_Error_Incorrect_HCCL_Version

## 错误信息

报错格式如下，占位符%s的含义依次为模块名、本端版本、对端版本：

```text
The %s versions are inconsistent. The local %s, while the remote %s.
```

报错示例1如下：

```text
The ops (cann-hccl) versions are inconsistent. The local version is 9.2.0 (version file: /usr/local/Ascend/cann-9.2.0/share/info/hccl/version.info), while the remote version is 9.1.0.
```

报错示例2如下：

```text
The Toolkit (cann-hcomm) and ops (cann-hccl) versions are inconsistent. The local Toolkit (cann-hcomm) version is 9.1.0 and ops (cann-hccl) version is 9.1.0 (version file: /usr/local/Ascend/cann-9.2.0/share/info/hccl/version.info), while the remote Toolkit (cann-hcomm) version is 9.2.0 and ops (cann-hccl) version is 9.2.0.
```

## 解决方法

请安装相同的版本，本端的toolkit (cann-hcomm)版本、ops (cann-hccl)版本必须和远端对应的版本一致。

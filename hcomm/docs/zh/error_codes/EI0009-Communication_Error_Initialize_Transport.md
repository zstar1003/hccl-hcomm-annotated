# EI0009 Communication_Error_Initialize_Transport

## 错误信息

报错格式如下，占位符%s的含义依次为Device ID、报错原因：

```text
Device %s transport init error. Reason: %s.
```

报错示例如下：

```text
Device 0 transport init error. Reason: The network port is down.
```

## 解决方法

通过hccn_tool的如下命令排查端口是否linkdown（i的范围代表每个节点的npu数量，以8个为例）：

1. 使用以下命令检查光模块是否在位：

    ```bash
    for i in {0..7}; do hccn_tool -i $i -optical -g; done | grep present
    ```

2. 使用以下命令检查是否已配置IP地址：

    ```bash
    for i in {0..7}; do hccn_tool -i $i -ip -g; done
    ```

3. 使用以下命令检查交换机是否已连接：

    ```bash
    for i in {0..7}; do hccn_tool -i $i -lldp -g; done
    ```

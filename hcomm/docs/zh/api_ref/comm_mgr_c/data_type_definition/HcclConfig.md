# HcclConfig

## 功能说明

定义集合通信相关配置。

## 定义原型

```c
typedef enum {
    HCCL_DETERMINISTIC = 0, /* 0: non-deterministic, 1: deterministic, 2: strict(order-preserving) */
    HCCL_CONFIG_RESERVED
} HcclConfig;
```

## 参数说明

- HCCL_DETERMINISTIC：是否开启确定性计算。

  - 0：不开启确定性计算。
  - 1：开启确定性计算。
  - 2：开启保序功能（仅Atlas A2训练系列产品/Atlas A2推理系列产品支持）。

- HCCL_CONFIG_RESERVED：预留参数。

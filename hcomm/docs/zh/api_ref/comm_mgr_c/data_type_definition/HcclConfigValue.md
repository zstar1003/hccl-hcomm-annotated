# HcclConfigValue

## 功能说明

定义[HcclConfig](HcclConfig.md)中可配置的参数的值。

## 定义原型

```c
union HcclConfigValue {
    int32_t value;
};
```

value为[HcclConfig](HcclConfig.md)中“HCCL_DETERMINISTIC”参数的值。

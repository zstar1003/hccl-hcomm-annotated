# HcommCcuInsResDescDestroy

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

销毁由`HcommCcuInsResDescCreate`创建的CCU资源描述符，释放其占用的内存。销毁后该句柄失效，不应再使用。

## 函数原型

```c
CcuResult HcommCcuInsResDescDestroy(HcommCcuResDescHandle resDesc)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| resDesc | 输入 | 资源描述符句柄，由`HcommCcuInsResDescCreate`创建。 |

## 返回值

[CcuResult](../../datatype_definition/CcuResult.md)：接口成功返回`CCU_SUCCESS`，其他值表示失败。

| 返回值 | 说明 |
| --- | --- |
| `CCU_SUCCESS` | 销毁成功。 |
| `CCU_E_NOT_FOUND` | `resDesc`未注册，或已被销毁。 |

## 约束说明

- 销毁后该句柄永久失效，不可再用于任何接口。重复销毁将返回`CCU_E_NOT_FOUND`。
- 本接口只能在主机侧调用。

## 依赖关系

- 依赖`HcommCcuInsResDescCreate`创建的资源描述符句柄。

## 调用示例

```c
HcommCcuResDescHandle resDesc = 0;
CcuResult ret = HcommCcuInsResDescCreate(0, &resDesc);
if (ret != CCU_SUCCESS) {
    return ret;
}

// 使用 resDesc ...

ret = HcommCcuInsResDescDestroy(resDesc);
return ret;
```

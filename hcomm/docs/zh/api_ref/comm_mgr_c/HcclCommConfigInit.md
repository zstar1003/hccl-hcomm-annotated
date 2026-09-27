# HcclCommConfigInit

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：支持
<!-- end id5 -->

## 功能说明

初始化通信域配置项。

## 函数原型

```c
static inline void HcclCommConfigInit(HcclCommConfig *config)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| config | 输出 | 需要初始化的通信域配置项。<br>HcclCommConfig类型的定义可参见[HcclCommConfig](./data_type_definition/HcclCommConfig.md)。 |

## 返回值

无

## 约束说明

无

## 调用示例

```c
uint32_t rankSize = 8;
uint32_t deviceId = 0;
// 生成root节点的rank标识信息
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);

// 创建并初始化通信域配置项
HcclCommConfig config;
HcclCommConfigInit(&config);
// 按需修改通信域配置
config.hcclBufferSize = 1024;  // 共享数据的缓存区大小，单位为：MB，取值需 >= 1，默认值为：200
config.hcclDeterministic = 1;  // 开启归约类通信算子的确定性计算，默认值为：0，表示关闭确定性计算功能
strncpy(config.hcclCommName, "comm_1", COMM_NAME_MAX_LENGTH - 1);
config.hcclCommName[COMM_NAME_MAX_LENGTH - 1] = '\0';
// 初始化集合通信域
HcclComm hcclComm;
HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, deviceId, &config, &hcclComm));

// 销毁通信域
HcclCommDestroy(hcclComm);
```

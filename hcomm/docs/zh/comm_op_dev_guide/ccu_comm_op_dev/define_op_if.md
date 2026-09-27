# 定义算子接口

开发者首先需依据通信算子的功能作用，创建算子接口定义头文件，后续调用此通信算子时依赖该头文件。

以自定义通信算子AllGather为例，该类算子接口需要传入源地址、目的地址、源数据量、数据类型，以及通信域和流信息，其接口定义如下：

```c
HcclResult HcclAllGatherCustom(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream);
```

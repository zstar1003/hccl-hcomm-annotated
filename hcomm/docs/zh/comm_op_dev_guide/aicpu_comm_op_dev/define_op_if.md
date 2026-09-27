# 定义算子接口

开发者首先需依据通信算子的功能作用，创建算子接口定义头文件，后续调用此通信算子时依赖该头文件。

以自定义点对点通信算子Send/Receive为例，Send算子是将本端rank指定位置的数据发送到对端，Receive算子是接收对端数据到本端指定位置上，二者需配对使用。因此，该类算子接口除了需要传入通信域信息、流信息外，还需要传入数据地址、数据量大小、数据类型、对端rank编号，其接口定义如下：

```c
HcclResult HcclSendCustom(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream);
HcclResult HcclRecvCustom(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream);
```

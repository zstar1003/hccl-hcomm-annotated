# 简介

本节提供CCU kernel内在本端HBM、本端MS Buffer与跨rank对端HBM之间搬运字节的数据搬运接口，可选在搬运过程中执行归约操作。

所有数据搬运接口均为异步接口，硬件完成搬运时自动将`event`的第`mask`位置1，下游通过`EventWait`等待完成信号。按数据通路分为以下两类：

| 类型 | 适用场景 | 接口 |
| --- | --- | --- |
| 本地操作 | 本端HBM↔本端HBM或本端HBM↔本端MS Buffer之间的拷贝与归约 | [LocalCopy](LocalCopy.md)、[LocalReduce](LocalReduce.md) |
| 跨rank操作 | 通过channel在本端与对端HBM之间读写数据，可选归约 | [Read](Read.md)、[ReadReduce](ReadReduce.md)、[Write](Write.md)、[WriteReduce](WriteReduce.md) |

跨rank操作要求所有`ChannelHandle`属于同一die，由框架统一校验。

## 接口列表

- [LocalCopy](LocalCopy.md)
- [LocalReduce](LocalReduce.md)
- [Read](Read.md)
- [ReadReduce](ReadReduce.md)
- [Write](Write.md)
- [WriteReduce](WriteReduce.md)

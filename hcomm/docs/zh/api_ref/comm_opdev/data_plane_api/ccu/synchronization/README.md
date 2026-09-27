# 简介

本节提供CCU kernel内通过Event和Notify协调异步操作完成顺序的同步接口。

CCU kernel中数据搬运、跨kernel阶段执行均为异步推进，需要在不同kernel或不同搬运操作之间显式建立"先后关系"时，由生产侧调用Record接口写入完成信号，消费侧调用Wait接口阻塞等待信号到达。按生产方与消费方的位置关系，同步分为以下三类：

| 类型 | 适用场景 | 生产侧接口 | 消费侧接口 |
| --- | --- | --- | --- |
| 本地Event | 同kernel：等待本kernel内发起的异步搬运完成 | [EventRecord](EventRecord.md) | [EventWait](EventWait.md) |
| 本地Notify | 同die内跨kernel：不同kernel间顺序协调，以字符串tag配对 | [LocalNotifyRecord](LocalNotifyRecord.md) | [LocalNotifyWait](LocalNotifyWait.md) |
| 远端Notify | 跨die（含跨device、跨节点）：通过channel传递信号 | [NotifyRecord](NotifyRecord.md) | [NotifyWait](NotifyWait.md) |

[WriteVariableWithNotify](WriteVariableWithNotify.md)是远端Notify的扩展接口，将"写远端Variable值"与"触发远端Notify"合并为原子操作，用于需要将标量值连同完成信号一起发送给对端的场景。

## 接口列表

- [EventRecord](EventRecord.md)
- [EventWait](EventWait.md)
- [LocalNotifyRecord](LocalNotifyRecord.md)
- [LocalNotifyWait](LocalNotifyWait.md)
- [NotifyRecord](NotifyRecord.md)
- [NotifyWait](NotifyWait.md)
- [WriteVariableWithNotify](WriteVariableWithNotify.md)

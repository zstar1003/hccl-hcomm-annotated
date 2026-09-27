# 简介

本节提供CCU kernel内Variable与外部数据源之间传递标量值的接口，用于在注册阶段声明运行期数据的来源或去处。

Variable在注册阶段仅是标量占位符，本身不带值。本节接口负责在kernel执行前/后与外部进行数据交换。外部来源分为以下两类：

| 来源/去处 | 地址何时确定 | 接口 |
| --- | --- | --- |
| `taskArgs[]`（host每次Launch时注入） | 运行期Launch时 | [LoadArg](LoadArg.md) |
| HBM立即数地址（注册期固定常量） | 注册阶段 | [Load](Load.md)（重载1/2）、[Store](Store.md)（重载1/2） |
| HBM间接地址（运行期由Variable决定） | 运行期 | [Load](Load.md)（重载3/4）、[Store](Store.md)（重载3/4） |

`Load`和`Store`均通过第一参数的类型（`uint64_t`或`Variable`）自动选择立即数寻址或间接寻址路径，各路径内部支持单个Variable与批量`Array<Variable>`两种粒度。

## 接口列表

- [LoadArg](LoadArg.md)
- [Load](Load.md)
- [Store](Store.md)

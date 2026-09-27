# 简介

本节提供CCU kernel内表达动态控制流的接口，分为三个子类：

| 子类 | 适用场景 | 接口 |
| --- | --- | --- |
| 软件分支/循环 | 灵活控制流，嵌套结构，body内可用任意CCU接口 | [CCU_IF](CCU_IF.md)、[CCU_ELSE](CCU_ELSE.md)、[CCU_WHILE](CCU_WHILE.md)、[CCU_DO](CCU_DO.md) |
| 硬件Loop | 大量同结构迭代，body内为本地搬运类操作，追求最小指令开销 | [Loop](Loop.md)、[LoopGroup](LoopGroup.md) |
| FuncBlock | 同一段逻辑在kernel内多处调用，节省SRAM | [Func](Func.md)、[CallFunc](CallFunc.md) |

三种子类的选择建议：

- 控制流逻辑复杂、有嵌套、body内有多种CCU操作 → 使用软件分支/循环宏。
- 大量同结构迭代且body内只有本地搬运操作 → 使用硬件Loop（`ccu::Loop`/`ccu::LoopGroup`）。
- 同一段逻辑在kernel内被调用两次或以上 → 使用FuncBlock（`ccu::Func`+`ccu::CallFunc`）。

三种子类可以组合：允许在软件循环（`CCU_WHILE`/`CCU_IF`）内嵌套硬件Loop。不要反向嵌套——硬件Loop body内不应使用软件控制流（`CCU_IF`/`CCU_WHILE`/`CCU_DO`）。框架不强制校验反向嵌套，但行为未定义，请勿使用；仅`CallFunc`在硬件Loop body内会抛出携带`CCU_E_INTERNAL`的异常（经kernel注册入口统一捕获后，注册接口返回`CCU_E_INTERNAL`）。

## 接口列表

- [CCU_IF](CCU_IF.md)
- [CCU_ELSE](CCU_ELSE.md)
- [CCU_WHILE](CCU_WHILE.md)
- [CCU_DO](CCU_DO.md)
- [Loop](Loop.md)
- [LoopGroup](LoopGroup.md)
- [Func](Func.md)
- [CallFunc](CallFunc.md)

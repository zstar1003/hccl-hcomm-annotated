# HCCL架构文档

本文档描述HCCL（Huawei Collective Communication Library）的软件架构。

## 文档目录

- [系统概述](./overview.md) - 系统整体架构与模块划分
- [base_comm](./base_comm/) - 基础通信模块
- [coll_comm](./coll_comm/) - 集合通信域管理模块

## 架构原则

1. **模块化设计** - 各模块职责单一，模块间解耦
2. **平台抽象** - 屏蔽底层硬件差异，提供统一接口
3. **可扩展性** - 支持新算法、新通信原语的灵活扩展

## 相关链接

- [贡献指南](../../../CONTRIBUTING.md)
- [RFC文档](../rfcs/)
- [开发者指南](../comm_op_dev_guide/)

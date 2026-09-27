# HCCL 代码示例

简体中文 | [English](./README_en.md)

本目录提供了不同场景下使用 HCCL 接口实现集合通信功能的示例代码。

## 通信域管理

- [每个进程管理一个 NPU 设备（基于 root 节点信息初始化通信域）](./01_communicators/01_one_device_per_process)
- [每个进程管理一个 NPU 设备（基于 rank table 初始化通信域）](./01_communicators/02_one_device_per_process_rank_table)
- [每个线程管理一个 NPU 设备（基于 root 节点信息初始化通信域）](./01_communicators/03_one_device_per_pthread)

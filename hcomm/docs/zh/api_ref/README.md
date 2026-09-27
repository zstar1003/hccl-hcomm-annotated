# HCOMM API总览

- [对外头文件与库文件说明](./hcomm_header_and_lib.md)：列举HCOMM正式对外的头文件与库文件，供通信库/算子开发与部署参考。
- [通信域创建与管理接口（C语言）](./comm_mgr_c/README.md)：用于实现单算子模式下的框架适配，实现分布式能力。
- [通信域创建与管理接口（Python语言）](./comm_mgr_python/README.md)：用于实现图模式下的框架适配，当前仅用于TensorFlow网络在NPU执行分布式优化。
- [通信算子开发接口](./comm_opdev/README.md)：提供控制面接口与数据面接口，支撑开发者自定义通信算子。

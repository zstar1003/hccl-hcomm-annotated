# Unit tests

本目录实现 RFC 第 10 节中的 T1、T2、T3、T4、T5、T6。测试均为离线 UT，不连接 NPU、`rpc_host` 或交换机；ST 不在本目录范围。

| 测试 | 文件 | 覆盖内容 |
| --- | --- | --- |
| T1 | `t1_config_test.cpp` | 配置默认值、重复/非法 Device ID、配对章节、正整数和输出路径校验 |
| T2 | `t2_topology_test.cpp` | ControlTopo 路由、Device 数完整性、LLDP 分域、未知邻居、MeshTopo JSON 往返和异常访问 |
| T3 | `t3_probe_plan_test.cpp` | L1 环、L2 相邻域 FullMesh、任务规模、去重和单域降级 |
| T4 | `t4_metric_reduction_test.cpp` | P90/P99/Mean/Pass 映射、全丢包、越界 Pass、缺字段和非法 times |
| T5 | `t5_link_solver_test.cpp` | 精确解、超定解、秩不足和 NaN 输入 |
| T6 | `t6_metrics_collector_test.cpp` | 计数器文本解析、缺 Key/非法值、`valid` 标记、真实零值、Device ID 归一化和重置/回绕差值 |

运行方式：

```bash
cmake -S . -B build-test \
  -DTHIRDLIB_ROOT="$THIRDLIB_ROOT" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-test -j"$(nproc)"
ctest --test-dir build-test --output-on-failure -L UT
```

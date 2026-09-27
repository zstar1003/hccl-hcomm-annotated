# 用户运行

## 环境变量配置

```bash
cd .../disp_probe-main

export THIRDLIB_ROOT=/usr/local/third_lib
export ASCEND_HOME_PATH=/usr/local/Ascend
export ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux
source "$THIRDLIB_ROOT/share/disp_probe/third_party/env.sh"
```

## 编译

```bash
cmake -S . -B build \
  -DTHIRDLIB_ROOT="$THIRDLIB_ROOT" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j"$(nproc)"
```

## 运行

```bash
# 清理残留进程、分发 rpc_host 和配置文件、启动 dev 侧程序
./run.sh deploy

# 启动新终端， 启动监控系统（网络状态探测）
./build/probe_controller -f ./control_json/910b2_info.json
```

### 开启NPU侧探测日志

​`--pingpong-local-log`：开启每个NPU侧结果日志。

​`--pingpong-log-dir <PATH>`：NPU侧日志根目录，默认为`/root/output`。

```bash
./run.sh deploy --pingpong-log /root/output
```

`rpc_host` 默认读取远端部署目录下的 `control_json/910b2_info.json`，也可以用 `-f` 指定其他 control_json；v2 配置会将 `probe.scope` 中的 host id 转换为对应 IP，再根据本机 IP 对应的卡号初始化 HCCN，未在探测范围中使用的卡不会初始化。

`probe.pingpong.payload_len` 用于设置 HCCN Rping Payload 长度，默认 `12` 字节，范围 `1～1500`。Payload 在每个目标 AddTarget 时使用随机字节填充；修改配置后需要重启对应 Host 上的 `rpc_host` 才会生效。

`probe.pingpong.interval_ms` 用于设置传给 `HccnRpingBatchPingStart` 的 Ping 报文发送间隔，默认 `1` 毫秒，必须为正整数；修改配置后需要重启对应 Host 上的 `rpc_host`。

### 全网络拓扑生成

```bash
./build/probe_topo -f ./control_json/910b2_info.json
```

拓扑探测默认分为四轮；`probe.topology.l2_path_aware` 默认为 `true`，如设置为 `false` 则跳过第 4 轮：

1. batch 1：环状 tracert，用于获取网络分层，输出基础分层拓扑到 `output/probe_topo.json`。
2. batch 2：对每张卡执行 LLDP 探测，通过交换机管理面 IP 合并第 0 层分组，输出到 `output/probe_topo_lldp.json`。
3. batch 3：对同层不同交换机分组之间做多路径覆盖；当 `probe.topology.topology_optimized=true` 时，按 ToR 域环形探测同编号设备，即 `TOPO[0,X]` 的第 N 个设备探测 `TOPO[0,X+1]` 的第 N 个设备，最后一个域再探测 `TOPO[0,0]`；当 `topology_optimized=false` 时，探测不同 ToR 域间所有设备两两 directed pair。每条 directed pair 的覆盖次数等于 `control_json/910b2_info.json` 中的 `probe.topology.sport_count`，输出到 `output/allpath.json`。
4. batch 4：L2 域间环形 FullMesh 路径探测。开启 `probe.topology.l2_path_aware=true` 时执行，固定使用 `srcPort=49152`，由 `TOPO[0,X]` 的每个设备依次探测 `TOPO[0,X+1]` 的所有设备，最后一个 `TOPO[0,X+1]` 再探测 `TOPO[0,0]`，用于获取 L2 pair 的具体多跳路径，输出到 `output/l2_fullmesh_path.json`。

输出在： `output/probe_topo.json` 、`output/probe_topo_lldp.json`（LLDP聚合后） 、`output/allpath.json`；开启 `l2_path_aware` 时额外输出 `output/l2_fullmesh_path.json`。

拓扑探测完成后，读取`output/probe_topo_lldp.json` 、`output/allpath.json`，进行拓扑图绘制。

```bash
python3 ./plot/topo_plot.py
```

最终输出的拓扑图保存在output/topo_plot/topo.png。

### 执行pinglist

`probe_controller` 默认读取 `output/probe_topo_lldp.json` 作为 L1 pinglist 拓扑输入。默认会额外添加 L2 域间 环形FullMesh 探测：固定使用 `srcPort=49152`，由 `TOPO[0,X]` 的每个设备依次探测 `TOPO[0,X+1]` 的所有设备，最后一个 `TOPO[0,X+1]` 再探测 `TOPO[0,0]`。

```bash
./build/probe_controller -f ./control_json/910b2_info.json
```

`probe_controller` 支持的运行选项如下：

| 选项 | 说明 |
| --- | --- |
| `-h, --help` | 输出帮助信息并退出。 |
| `-f, --file <PATH>` | 指定 control_json 配置文件路径。 |
| `--print-pingpong-plan` | 只打印 pingpong 探测计划，打印后退出，不下发 pinglist、不执行 pingpong。 |
| `--l1-only` | 只构建并求解 L1(ToR 下侧链路) 探测任务，关闭 L2 域间 FullMesh 探测和 L2 输出。 |
| `--no-metrics` | 关闭 PFC/CNP 计数器采集。默认开启计数器采集。 |

默认会每秒采集一次探测范围（v2 `probe.scope` 或 v1 `probe_scope`）涉及 NPU 的 PFC/CNP 计数器并输出到 `output/<时间>/metrics/`。如需关闭计数器采集：

```bash
./build/probe_controller -f ./control_json/910b2_info.json --no-metrics
```

如果只需要构建 L1(ToR 下侧链路) 的探测计划并进行求解，关闭 L2 FullMesh 探测计划和 L2 扣减输出：

```bash
./build/probe_controller -f ./control_json/910b2_info.json --l1-only
./build/probe_controller -f ./control_json/910b2_info.json --print-pingpong-plan --l1-only
```

### 结果分析

网络状态分析默认读取 `output/<时间>/` 下最近一次运行结果，并提取最近 100 轮数据：

```bash
python3 ./plot/status_plot.py
```

也可以指定读取某一次运行结果：

```bash
python3 ./plot/status_plot.py --input-dir "output/2026-07-02 21:10:19"
# 或
python3 ./plot/status_plot.py "output/2026-07-02 21:10:19"
```

状态图输出到 `output/<时间>/status_plot/`：

- `l1_latency.png`：L1 latency 最近 100 轮变化。
- `l1_passrate.png`：L1 pass rate 最近 100 轮变化。
- `l2_passrate.png`：L2 pass rate 最近 100 轮变化。
- `l2_latency_top10.png`：按最近 100 轮平均 L2 latency 选出的前 10 个 pair 的变化。
- `l2_latency_top10.txt`：前 10 个 L2 latency pair 的平均值列表；如果存在 `output/l2_fullmesh_path.json`，会同时输出该 pair 的 `srcPort=49152` 多跳路径。

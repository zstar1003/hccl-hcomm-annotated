# User Execution

## Environment Variable Configuration

```bash
cd .../disp_probe-main

export THIRDLIB_ROOT=/usr/local/third_lib
export ASCEND_HOME_PATH=/usr/local/Ascend
export ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux
source "$THIRDLIB_ROOT/share/disp_probe/third_party/env.sh"
```

## Compilation

```bash
cmake -S . -B build \
  -DTHIRDLIB_ROOT="$THIRDLIB_ROOT" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j"$(nproc)"
```

## Execution

```bash
# Clean up residual processes, distribute rpc_host and configuration files, start device-side programs
./run.sh deploy

# Open a new terminal and start the monitoring system (network status probing)
./build/probe_controller -f ./control_json/910b2_info.json
```

### Enable NPU-Side Probing Logs

`--pingpong-local-log`: Enable result logging on each NPU side.

`--pingpong-log-dir <PATH>`: NPU-side log root directory, default `/root/output`.

```bash
./run.sh deploy --pingpong-log /root/output
```

`rpc_host` reads `control_json/910b2_info.json` from the remote deployment directory by default. You can also use `-f` to specify a different control_json. The v2 configuration converts the host ID in `probe.scope` to the corresponding IP and initializes HCCN based on the device IDs mapped to the local IP. Devices not in the probe scope are not initialized.

`probe.pingpong.payload_len` sets the HCCN Rping payload length, with a default of `12` bytes and a range of `1-1500`. The payload is filled with random bytes during AddTarget for each target. After modifying the configuration, restart `rpc_host` on the corresponding host for the changes to take effect.

`probe.pingpong.interval_ms` sets the ping packet sending interval passed to `HccnRpingBatchPingStart`, with a default of `1` millisecond. The value must be a positive integer. After modifying the configuration, restart `rpc_host` on the corresponding host.

### Full Network Topology Generation

```bash
./build/probe_topo -f ./control_json/910b2_info.json
```

Topology probing runs in four batches by default. `probe.topology.l2_path_aware` defaults to `true`; if set to `false`, batch 4 is skipped:

1. Batch 1: Ring tracert to obtain network layering. Outputs the base layered topology to `output/probe_topo.json`.
2. Batch 2: LLDP probing for each device. Merges layer-0 groups by switch management IP. Outputs to `output/probe_topo_lldp.json`.
3. Batch 3: Multi-path coverage between different switch groups at the same layer. When `probe.topology.topology_optimized=true`, ring probing is performed on same-index devices across ToR domains: the Nth device in `TOPO[0,X]` probes the Nth device in `TOPO[0,X+1]`, and the last domain probes back to `TOPO[0,0]`. When `topology_optimized=false`, all directed pairs between devices in different ToR domains are probed. The coverage count per directed pair equals `probe.topology.sport_count` in `control_json/910b2_info.json`. Outputs to `output/allpath.json`.
4. Batch 4: L2 inter-domain ring FullMesh path probing. Executed when `probe.topology.l2_path_aware=true`. Uses a fixed `srcPort=49152`. Each device in `TOPO[0,X]` probes all devices in `TOPO[0,X+1]` sequentially, and the last `TOPO[0,X+1]` probes back to `TOPO[0,0]`. This obtains the specific multi-hop paths for L2 pairs. Outputs to `output/l2_fullmesh_path.json`.

Output files: `output/probe_topo.json`, `output/probe_topo_lldp.json` (after LLDP aggregation), and `output/allpath.json`. When `l2_path_aware` is enabled, `output/l2_fullmesh_path.json` is also generated.

After topology probing completes, read `output/probe_topo_lldp.json` and `output/allpath.json` to generate the topology chart.

```bash
python3 ./plot/topo_plot.py
```

The final topology chart is saved to `output/topo_plot/topo.png`.

### Execute PingList

`probe_controller` reads `output/probe_topo_lldp.json` as the L1 pinglist topology input by default. It also adds L2 inter-domain ring FullMesh probing by default: uses a fixed `srcPort=49152`, where each device in `TOPO[0,X]` probes all devices in `TOPO[0,X+1]` sequentially, and the last `TOPO[0,X+1]` probes back to `TOPO[0,0]`.

```bash
./build/probe_controller -f ./control_json/910b2_info.json
```

The following runtime options are supported by `probe_controller`:

| Option | Description |
| --- | --- |
| `-h, --help` | Print help information and exit. |
| `-f, --file <PATH>` | Specify the control_json configuration file path. |
| `--print-pingpong-plan` | Print the pingpong probe plan only and exit; do not distribute the pinglist or execute pingpong. |
| `--l1-only` | Build and solve L1 (below-ToR link) probe tasks only; disable L2 inter-domain FullMesh probing and L2 output. |
| `--no-metrics` | Disable PFC/CNP counter collection. Counter collection is enabled by default. |

By default, PFC/CNP counters for NPUs in the probe scope (v2 `probe.scope` or v1 `probe_scope`) are collected every second and output to `output/<time>/metrics/`. To disable counter collection:

```bash
./build/probe_controller -f ./control_json/910b2_info.json --no-metrics
```

To build only the L1 (below-ToR link) probe plan and perform solving, disabling L2 FullMesh probe plans and L2 subtraction output:

```bash
./build/probe_controller -f ./control_json/910b2_info.json --l1-only
./build/probe_controller -f ./control_json/910b2_info.json --print-pingpong-plan --l1-only
```

### Result Analysis

Network status analysis reads the most recent run results from `output/<time>/` by default and extracts the most recent 100 turns of data:

```bash
python3 ./plot/status_plot.py
```

You can also specify a particular run result:

```bash
python3 ./plot/status_plot.py --input-dir "output/2026-07-02 21:10:19"
# Or
python3 ./plot/status_plot.py "output/2026-07-02 21:10:19"
```

Status charts are output to `output/<time>/status_plot/`:

- `l1_latency.png`: L1 latency changes over the most recent 100 turns.
- `l1_passrate.png`: L1 pass rate changes over the most recent 100 turns.
- `l2_passrate.png`: L2 pass rate changes over the most recent 100 turns.
- `l2_latency_top10.png`: Changes for the top 10 pairs selected by average L2 latency over the most recent 100 turns.
- `l2_latency_top10.txt`: A list of average values for the top 10 L2 latency pairs. If `output/l2_fullmesh_path.json` exists, the `srcPort=49152` multi-hop path for each pair is also output.

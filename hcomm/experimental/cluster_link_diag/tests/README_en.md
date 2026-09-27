# Unit Tests

This directory implements T1, T2, T3, T4, T5, and T6 from Section 10 of the RFC. All tests are offline unit tests and do not connect to NPUs, `rpc_host`, or switches. System tests are not in scope for this directory.

| Test | File | Coverage |
| --- | --- | --- |
| T1 | `t1_config_test.cpp` | Configuration defaults, duplicate/illegal device IDs, pairing sections, positive integer and output path validation |
| T2 | `t2_topology_test.cpp` | ControlTopo routing, device count completeness, LLDP domain partitioning, unknown neighbors, MeshTopo JSON round-trip, and exception access |
| T3 | `t3_probe_plan_test.cpp` | L1 ring, L2 adjacent-domain FullMesh, task scale, deduplication, and single-domain degradation |
| T4 | `t4_metric_reduction_test.cpp` | P90/P99/Mean/Pass mapping, total packet loss, out-of-range Pass, missing fields, and illegal times |
| T5 | `t5_link_solver_test.cpp` | Exact solution, overdetermined solution, rank deficiency, and NaN input |
| T6 | `t6_metrics_collector_test.cpp` | Counter text parsing, missing key/illegal value, `valid` flag, genuine zero value, device ID normalization, and reset/wraparound delta |

How to run:

```bash
cmake -S . -B build-test \
  -DTHIRDLIB_ROOT="$THIRDLIB_ROOT" \
  -DBUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-test -j"$(nproc)"
ctest --test-dir build-test --output-on-failure -L UT
```

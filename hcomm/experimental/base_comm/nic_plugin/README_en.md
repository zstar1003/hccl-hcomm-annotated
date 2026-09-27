# HCOMM Experimental NIC Plugin Guide

## Overview

The HCOMM Experimental NIC Plugin extends the host-side NIC communication implementation in general server scenarios. The plugin is built and deployed as an independent `.so` file. When creating a HOST endpoint or channel, HCOMM searches for registered plugins based on `EndpointDesc.protocol`. If a matching plugin protocol is found, the plugin implementation is used. Otherwise, the original built-in implementation is used.

This directory provides two example plugins:

| Plugin | Artifact | Registered Protocol |
| --- | --- | --- |
| HOST RoCE plugin | `libhcomm_cpu_roce_plugin.so` | `COMM_PROTOCOL_ROCE` |
| HOST UB plugin | `libhcomm_cpu_ub_plugin.so` | `COMM_PROTOCOL_UB_CTP` |

## Building and Packaging

Before compiling, complete the dependency installation, CANN software package installation, and environment variable configuration as described in `docs/en/build/build.md`.

After navigating to the repository root directory, execute the following command to compile the host package with experimental plugins enabled:

```bash
bash build.sh --pkg --experimental
```

To compile both the host and device packages:

```bash
bash build.sh --pkg --full --experimental
```

If the compilation environment does not have network access, download the third-party dependency packages in a networked environment, upload them to the compilation environment, and specify the dependency package path using `--cann_3rd_lib_path`:

```bash
bash build.sh --pkg --experimental --cann_3rd_lib_path={your_3rd_party_path}
```

After compilation, the HCOMM software package is generated in the `build_out` directory at the repository root:

```text
./build_out/cann-hcomm_<version>_linux-<arch>.run
```

`<version>` is the software version number, and `<arch>` is the system architecture, for example, `x86_64` or `aarch64`.

## Installation and Deployment

Install the compiled HCOMM software package:

```bash
bash ./build_out/cann-hcomm_<version>_linux-<arch>.run --full
```

Replace the package name in the command with the actual generated file name. After installation, the experimental plugins are installed in the HCOMM plugin directory:

```text
${ASCEND_HOME_PATH}/hcomm_plugin/libhcomm_cpu_roce_plugin.so
${ASCEND_HOME_PATH}/hcomm_plugin/libhcomm_cpu_ub_plugin.so
```

For manual debugging, you can also copy the plugin `.so` files directly to the `${ASCEND_HOME_PATH}/hcomm_plugin/` directory.

## Signature Verification Notes

Host-only plugin usage does not involve device package signature verification. If you compile and install the `--full` package and run on-board tests, refer to `docs/en/build/build.md` for the steps to disable signature verification.

## Enabling at Runtime

Before running, load the CANN or HCOMM environment variables. For the default installation path, execute:

```bash
source /usr/local/Ascend/cann/set_env.sh
```

For a specified installation path, execute:

```bash
source ${install_path}/cann/set_env.sh
```

The plugin loading rules are as follows:

- When `ASCEND_HOME_PATH` is not empty, HCOMM scans `${ASCEND_HOME_PATH}/hcomm_plugin/*.so`.
- When `ASCEND_HOME_PATH` is empty, HCOMM reads the plugin path specified by `HCOMM_NIC_PLUGIN_SO`.
- `HCOMM_NIC_PLUGIN_SO` supports multiple `.so` paths separated by colons.

Example:

```bash
export HCOMM_NIC_PLUGIN_SO=/path/to/libhcomm_cpu_roce_plugin.so:/path/to/libhcomm_cpu_ub_plugin.so
```

Business code does not need to call plugin interfaces directly. When HCOMM creates a HOST endpoint through `HcommEndpointCreate`, it searches for plugins based on `EndpointDesc.protocol`. If the protocol is registered by a plugin, subsequent endpoint, channel, and data plane interfaces are dispatched to the plugin `ops`. If no matching plugin is found, the original built-in path is used.

## Usage Conditions and Limitations

- The plugin only applies to endpoints where `EndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_HOST`.
- The current example plugins target general server host-only scenarios.
- The current loader skips plugin loading when it detects that the number of runtime devices is non-zero.
- When multiple plugins register the same protocol, the later-loaded plugin overrides the earlier one.
- The plugin ABI must match the magic word, version, and size defined in `HcommNicPluginInfo`, `HcommNicEndpointOps`, and `HcommNicChannelOps`.

## Custom Plugin Development Entry Points

Custom plugins need to export the following C ABI symbols:

```cpp
const HcommNicPluginInfo *HcommNicPluginGetInfo(void);
int32_t HcommNicPluginCreateEndpoint(
    const EndpointDesc *endpointDesc, void **outCtx, HcommNicEndpointOps **outOps);
int32_t HcommNicPluginCreateChannel(
    void *epCtx, const HcommChannelDesc *channelDesc, void **outCtx, HcommNicChannelOps **outOps);
```

Development requirements:

- Declare the plugin name and supported `CommProtocol` in `HcommNicPluginInfo`.
- Fill `HcommNicEndpointOps` for the endpoint, implementing interfaces such as initialization, memory registration, export, import, and destroy.
- Fill `HcommNicChannelOps` for the channel, implementing interfaces such as initialization, status query, read, write, notify, fence, and destroy.
- Ensure the magic word, version, and size in the ABI header match the definitions in `hcomm_nic_plugin.h`.

For reference, see the following files:

- `host_roce_plugin.cc`: HOST RoCE plugin entry point.
- `host_ub_plugin.cc`: HOST UB plugin entry point.
- `nic_plugin_ops.h`: Example plugin ops adaptation template.
- `plugin_core.cc`: Plugin common auxiliary logic.

## Troubleshooting

If the plugin does not take effect, check the following items in order:

1. Confirm that `--experimental` was added during the build and that the plugin `.so` is included in the software package.
2. Confirm that the plugin `.so` is installed in `${ASCEND_HOME_PATH}/hcomm_plugin/`.
3. Confirm that `ASCEND_HOME_PATH` or `HCOMM_NIC_PLUGIN_SO` is set according to the current loading rules.
4. Confirm that the business creates a HOST endpoint, that is, `EndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_HOST`.
5. Confirm that `EndpointDesc.protocol` is registered in the plugin `HcommNicPluginInfo.protocols`.
6. Check the runtime log for the `[NicPlugin]` keyword to see whether the plugin was scanned, loaded, and its protocol registered.

# HCOMM Experimental NIC Plugin 使用说明

## 功能简介

HCOMM Experimental NIC Plugin 用于在通用服务器场景下扩展 host 侧网卡通信实现。插件以独立`.so`形式构建和部署，HCOMM 在创建 HOST endpoint/channel 时根据 `EndpointDesc.protocol` 查找已注册插件；命中插件协议后走插件实现，未命中时继续走原有内置实现。

当前目录提供两个示例插件：

| 插件 | 产物 | 注册协议 |
| --- | --- | --- |
| HOST RoCE 插件 | `libhcomm_cpu_roce_plugin.so` | `COMM_PROTOCOL_ROCE` |
| HOST UB 插件 | `libhcomm_cpu_ub_plugin.so` | `COMM_PROTOCOL_UB_CTP` |

## 编译打包

编译前请先参考仓库根目录下的 `docs/zh/build/build.md` 完成依赖安装、CANN 软件包安装和环境变量配置。

进入代码仓根目录后，执行以下命令编译 host 包并开启 experimental 插件：

```bash
bash build.sh --pkg --experimental
```

如需同时编译 host + device 包，执行：

```bash
bash build.sh --pkg --full --experimental
```

如果编译环境无法访问网络，可在联网环境下载第三方依赖包后上传到编译环境，并通过`--cann_3rd_lib_path` 指定依赖包路径：

```bash
bash build.sh --pkg --experimental --cann_3rd_lib_path={your_3rd_party_path}
```

编译完成后，会在仓库根目录的 `build_out` 目录生成 HCOMM 软件包：

```text
./build_out/cann-hcomm_<version>_linux-<arch>.run
```

其中 `<version>` 为软件版本号，`<arch>` 为系统架构，例如 `x86_64` 或 `aarch64`。

## 安装部署

安装编译生成的 HCOMM 软件包：

```bash
bash ./build_out/cann-hcomm_<version>_linux-<arch>.run --full
```

请将命令中的软件包名称替换为实际生成的文件名。安装完成后，experimental 插件会安装到 HCOMM 插件目录：

```text
${ASCEND_HOME_PATH}/hcomm_plugin/libhcomm_cpu_roce_plugin.so
${ASCEND_HOME_PATH}/hcomm_plugin/libhcomm_cpu_ub_plugin.so
```

手动调试时，也可以直接将插件 `.so` 复制到 `${ASCEND_HOME_PATH}/hcomm_plugin/` 目录。

## 验签说明

host-only 插件使用不涉及 device 包验签；如果编译安装 `--full` 包并运行上板测试，关闭验签步骤参考`docs/zh/build/build.md`。

## 运行启用

运行前先加载 CANN/HCOMM 环境变量。默认路径安装时可执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

指定路径安装时可执行：

```bash
source ${install_path}/cann/set_env.sh
```

插件加载规则如下：

- `ASCEND_HOME_PATH` 非空时，HCOMM 扫描 `${ASCEND_HOME_PATH}/hcomm_plugin/*.so`。
- `ASCEND_HOME_PATH` 为空时，HCOMM 读取 `HCOMM_NIC_PLUGIN_SO` 指定的插件路径。
- `HCOMM_NIC_PLUGIN_SO` 支持使用冒号分隔多个 `.so` 路径。

示例：

```bash
export HCOMM_NIC_PLUGIN_SO=/path/to/libhcomm_cpu_roce_plugin.so:/path/to/libhcomm_cpu_ub_plugin.so
```

业务代码无需直接调用插件接口。HCOMM 在 `HcommEndpointCreate` 创建 HOST endpoint 时，会根据`EndpointDesc.protocol` 查找插件；如果协议由插件注册，则后续 endpoint、channel 和数据面接口会分发到插件`ops`；如果没有匹配插件，则保持原有内置路径。

## 使用条件和限制

- 插件仅对 `EndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_HOST` 的 endpoint 生效。
- 当前示例插件面向通用服务器 host-only 场景。
- 当前加载器在检测到 runtime device 数量非 0 时会跳过插件加载。
- 多个插件注册同一协议时，后加载的插件会覆盖先加载插件。
- 插件 ABI 需要匹配 `HcommNicPluginInfo`、`HcommNicEndpointOps`、`HcommNicChannelOps` 中定义的magic word、version和size。

## 自定义插件开发入口

自定义插件需要导出以下 C ABI 符号：

```cpp
const HcommNicPluginInfo *HcommNicPluginGetInfo(void);
int32_t HcommNicPluginCreateEndpoint(
    const EndpointDesc *endpointDesc, void **outCtx, HcommNicEndpointOps **outOps);
int32_t HcommNicPluginCreateChannel(
    void *epCtx, const HcommChannelDesc *channelDesc, void **outCtx, HcommNicChannelOps **outOps);
```

开发时需要：

- 在 `HcommNicPluginInfo` 中声明插件名称和支持的 `CommProtocol`。
- 为 endpoint 填充 `HcommNicEndpointOps`，实现初始化、内存注册、导出、导入和销毁等接口。
- 为 channel 填充 `HcommNicChannelOps`，实现初始化、状态查询、读写、notify、fence 和销毁等接口。
- 保持 ABI header 中的 magic word、version 和 size 与 `hcomm_nic_plugin.h` 定义一致。

可参考以下文件：

- `host_roce_plugin.cc`：HOST RoCE 插件入口。
- `host_ub_plugin.cc`：HOST UB 插件入口。
- `nic_plugin_ops.h`：示例插件 ops 适配模板。
- `plugin_core.cc`：插件公共辅助逻辑。

## 问题定位

如插件未生效，可按以下顺序检查：

1. 确认构建时已添加 `--experimental`，并且软件包中包含插件 `.so`。
2. 确认插件 `.so` 已安装到 `${ASCEND_HOME_PATH}/hcomm_plugin/`。
3. 确认 `ASCEND_HOME_PATH` 或 `HCOMM_NIC_PLUGIN_SO` 设置符合当前加载规则。
4. 确认业务创建的是 HOST endpoint，即 `EndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_HOST`。
5. 确认 `EndpointDesc.protocol` 已在插件 `HcommNicPluginInfo.protocols` 中注册。
6. 查看运行日志中的 `[NicPlugin]` 关键字，确认插件是否被扫描、加载和注册协议。

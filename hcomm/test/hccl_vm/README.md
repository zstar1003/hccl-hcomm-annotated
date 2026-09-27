# HCCL-VM 指导手册

## 1. 概述

HCCL-VM是面向华为昇腾NPU卡的高性能集合通信的虚拟执行环境，该工具旨在无真实昇腾硬件的条件下，实现HCCL集合通信算子的开发和功能验证。

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. 前置依赖

|   依赖项   |     版本要求     |
| -------- | ------------- |
| 系统架构  | x86_64 Ubuntu22.04及以上 |
| 规格约束  | Ascend950，其余参照[约束详情](#45-工具规格约束) |

### 2.1 CANN包安装

安装最新版本CANN Toolkit开发套件包和CANN ops算子包 [下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)

```bash
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# 安装命令
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```

### 2.2 hccl_test编译

hccl_test是昇腾官方提供的HCCL性能测试工具，详见[HCCL性能测试工具](https://www.hiascend.com/document/redirect/CANNCommunityToolHcclTest)，HCCL-VM支持在虚拟环境中运行hccl_test用例。请先参照[hccl_test用例构建](#42-hccl-test用例构建)章节进行用例二进制程序的编译。

备注：可选，未来支持Pytorch用例。

---

## 3. 快速上手

### 3.1 一键安装

一行完成依赖安装、源码拉取、CANN 检测与编译（默认使用 `main` profile，对应 hcomm/hccl 的 `master` 主线）。工作目录与手动安装保持一致，用 `/home/workspace`（后文示例路径均以此为准）：

```bash
# 创建并进入工作目录（脚本默认装到当前目录）
mkdir -p /home/workspace && cd /home/workspace
curl -fsSL https://raw.gitcode.com/cann/hcomm/raw/master/test/hccl_vm/hccl_vm_installer | bash
```

也可下载后本地运行（便于先审阅或离线分发）：`bash hccl_vm_installer`；或用 `--workspace` 显式指定：`... | bash -s -- --workspace /home/workspace`。

**前提**：x86_64 Linux；工具链需满足 hcomm build.md 要求——gcc/g++ 7.3.0–13.3.x、cmake ≥ 3.16.0（同时约束宿主与 aarch64 交叉编译器）。Ubuntu 22.04 / 24.04 开箱即用；更高版本默认 gcc（14/15）超范围，脚本会告警并继续尝试，建议在满足范围的环境编译。

**CANN**：脚本只在工作目录 `<workspace>/Ascend`（或 `--ascend-path` 指定的路径）探测 CANN；未找到即自动下载配套版本并安装到该处，root 与普通用户行为一致。`--offline` 只检测、从不下载；内网无公网时自动回退为打印自备 CANN 引导。

**hccl_test**：默认一并编译 OpenMPI 与 hccl_test 性能测试工具，`--skip-hccl-test` 可关闭。

**常用参数**：

- `--profile <名>`：配套方案（默认 `main`，`--list-profiles` 列全部）
- `--workspace <路径>`：工作目录，源码／编译／产物所在（默认当前目录）
- `--ascend-path <路径>`：指定 CANN 目录，有则复用、无则装到此处
- `--reinstall-cann`：重新下载覆盖现有 CANN（版本不匹配时用；默认保留）
- `--offline`：只用现有 CANN、从不下载
- `--skip-hccl-test`：跳过 hccl_test 编译
- `-h`：完整帮助

完成后工具位于 `/home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin/hccl-vm`。删除工作目录即可清理本工具产物（apt 安装的系统依赖如需卸载请自行 `apt remove`），本工具不改动 CANN。（若用 `--workspace` 指定了别的目录，后文示例中的 `/home/workspace` 请相应替换。）

> 一键安装已自动完成 `build.sh` 编译与 `build_pkg.sh` 子包安装（含 AICPU/AIV 所需的设备侧符号），装完后 [使用示例](#33-使用示例) 中 CCU/AICPU/AIV 各模式均可直接运行，无需再单独执行 `build_pkg.sh`。

### 3.2 手动构建&安装

```bash
# 1. 创建工作目录
mkdir -p /home/workspace
cd /home/workspace

# 2. 下载依赖源码
git clone https://gitcode.com/cann/hccl.git
git clone https://gitcode.com/cann/hcomm.git

# 3. 安装第三方依赖
sudo apt-get update
sudo apt install build-essential cmake libsqlite3-dev libboost-all-dev rdma-core libibverbs-dev pkg-config gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user-static binfmt-support

# 4. 编译HCCL-VM工具，下载hcomm代码之后，工具源码所在路径：/home/workspace/hcomm/test/hccl_vm
cd /home/workspace/hcomm/test/hccl_vm
source /home/workspace/Ascend/cann/set_env.sh
export HCCL_CODE_HOME=/home/workspace/hccl
export HCOMM_CODE_HOME=/home/workspace/hcomm
bash ./build.sh --full

# 5.从CANN安装目录拷贝解压aicpu_hcxx.tar.gz
bash build_pkg.sh
```

### 3.3 使用示例

#### 3.3.1 环境配置

请参照[hccl_rootinfo文件内容](#47-hccl_rootinfojson文件)，创建并配置hccl_rootinfo.json文件。

#### 3.3.2 CCU模式

1. 配置环境变量。

   ```bash
   # 进入工具安装目录
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
   source /home/workspace/Ascend/cann/set_env.sh
   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
   export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
   export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
   ```

2. 执行。

   ```bash
   # 需要进入到新的bin文件目录下执行hccl-vm
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
   
   # 选择昇腾集群拓扑配置文件，启动工具，初始化集群环境，进入工具命令行
   ./hccl-vm start ascend950_cluster_32_server_normal.yaml
   
   # 如需启用runner插件（可选）
   (hvm)$> hccl-vm plugin install @runner
   
   # 选择本次算子执行的通信域配置文件（在1个超节点1个Server1个NPU的集群环境运行hccl_test用例）
   (hvm)$> hccl-vm mock-comm 112
   (hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   
   # 执行checker校验
   (hvm)$> hccl-vm plugin run @checker
   
   # 退出工具终端
   (hvm)$> exit
   ```

3. 验证hccl_test用例运行结果。

   [Runner结果查看](#491-runner插件结果) 
   [Checker结果查看](#492-checker插件结果)

#### 3.3.3 AICPU模式

AICPU展开模式需要将算法展开步骤放到设备侧执行，因此hccl-vm工具需要将HCCL的设备侧的符号编译并模拟执行。由于设备侧符号是ARM架构的，因此在X86环境上编译时需要借助交叉编译器，运行时需要借助QEMU实现AICPU模式的模拟运行。

1. 环境变量配置。

   ```bash
   # 进入工具安装目录
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
   source /home/workspace/Ascend/cann/set_env.sh
   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
   export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
   export HCCL_OP_EXPANSION_MODE="AI_CPU"
   ```

2. 执行

   ```bash
   # 需要进入到新的bin文件目录下执行hccl-vm
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
   
   # 选择昇腾集群拓扑配置文件，启动工具，初始化集群环境，进入工具命令行
   ./hccl-vm start ascend950_cluster_32_server_normal.yaml
   
   # 如需启用runner插件（可选）
   (hvm)$> hccl-vm plugin install @runner
   
   # 选择本次算子执行的通信域配置文件（在1个超节点1个Server1个NPU的集群环境运行hccl_test用例）
   (hvm)$> hccl-vm mock-comm 112
   (hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   
   # 执行checker校验
   (hvm)$> hccl-vm plugin run @checker
   
   # 退出工具终端
   (hvm)$> exit
   ```

3. 验证hccl_test用例运行结果 [Runner结果查看](#491-runner插件结果) [Checker结果查看](#492-checker插件结果)

#### 3.3.4 AIV模式

1. 环境变量配置。

   ```bash
   # 进入工具安装目录
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
   source /home/workspace/Ascend/cann/set_env.sh
   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
   export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
   export HCCL_OP_EXPANSION_MODE="AIV"
   ```

2. 执行

   ```bash
   # 需要进入到新的bin文件目录下执行hccl-vm
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
   
   # 选择昇腾集群拓扑配置文件，启动工具，初始化集群环境，进入工具命令行
   ./hccl-vm start ascend950_cluster_32_server_normal.yaml
   
   # 如需启用runner插件（可选）
   (hvm)$> hccl-vm plugin install @runner
   
   # 选择本次算子执行的通信域配置文件（在1个超节点1个Server1个NPU的集群环境运行hccl_test用例）
   (hvm)$> hccl-vm mock-comm 112
   (hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   
   # 执行checker校验
   (hvm)$> hccl-vm plugin run @checker
   
   # 退出工具终端
   (hvm)$> exit
   ```

3. 验证hccl_test用例运行结果 [Runner结果查看](#491-runner插件结果) [Checker结果查看](#492-checker插件结果)

### 3.4 Pytorch用例示例

暂不支持。

### 3.5 hccl代码修改验证示例

若您修改了CANN的算子包代码，如新增了算法类型，为保证您的修改生效，需要按照如下步骤操作执行。build_pkg.sh脚本帮助用户执行编包、装包、拷贝Device侧依赖符号，执行前需设置环境变量:

```bash
# 假设您的CANN安装目录为：/home/workspace/Ascend
source /home/workspace/Ascend/cann/set_env.sh
# 配置hccl代码仓路径
export HCCL_CODE_HOME=/home/workspace/hccl
# 配置hcomm代码仓路径
export HCOMM_CODE_HOME=/home/workspace/hcomm
```

1. 若您更新/修改了CANN hccl仓代码，请执行`bash build_pkg.sh --install hccl`。
2. 若您更新/修改了CANN hcomm仓代码，请执行`bash build_pkg.sh --install hcomm`。
3. 若您同时更新/修改了CANN hccl、hcomm仓代码，请执行`bash build_pkg.sh --full`。
4. 参考[使用示例](#33-使用示例)步骤，重新运行用例。

---

## 4. 详细指导

### 4.1 工具环境变量配置

**HCCL-VM环境变量说明**：

| 环境变量                      | 用途                                                                                                        | 示例                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| `HCCL_CODE_HOME`         | HCCL-VM编译指定HCCL源码路径。默认未配置。     | `export HCCL_CODE_HOME=/home/workspace/hccl`                     |
| `HCOMM_CODE_HOME`         | HCCL-VM编译指定HCOMM源码路径。默认未配置。     | `export HCOMM_CODE_HOME=/home/workspace/hcomm`                     |
| `HCCLVM_ENABLE_DUMP_DATA` | 使能Runner插件dump input\&output数据。若使能，则在测试用例执行过程中，会将每个算子的input\&output数据dump到all\_rank\_input\_output.txt文件中 | `export HCCLVM_ENABLE_DUMP_DATA=1 使能，export HCCLVM_ENABLE_DUMP_DATA=0 禁用` |

### 4.2 HCCL-Test用例构建

hccl_test用例源码在CANN包安装目录下，支持OpenMPI和MPICH两种环境编译、运行，运行时差异详见[OpenMPI和MPICH环境用例执行差异](#48-openmpi和mpich环境用例运行差异)，本用例指导中以OpenMPI环境为例。

#### 4.2.1 OpenMPI环境编译

1. 安装OpenMPI

   ```bash
   sudo apt-get update
   sudo apt install openmpi-bin libopenmpi-dev
   ```

2. 编译hccl_test

   ```bash
   # 修改CANN安装目录权限
   chmod -R 755 /home/workspace/Ascend
   
   # 进入hccl_test用例源码目录
   cd /home/workspace/Ascend/cann/tools/hccl_test
   
   # 设置CANN环境变量
   source /home/workspace/Ascend/cann/set_env.sh
   
   # 临时修改Makefile脚本
   if ! grep -q '\-lmpi_cxx' Makefile; then
       sed -i 's/-lmpi/-lmpi -lmpi_cxx/g' Makefile
   fi
   
   # 编译hccl_test用例
   MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH}
   ```

#### 4.2.2 MPICH环境编译

假设mpich的路径为: `/usr/lib/mpich`。

```bash
# 进入hccl_test用例源码目录
cd /home/workspace/Ascend/cann/tools/hccl_test

# 设置CANN环境变量
source /home/workspace/Ascend/cann/set_env.sh

# 配置环境变量
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH

# 编译hccl_test用例
make MPI_HOME=/usr/lib/mpich/ ASCEND_DIR=${ASCEND_HOME_PATH}
```

### 4.3 昇腾集群拓扑配置文件说明

#### 4.3.1 Server/Pod拓扑配置文件说明

昇腾集群拓扑是由一个或多个Server/Pod形态的子拓扑，按照CLOS分层网络规则组合而成的。因此用户在生成集群拓扑前，需要确认每个Server/Pod的拓扑类型。
用户既可以选择HCCL-VM工具提供的预定义拓扑类型，也可以根据配置文件格式要求，自定义Server/Pod的拓扑类型。

描述一个Server/Pod的拓扑网络关系，主要包括以下方面：

 - **端口配置表**: 描述一个NPU卡的物理端口配置信息，如NPU与NPU直连端口（P2P），NPU出框端口（P2NET）等。
 - **链路配置表**: 描述一个Server/Pod内的所有NPU卡之间的连接关系，如全连接（Full Mesh）等。
 - **PortBound**: 描述一个NPU卡的某些端口的绑定关系，即多个端口绑定成一个PortGroup。

```yaml
type: "server_intra_links"
name: "ascend950_links_topo_demo"
description: "ascend950芯片普通拓扑连接关系描述文件"

soc_version: "Ascend950"
device_num: 16

device_ports_allocate_map:
 # port分配表: 0: 不使用, 1: device直连, 2: device连交换机, 3: d2h端口
 #                 portId:  0  1  2  3  4  5  6  7  8
    - {die_id: 0, pin_map: [1, 1, 1, 0, 2, 2, 2, 2, 3]} # die0
    - {die_id: 1, pin_map: [0, 0, 0, 0, 0, 0, 0, 0, 0]} # die1

# port_group: 描述哪些port合并为一个portGroup，相同portGroup的port对于IP地址相同
port_group:
    - {layer: 0, ports: ["0/4", "0/5", "0/6", "0/7"]}

links:
  # ── 描述法：每行8个device组成全互联 ──
  - link_mode: "fullmesh"
    connections:
      # 如下示例表示：die0的device 0, 1, 2, 3都通过die0的port进行全连接
      - {die_id: 0, devices_range: [0, 3]}
      - {die_id: 0, devices_range: [4, 7]}
      - {die_id: 0, devices_range: [8, 11]}
      - {die_id: 0, devices_range: [12, 15]}
  
  #- link_mode: "enum"
  #  device_to_device_links:
  #    如下示例表示：device 0和1都通过die0的port，分别与device 1, 3, 5, 7的die1的port进行连接
  #            即：device0与device1, device3, device5, device7进行连接，device1与device3, device5, device7进行连接
  #    - {src_die_id: 1, src_local_id_range: [0, 2], dst_die_id: 1, dst_local_id_range: [1, 3, 5, 7]}

  - link_mode: "enum"
    device_to_switch_links:
      # 如下示例表示：device0到device15都通过die0的port连接到交换机。结合portGroup可知，device0到device15都通过portGroup[0/4, 0/5, 0/6, 0/7]连接到交换机。
      - {die_id: 0, devices_range: [0, 15]}

```

**字段说明**：

 - **soc_version**: 芯片型号，如 `Ascend950`。
 - **device_num**: 设备总数，根据芯片型号和拓扑类型确定。
 - **device_ports_allocate_map**: 端口分配表，描述每个die的端口配置。1表示device直连端口，2表示device连交换机端口，3表示d2h端口。
 - **port_group**: 描述哪些port合并为一个portGroup，相同portGroup的port对应IP地址相同。没有配置的port，则默认每个port为一个portGroup。
 - **links**: 链路配置表，描述Server/Pod内的所有NPU卡之间，以及NPU与交换机之间的连接关系。
   - **NPU直连关系**: 工具提供了两种方式配置NPU直连关系：
     - **link_mode == "fullmesh"**: 表示所有Device基于一个Die的Port进行全连接。后续有新增典型的连接方式，可以新增link_mode类型，如"ring"。
     - **link_mode == "enum"**: 枚举法。当Server/Pod内的NPU连接方式比较复杂时，可以通过枚举所有的链路关系来描述。
   - **NPU与交换机连接关系**: 用户可以通过枚举法配置NPU与交换机之间的连接关系。
 - **device_to_device_links**: 描述NPU与NPU之间的连接关系。
 - **device_to_switch_links**: 描述NPU与交换机之间的连接关系。

#### 4.3.2 集群拓扑配置文件说明

昇腾集群网络是由一个或多个Server/Pod形态的子拓扑，按照CLOS分层网络规则组合而成的。用户可根据集群规模和需求，选择不同的Server/Pod拓扑类型。

用户可按照如下配置文件格式，自定义集群拓扑配置文件：

```yaml
name: "ascend950_cluster_32_server_normal"
description: "昇腾950 normal组网：32个超级节点，每个超级节点1个服务器"

# 超节点总数量
super_node_num: 4
# sever/pod总数量
server_num: 32
server_list:
  # 0-7 server: 均采用ascend950_server_topo_normal拓扑类型
  - {super_pod_id: 0, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 1, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 2, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 3, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}

```

上述配置文件描述了一个包含4个超节点，32个Server，共128个NPU卡的集群拓扑。其中，每个Server/Pod采用ascend950_server_topo_normal拓扑类型。

**字段说明**：

 - **super_node_num**: 超节点总数量。
 - **server_num**: sever/pod总数量。
 - **server_list**: 每个Server/Pod的配置信息，包括超节点ID、设备ID范围、芯片型号、Server/Pod拓扑配置文件路径。

#### 4.3.3 通信域配置文件说明

用户在昇腾集群环境中，需要根据待执行算子所需的通信域不同，选择不同的通信域配置文件。

工具提供了hccl-vm mock-comm命令读取和配置算子通信域配置文件。通信域配置文件格式为yaml，路径为hccl_vm_install/config/topo_meta。若目录中没有对应的通信域配置文件，则用户需要先创建一个。

hccl-vm工具支持非对称拓扑通信域配置。如下所示：

```yaml
# 1. 全局统计信息 podNum, serNum, rankNum 都小于 1024
meta:
  podNum: 1  # 总的超节点数
  serNum: 2  # 总的server数
  rankNum: 6 # 总的rank数

# 2. 详细拓扑结构 
topology:
  - podId: 0
    servers:
      - serId: 0
        # 每个server实际跑的rank的local id
        ranks: [0, 2]
      - serId: 1
        # 每个server实际跑的rank的local id
        ranks: [1, 3, 5, 7]
```

**注意事项**：

- 配置通信域时，工具会根据指定的通信域配置编号重新生成topo.json和ranktable.json文件。
- 上述通信域配置yaml文件中，ranks字段表示每个server实际跑的rank的local id（即device的物理ID）列表。

#### 4.3.4 topo和ranktable.json文件说明

`topo.json` 和 `ranktable.json` 文件不需要手动创建，工具会根据以下信息自动生成：

- **拓扑配置编号**：用户在启动时指定的编号（如 112、113 等）
- **芯片类型**：根据运行环境自动识别的芯片类型。

虽然配置文件由工具自动生成，但了解其结构有助于理解拓扑配置。

**topo.json 结构**：

`topo.json` 描述了一个 server 内所有设备的连接关系：

```json
{
  "server": {
    "device_count": 8,
    "groups": [
      {
        "group_id": 0,
        "device_start": 0,
        "device_count": 8,
        "topo_layout": "1D"
      }
    ]
  },
  "ports": [
    {
      "ccu": "die0",
      "port_pattern": "0/{0-6}",
      "protocol": "HCCS",
      "func_id": 2,
      "usage": "peer2peer",
      "ip_binding": "independent"
    },
    {
      "ccu": "die0",
      "port_pattern": "0/7,0/8",
      "protocol": "ROCE",
      "func_id": 3,
      "usage": "peer2net",
      "ip_binding": "independent"
    }
  ],
  "links": [
    {
      "net_layer": 0,
      "link_type": "PEER2PEER",
      "topo_type": "1DMESH",
      "ccu": "die0",
      "port_pattern": "0/{0-6}",
      "connect_pattern": "full_mesh",
      "group_id": 0
    },
    {
      "net_layer": 1,
      "link_type": "PEER2NET",
      "topo_type": "CLOS",
      "ccu": "die0",
      "port_pattern": "0/7,0/8",
      "connect_pattern": "all_to_net",
      "group_id": 0
    }
  ]
}
```

**字段说明**：

- `server.device_count`：设备总数。
- `server.groups`：设备分组信息。
- `ports`：端口配置。
  - `usage`：端口用途（`peer2peer` 表示设备间连接，`peer2net` 表示与外部连接）
- `links`：链路配置。
  - `link_type`：链路类型（`PEER2PEER` 或 `PEER2NET`）
  - `topo_type`：拓扑类型（`1DMESH`、`CLOS` 等）

**ranktable.json 结构**：

`ranktable.json` 描述了本次运行使用的设备和 IP 映射：

```json
{
  "version": "1.0",
  "server_count": 1,
  "device_count": 8,
  "server_list": [
    {
      "server_id": 0,
      "device_id": 0,
      "device_ip": "192.168.1.10",
      "port": "2222"
    }
  ]
}
```

**字段说明**：

- `server_count`：服务器数量。
- `device_count`：设备总数。
- `server_list`：服务器和设备列表。
  - `device_ip`：设备 IP 地址。
  - `port`：设备端口号。

### 4.4 hccl\_config.sh文件说明

hccl\_config.sh文件中包含了HCCL\_Test用例运行所需的环境变量配置。其中的环境变量与Hccl\_Test用例在真机环境中的环境变量一致。
用户需要根据自己的用例和需求，修改hccl\_config.sh脚本，配置HCCL用例运行环境变量。

```bash
#!/bin/bash
# hccl_config.sh - HCCL 环境变量配置

remove_files_by_prefix() {
  if [ "$#" -ne 1 ]; then
    echo "Usage: remove_files_by_prefix <prefix>" >&2
    return 2
  fi

  local prefix="$1"
  if [ -z "$prefix" ]; then
    return 0
  fi

  shopt -s nullglob
  local any_deleted=0
  for f in "${prefix}"*; do
    if [ -f "$f" ]; then
      rm -f -- "$f" && any_deleted=1
    fi
  done
  shopt -u nullglob

  # 无论是否删除了文件，均返回 0，确保脚本继续执行
  return 0
}

# 清理 data/ 目录的冗余文件（CCU 模式下生成的临时文件）
cd "${HCCL_VM_INSTALL_DIR}/data" 2>/dev/null && {
  remove_files_by_prefix "sqe_info_rank_"
  remove_files_by_prefix "mc_instr_info_rank_"
  rm -f "all_rank_input_output.txt"
  cd "${HCCL_VM_INSTALL_DIR}"
}

# 设置CANN环境变量
source /home/workspace/Ascend/cann/set_env.sh

# 关闭hccl心跳功能
export HCCL_DFS_CONFIG=cluster_heartbeat:off

# 设置 HCCL-VM 安装路径，基于脚本自身位置推断（兼容 bin/ 与 script/ 子目录）
_INSTALL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$(basename "${_INSTALL_SCRIPT_DIR}")" in
    bin|script)
        export HCCL_VM_INSTALL_DIR="$(dirname "${_INSTALL_SCRIPT_DIR}")"
        ;;
    *)
        export HCCL_VM_INSTALL_DIR="${_INSTALL_SCRIPT_DIR}"
        ;;
esac
unset _INSTALL_SCRIPT_DIR

# 配置LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH

# 设置 ranktable.json文件路径 (与 mock-comm 生成路径保持一致)
export RANK_TABLE_FILE=${HCCL_VM_INSTALL_DIR}/data/ranktable.json

# 设置日志级别
export ASCEND_GLOBAL_LOG_LEVEL=1

# 设置日志打屏输出
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 设置 HCCL 运行模式（CCU、AI_CPU、AIV等展开模式）
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# 或设置 HCCL 运行参数(AI_CPU展开模式) AI_CPU模式环境变量与其他模式不可同时设置
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# 或设置 HCCL 运行参数(AIV展开模式) AIV模式环境变量与其他模式不可同时设置
# export HCCL_OP_EXPANSION_MODE="AIV"

echo "HCCL-VM environment configured successfully!"

```

### 4.5 工具规格约束

**支持算子类型**：

工具支持算子类型包含：allgather/allreduce/alltoall/reduce/reduce\_scatter/scatter/alltoallv.

**支持数据类型**：

HCCL-VM工具支持数据类型包含：int8/int16/int32/fp16/fp32/uint8/uint16/uint32/bfp16/hif8/fp8e4m3/fp8e5m2/fp8e8m0.

HCCL-VM Runner插件支持数据类如下：

| ReduceOp | DataType                           |
| -------- | ---------------------------------- |
| `ADD`    | `int8/int16/int32/uint8` |
| `MIN`    | `int8/int16/int32/uint8` |
| `MAX`    | `int8/int16/int32/uint8` |

**硬件规格**：

目前本工具仅适配支持Ascend950芯片。单server内最大支持8张卡；超过8张卡，需要跨server运行。

### 4.6 HCCL-VM插件功能

#### 4.6.1 Runner插件

Runner插件，即模拟执行Hccl业务编排生成的任务，输出output数据。
模拟运行器插件在hccl\_test用例执行过程中默认**关闭**。hccl\_test用例调用算子接口后，通过aclrtSynchronizeStream接口等待算子任务执行完成。模拟运行器工具会等待所有rank都处于等待状态时，开启模拟执行所有rank的task。执行完成后，通知各个rank的用例进行继续执行。
用例执行完成后，用户可以通过执行目录下的"all\_rank\_input\_output.txt"文件，查看每个rank的input buffer和output buffer数据。此功能默认不开启，用户可以在用例执行前通过对应命令开启。

**安装与卸载**：

Runner插件支持通过 `hccl-vm plugin install/uninstall` 命令进行安装和卸载。需在进入hccl-vm工具命令行后、执行算例前安装runner插件，后续的执行才会运行runner。

```bash
# 安装runner插件
(hvm)$> hccl-vm plugin install @runner

# 卸载runner插件
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker插件

Checker插件，即算法分析器插件：功能是将hccl生成的所有task形成一个DAG图，并且通过分析DAG图，判断是否存在内存冲突；通过模拟执行DAG图，判断是否存在语义错误等问题。
算法分析器插件，是由用户自行通过命令启动执行。

Checker插件正处于新旧交替阶段，Checker V3为原Checker的重构版，主要提高了校验性能，在默认情况下将会运行新Checker（Checker V3）。Checker V3大图校验默认开启：每个sync window内的多个算子会合并成一张大图，执行跨算子同步资源冲突校验。大图校验与老Checker、新Checker的开关相互独立，可以通过修改Checker的`manifest.json`文件中的配置参数进行调整。

```bash

# 配置文件位于 /pathto/hccl_vm_install/plugin/checker/manifest.json

{
  "name": "checker",        // Checker插件名
  "version": "1.0.0",       // Checker插件版本
  "entry": "./checker",     // Checker插件启动指令
  "dependency": {
      "min_core_version": "1.0.0"
  },
  "setting": {              // Checker插件配置项
      "enable_new_checker": true,           // 是否启用新Checker（Checker V3，默认开启）
      "enable_old_checker": false,          // 是否启用老Checker（默认关闭）
      "enable_big_graph_checker": true,     // 是否启用Checker V3大图校验（默认开启，与old/new checker独立）
      "enable_insight_dump": false,         // 是否启用可视化数据输出（默认关闭，仅支持老Checker）
      "enable_memory_snapshot_dump": false  // 是否启用可视化内存快照数据输出（默认关闭，仅支持老Checker，需要先开启可视化数据输出"enable_insight_dump"）
  }
}
```

### 4.7 hccl_rootinfo.json文件

目前工具初始化通信域使用的是ranktable.json文件，因此hccl_rootinfo.json文件的作用仅限于提供topo.json文件的路径。
若/etc路径下没有hccl_rootinfo.json文件，则用户需自行创建该文件，内容如下：

```json
{
  "version": "2.0",
  "topo_file_path": "/home/workspace/hcomm/test/hccl_vm/hccl_vm_install/data/topo.json"
}
```

### 4.8 OpenMPI和MPICH环境用例运行差异

运行hccl_test用例前，用户可以通过which命令判断当前环境使用的是哪个mpirun。

#### 4.8.1 环境变量配置差异

环境中一般默认配置的是OpenMPI，若用户使用OpenMPI运行用例，一般不需要额外配置环境变量。
若用户使用MPICH环境运行用例，需要按照如下方式配置环境变量：

```bash
# 配置mpich环境变量
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.8.2 mpirun命令参数差异

OpenMPI环境，用户按照如下命令运行hccl_test用例：

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**参数说明**：

 - --allow-run-as-root：OpenMPI专属参数，允许以root用户运行MPI进程，用于在没有root权限的环境中运行。
 - --oversubscribe：OpenMPI专属参数，放开CPU槽位(slot)限制，允许单节点启动[进程数>CPU逻辑核数]，也就是超额超配跑进程。
 - -np 2：指定进程数为2，与节点数一致。

MPICH环境，用户按照如下命令运行hccl_test用例：

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**参数说明**：

 - -np 2：指定进程数为2，与节点数一致。

### 4.9 结果查看

#### 4.9.1 Runner插件结果

如果在hccl-vm终端执行hccl-vm plugin install @runner安装插件后，算子流程执行完成后会自动触发runner插件的执行，最终结果依赖hccl_test进行校验，用户需关注在重定向的日志文件中是否存在[error]级别日志和最终校验结果：

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.9.2 Checker插件结果

在hccl-vm终端内执行hccl-vm plugin run @checker后，Checker校验流程及结果会打印在终端内，用于需关注是否存在[error]级别日志和最终校验结果：

大图校验按每个sync window执行一次。大图校验失败不会阻断其他校验流程，需通过日志中的`BigGraphCheckerV3 failed`或`Big graph sync-conflict check failed`等`error`级别日志识别。

```bash
[info][PID:144373][TID:144880][main.cc][RunChecker] [RunChecker] op[0] Checker Success.
```

---

### 4.10 大块内存复用（仅校验模式）

仅校验模式用于大规模集群仅运行 Checker 校验的场景。开启后，单块 200MB 到 4GB 的大内存申请复用同一块 4GB 共享区 `HcclCommPool`，各 rank 共享、允许互相覆盖，以此大幅降低 `/dev/shm` 占用。此时大块内容不保证正确，仅适用于不读取缓冲区数据的 Checker V3 校验链路，需要数值正确的结果时请勿开启。

仅校验模式是会话级开关，在 `start` 子命令后追加 `--check-only` 显式开启；不加时为默认的普通模式，大块走真实独立分配，正确性无损。小于 200MB 的申请始终走真实分配，单块大于 4GB 在仅校验模式下直接报错拒绝。仅校验模式与 Runner 不互斥，但在仅校验模式开启时安装 Runner，大块复用仍会生效、可能覆盖 Runner 数据，工具会打印告警。

```bash

# 启动工具时开启仅校验模式
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only
```

---

## 5 附录

### 开源第三方软件依赖

编译本项目时，依赖的第三方开源软件列表如下，离线编译场景可下载并重命名软件包后放置在本项目内的third_party目录下。

| 开源软件       | 版本          |下载地址 |
| ------------  | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI11         | 2.2.0         | [cli11-2.2.0.tar.gz](https://raw.gitcode.com/src-openeuler/cli11/blobs/58c912141164a5c0f0139bfa91343fefe151d525/cli11-2.2.0.tar.gz) |
| json          | 3.11.3        | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| spdlog        | 1.11.0        | [spdlog-v1.11.0.tar.gz](https://raw.gitcode.com/src-openeuler/spdlog/blobs/c2dfb1aca26c607393665c836155613ff283de66/v1.11.0.tar.gz) |
| yaml-cpp      | 0.8.0         | [yaml-cpp-0.8.0.tar.gz](https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz) |
| sqlite        | 3.51.0        | [sqlite-amalgamation-3510300.zip](https://www.sqlite.org/2026/sqlite-amalgamation-3510300.zip) |
| googletest    | 1.14.0        | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| cann-cmake    | master-044    | [cmake-master-044.tar.gz](https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-044.tar.gz) |

### 术语表

| 术语       | 说明                                                      |
| -------- | ------------------------------------------------------- |
| HCCL     | Huawei Collective Communication Library，华为集合通信库         |
| NPU      | Neural Processing Unit，神经网络处理器                          |
| CANN     | Compute Architecture for Neural Networks，华为昇腾 AI 处理器软件栈 |
| MPI      | Message Passing Interface，消息传递接口                        |
| CCU      | Collective Communication Unit，集合通信单元                    |
| Topology | 拓扑，设备连接关系                                               |
| Rank     | 进程编号，在 MPI 中的标识                                         |

---

**文档版本**：v1.1。
**最后更新**：2026-06-30。

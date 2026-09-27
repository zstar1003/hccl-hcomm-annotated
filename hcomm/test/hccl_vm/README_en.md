# HCCL-VM User Guide

## 1. Overview

HCCL-VM is a virtual execution environment for high-performance collective communication on Huawei Ascend NPU cards. This tool enables HCCL collective communication operator development and functional verification without real Ascend hardware.

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. Prerequisites

|   Dependency   |     Version Requirement     |
| -------- | ------------- |
| System Architecture  | x86_64 Ubuntu22.04 and above |
| Specification Constraints  | Ascend950, others refer to [Constraint Details](#45-tool-specification-constraints) |

### 2.1 CANN Package Installation

Install the latest CANN Toolkit development kit package and CANN ops operator package [Download Link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)

```bash
# Ensure installation packages have executable permissions
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# Installation commands
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```

### 2.2 hccl_test Compilation

hccl_test is the official HCCL performance test tool provided by Ascend. For details, refer to [HCCL Performance Test Tool](https://www.hiascend.com/en/document/redirect/CANNCommunityToolHcclTest). HCCL-VM supports running hccl_test cases in the virtual environment. First follow the [hccl_test Case Build](#42-hccl-test-case-build) section to compile the case binary program.

Note: Optional. PyTorch cases will be supported in the future.

---

## 3. Quick Start

### 3.1 One-Click Installation

One command completes dependency installation, source code retrieval, CANN detection, and compilation (the default uses the `main` profile, corresponding to the `master` branch of hcomm/hccl). The working directory remains consistent with manual installation, using `/home/workspace` (all sample paths below follow this convention):

```bash
# Create and enter the working directory (the script installs to the current directory by default)
mkdir -p /home/workspace && cd /home/workspace
curl -fsSL https://raw.gitcode.com/cann/hcomm/raw/master/test/hccl_vm/hccl_vm_installer | bash
```

You may also download and run locally (for review or offline distribution): `bash hccl_vm_installer`; or specify explicitly with `--workspace`: `... | bash -s -- --workspace /home/workspace`.

**Prerequisites**: x86_64 Linux; the toolchain must meet hcomm build.md requirements: gcc/g++ 7.3.0-13.3.x, cmake >= 3.16.0 (constraints apply to both host and aarch64 cross-compilers). Ubuntu 22.04 / 24.04 work out of the box; newer versions with default gcc (14/15) exceed the supported range. The script issues a warning and continues. Compile in an environment that meets the version range.

**CANN**: The script probes CANN only in the working directory `<workspace>/Ascend` (or the path specified by `--ascend-path`). If CANN is not found, the script downloads and installs the matching version to that location. Behavior is identical for root and regular users. `--offline` only detects and never downloads. On internal networks without public internet access, the script falls back to printing a self-service CANN preparation guide.

**hccl_test**: The script compiles OpenMPI and the hccl_test performance test tool by default. Use `--skip-hccl-test` to disable this.

**Common Parameters**:

- `--profile <name>`: configuration profile (default `main`, use `--list-profiles` to list all)
- `--workspace <path>`: working directory for source code, compilation, and artifacts (default is the current directory)
- `--ascend-path <path>`: specify the CANN directory; reuse if present, install there if absent
- `--reinstall-cann`: re-download and overwrite existing CANN (use when versions do not match; default preserves existing)
- `--offline`: use only existing CANN, never download
- `--skip-hccl-test`: skip hccl_test compilation
- `-h`: full help

After completion, the tool resides at `/home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin/hccl-vm`. Delete the working directory to clean up tool artifacts (use `apt remove` manually to uninstall system dependencies installed by apt). This tool does not modify CANN. (If you specified a different directory with `--workspace`, replace `/home/workspace` in the examples below accordingly.)

> The one-click installation automatically completes `build.sh` compilation and `build_pkg.sh` sub-package installation (including device-side symbols required by AICPU/AIV). After installation, all CCU/AICPU/AIV modes in [Usage Examples](#33-usage-examples) run directly. No separate `build_pkg.sh` execution is needed.

### 3.2 Manual Build & Installation

```bash
# 1. Create the working directory
mkdir -p /home/workspace
cd /home/workspace

# 2. Download dependency source code
git clone https://gitcode.com/cann/hccl.git
git clone https://gitcode.com/cann/hcomm.git

# 3. Install third-party dependencies
sudo apt-get update
sudo apt install build-essential cmake libsqlite3-dev libboost-all-dev rdma-core libibverbs-dev pkg-config gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user-static binfmt-support

# 4. Compile the HCCL-VM tool. After downloading hcomm source code, the tool source path is: /home/workspace/hcomm/test/hccl_vm
cd /home/workspace/hcomm/test/hccl_vm
source /home/workspace/Ascend/cann/set_env.sh
export HCCL_CODE_HOME=/home/workspace/hccl
export HCOMM_CODE_HOME=/home/workspace/hcomm
bash ./build.sh --full

# 5. Copy and extract aicpu_hcxx.tar.gz from the CANN installation directory
bash build_pkg.sh
```

### 3.3 Usage Examples

#### 3.3.1 Environment Configuration

Refer to [hccl_rootinfo File Content](#47-hccl_rootinfojson-file) to create and configure the hccl_rootinfo.json file.

#### 3.3.2 CCU Mode

1. Configure environment variables.

   ```bash
   # Enter the tool installation directory
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
   source /home/workspace/Ascend/cann/set_env.sh
   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
   export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
   export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
   ```

2. Execute.

   ```bash
   # Enter the new bin directory to execute hccl-vm
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
   
   # Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
   ./hccl-vm start ascend950_cluster_32_server_normal.yaml
   
   # Enable the runner plugin if needed (optional)
   (hvm)$> hccl-vm plugin install @runner
   
   # Select the communication domain configuration file for this operator execution (run hccl_test cases in a cluster environment with 1 supernode, 1 server, and 1 NPU)
   (hvm)$> hccl-vm mock-comm 112
   (hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   
   # Execute checker verification
   (hvm)$> hccl-vm plugin run @checker
   
   # Exit the tool terminal
   (hvm)$> exit
   ```

3. Verify hccl_test case execution results.

   [View Runner Results](#491-runner-plugin-results) 
   [View Checker Results](#492-checker-plugin-results)

#### 3.3.3 AICPU Mode

The AICPU expansion mode executes algorithm expansion steps on the device side. Therefore the hccl-vm tool compiles and simulates HCCL device-side symbols. Device-side symbols use the ARM architecture. Compilation on x86 environments requires a cross-compiler. Execution requires QEMU to simulate AICPU mode.

1. Configure environment variables.

   ```bash
   # Enter the tool installation directory
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
   source /home/workspace/Ascend/cann/set_env.sh
   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
   export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
   export HCCL_OP_EXPANSION_MODE="AI_CPU"
   ```

2. Execute

   ```bash
   # Enter the new bin directory to execute hccl-vm
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
   
   # Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
   ./hccl-vm start ascend950_cluster_32_server_normal.yaml
   
   # Enable the runner plugin if needed (optional)
   (hvm)$> hccl-vm plugin install @runner
   
   # Select the communication domain configuration file for this operator execution (run hccl_test cases in a cluster environment with 1 supernode, 1 server, and 1 NPU)
   (hvm)$> hccl-vm mock-comm 112
   (hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   
   # Execute checker verification
   (hvm)$> hccl-vm plugin run @checker
   
   # Exit the tool terminal
   (hvm)$> exit
   ```

3. Verify hccl_test case execution results [View Runner Results](#491-runner-plugin-results) [View Checker Results](#492-checker-plugin-results)

#### 3.3.4 AIV Mode

1. Configure environment variables.

   ```bash
   # Enter the tool installation directory
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
   source /home/workspace/Ascend/cann/set_env.sh
   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
   export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
   export HCCL_OP_EXPANSION_MODE="AIV"
   ```

2. Execute

   ```bash
   # Enter the new bin directory to execute hccl-vm
   cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin
   
   # Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
   ./hccl-vm start ascend950_cluster_32_server_normal.yaml
   
   # Enable the runner plugin if needed (optional)
   (hvm)$> hccl-vm plugin install @runner
   
   # Select the communication domain configuration file for this operator execution (run hccl_test cases in a cluster environment with 1 supernode, 1 server, and 1 NPU)
   (hvm)$> hccl-vm mock-comm 112
   (hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt
   
   # Execute checker verification
   (hvm)$> hccl-vm plugin run @checker
   
   # Exit the tool terminal
   (hvm)$> exit
   ```

3. Verify hccl_test case execution results [View Runner Results](#491-runner-plugin-results) [View Checker Results](#492-checker-plugin-results)

### 3.4 PyTorch Case Example

Not supported yet.

### 3.5 HCCL Code Modification Verification Example

If you modified CANN operator package code, such as adding a new algorithm type, follow these steps to ensure your changes take effect. The build_pkg.sh script helps users perform package building, package installation, and device-side dependency symbol copying. Set environment variables before execution:

```bash
# Assume your CANN installation directory is: /home/workspace/Ascend
source /home/workspace/Ascend/cann/set_env.sh
# Configure the hccl code repository path
export HCCL_CODE_HOME=/home/workspace/hccl
# Configure the hcomm code repository path
export HCOMM_CODE_HOME=/home/workspace/hcomm
```

1. If you updated or modified the CANN hccl repository code, execute `bash build_pkg.sh --install hccl`.
2. If you updated or modified the CANN hcomm repository code, execute `bash build_pkg.sh --install hcomm`.
3. If you updated or modified both CANN hccl and hcomm repository code, execute `bash build_pkg.sh --full`.
4. Refer to the [Usage Examples](#33-usage-examples) steps and re-run the cases.

---

## 4. Detailed Guide

### 4.1 Tool Environment Variable Configuration

**HCCL-VM Environment Variable Description**:

| Environment Variable                      | Purpose                                                                                                        | Sample                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| `HCCL_CODE_HOME`         | Specifies the HCCL source code path for HCCL-VM compilation. Not configured by default.     | `export HCCL_CODE_HOME=/home/workspace/hccl`                     |
| `HCOMM_CODE_HOME`         | Specifies the HCOMM source code path for HCCL-VM compilation. Not configured by default.     | `export HCOMM_CODE_HOME=/home/workspace/hcomm`                     |
| `HCCLVM_ENABLE_DUMP_DATA` | Enables the Runner plugin to dump input\&output data. When enabled, the tool dumps input\&output data of each operator to the all\_rank\_input\_output.txt file during test case execution. | `export HCCLVM_ENABLE_DUMP_DATA=1 to enable, export HCCLVM_ENABLE_DUMP_DATA=0 to disable` |

### 4.2 HCCL-Test Case Build

The hccl_test case source code resides in the CANN package installation directory. The tool supports compilation and execution in both OpenMPI and MPICH environments. For runtime differences, refer to [OpenMPI and MPICH Environment Case Execution Differences](#48-openmpi-and-mpich-environment-case-execution-differences). This guide uses the OpenMPI environment as a sample.

#### 4.2.1 OpenMPI Environment Compilation

1. Install OpenMPI

   ```bash
   sudo apt-get update
   sudo apt install openmpi-bin libopenmpi-dev
   ```

2. Compile hccl_test

   ```bash
   # Modify CANN installation directory permissions
   chmod -R 755 /home/workspace/Ascend
   
   # Enter the hccl_test case source directory
   cd /home/workspace/Ascend/cann/tools/hccl_test
   
   # Set CANN environment variables
   source /home/workspace/Ascend/cann/set_env.sh
   
   # Temporarily modify the Makefile script
   if ! grep -q '\-lmpi_cxx' Makefile; then
       sed -i 's/-lmpi/-lmpi -lmpi_cxx/g' Makefile
   fi
   
   # Compile hccl_test cases
   MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH}
   ```

#### 4.2.2 MPICH Environment Compilation

Assume the mpich path is: `/usr/lib/mpich`.

```bash
# Enter the hccl_test case source directory
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Configure environment variables
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH

# Compile hccl_test cases
make MPI_HOME=/usr/lib/mpich/ ASCEND_DIR=${ASCEND_HOME_PATH}
```

### 4.3 Ascend Cluster Topology Configuration File Description

#### 4.3.1 Server/Pod Topology Configuration File Description

An Ascend cluster topology consists of one or more Server/Pod sub-topologies combined according to CLOS hierarchical network rules. Therefore the user must confirm the topology type of each Server/Pod before generating the cluster topology.
The user may select predefined topology types provided by the HCCL-VM tool, or define custom Server/Pod topology types according to the configuration file format requirements.

Describing the topology network relationships of a Server/Pod includes the following aspects:

 - **Port Configuration Table**: Describes the physical port configuration of an NPU card, such as NPU-to-NPU direct connection ports (P2P) and NPU outbound ports (P2NET).
 - **Link Configuration Table**: Describes the connection relationships among all NPU cards within a Server/Pod, such as full mesh connections.
 - **PortBound**: Describes port binding relationships of an NPU card, where multiple ports bind into a single PortGroup.

```yaml
type: "server_intra_links"
name: "ascend950_links_topo_demo"
description: "ascend950 chip normal topology connection relationship description file"

soc_version: "Ascend950"
device_num: 16

device_ports_allocate_map:
  # port allocation table: 0: unused, 1: device direct connection, 2: device to switch, 3: d2h port
  #                 portId:  0  1  2  3  4  5  6  7  8
    - {die_id: 0, pin_map: [1, 1, 1, 0, 2, 2, 2, 2, 3]} # die0
    - {die_id: 1, pin_map: [0, 0, 0, 0, 0, 0, 0, 0, 0]} # die1

# port_group: describes which ports merge into one portGroup. Ports in the same portGroup share the same IP address.
port_group:
    - {layer: 0, ports: ["0/4", "0/5", "0/6", "0/7"]}

links:
  # ── Description method: each row of 8 devices forms a full mesh ──
  - link_mode: "fullmesh"
    connections:
      # The following sample indicates: devices 0, 1, 2, 3 of die0 all connect through die0 ports in full mesh
      - {die_id: 0, devices_range: [0, 3]}
      - {die_id: 0, devices_range: [4, 7]}
      - {die_id: 0, devices_range: [8, 11]}
      - {die_id: 0, devices_range: [12, 15]}
  
  #- link_mode: "enum"
  #  device_to_device_links:
  #    The following sample indicates: devices 0 and 1 both use die0 ports to connect to devices 1, 3, 5, 7 on die1 ports respectively
  #            That is: device0 connects to device1, device3, device5, device7; device1 connects to device3, device5, device7
  #    - {src_die_id: 1, src_local_id_range: [0, 2], dst_die_id: 1, dst_local_id_range: [1, 3, 5, 7]}

  - link_mode: "enum"
    device_to_switch_links:
      # The following sample indicates: device0 through device15 all connect to the switch through die0 ports. Combined with portGroup, device0 through device15 all connect to the switch through portGroup[0/4, 0/5, 0/6, 0/7].
      - {die_id: 0, devices_range: [0, 15]}

```

**Field Description**:

 - **soc_version**: Chip model, for example `Ascend950`.
 - **device_num**: Total number of devices, determined by chip model and topology type.
 - **device_ports_allocate_map**: Port allocation table, describes the port configuration of each die. 1 indicates device direct connection ports, 2 indicates device-to-switch ports, 3 indicates d2h ports.
 - **port_group**: Describes which ports merge into one portGroup. Ports in the same portGroup share the same IP address. Unconfigured ports default to one portGroup per port.
 - **links**: Link configuration table, describes connection relationships among all NPU cards within a Server/Pod, and between NPUs and switches.
   - **NPU Direct Connections**: The tool provides two methods to configure NPU direct connections:
     - **link_mode == "fullmesh"**: All devices perform full mesh connections based on ports of one die. New typical connection methods can add new link_mode types in the future, for example "ring".
     - **link_mode == "enum"**: Enumeration method. When NPU connections within a Server/Pod are complex, enumerate all link relationships to describe them.
   - **NPU-to-Switch Connections**: The user configures NPU-to-switch connection relationships using the enumeration method.
 - **device_to_device_links**: Describes NPU-to-NPU connection relationships.
 - **device_to_switch_links**: Describes NPU-to-switch connection relationships.

#### 4.3.2 Cluster Topology Configuration File Description

An Ascend cluster network consists of one or more Server/Pod sub-topologies combined according to CLOS hierarchical network rules. The user selects different Server/Pod topology types based on cluster scale and requirements.

The user defines a custom cluster topology configuration file using the following format:

```yaml
name: "ascend950_cluster_32_server_normal"
description: "Ascend 950 normal network: 32 supernodes, each supernode has 1 server"

# Total number of supernodes
super_node_num: 4
# Total number of servers/pods
server_num: 32
server_list:
  # 0-7 servers: all use the ascend950_server_topo_normal topology type
  - {super_pod_id: 0, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 1, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 2, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 3, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}

```

The configuration file above describes a cluster topology with 4 supernodes, 32 servers, and a total of 128 NPU cards. Each Server/Pod uses the ascend950_server_topo_normal topology type.

**Field Description**:

 - **super_node_num**: Total number of supernodes.
 - **server_num**: Total number of servers/pods.
 - **server_list**: Configuration information for each Server/Pod, including supernode ID, device ID range, chip model, and Server/Pod topology configuration file path.

#### 4.3.3 Communication Domain Configuration File Description

In an Ascend cluster environment, the user selects different communication domain configuration files based on the communication domain required by the operator to execute.

The tool provides the hccl-vm mock-comm command to read and configure operator communication domain configuration files. The communication domain configuration file uses yaml format, and the path is hccl_vm_install/config/topo_meta. If the directory does not contain the corresponding communication domain configuration file, the user must create one first.

The hccl-vm tool supports asymmetric topology communication domain configuration. As shown below:

```yaml
# 1. Global statistics: podNum, serNum, rankNum are all less than 1024
meta:
  podNum: 1  # Total number of supernodes
  serNum: 2  # Total number of servers
  rankNum: 6 # Total number of ranks

# 2. Detailed topology structure
topology:
  - podId: 0
    servers:
      - serId: 0
        # The local IDs of ranks actually running on each server
        ranks: [0, 2]
      - serId: 1
        # The local IDs of ranks actually running on each server
        ranks: [1, 3, 5, 7]
```

**Precautions**:

- When configuring a communication domain, the tool regenerates topo.json and ranktable.json files based on the specified communication domain configuration number.
- In the communication domain configuration yaml file above, the ranks field represents the list of local IDs (that is, physical device IDs) of ranks actually running on each server.

#### 4.3.4 topo.json and ranktable.json File Description

The `topo.json` and `ranktable.json` files do not require manual creation. The tool generates them automatically based on the following information:

- **Topology Configuration Number**: The number specified by the user at startup (for example 112, 113, and so on)
- **Chip Type**: The chip type automatically identified from the runtime environment.

Although the tool generates configuration files automatically, understanding their structure helps the user comprehend topology configuration.

**topo.json Structure**:

`topo.json` describes the connection relationships of all devices within a server:

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

**Field Description**:

- `server.device_count`: Total number of devices.
- `server.groups`: Device grouping information.
- `ports`: Port configuration.
  - `usage`: Port purpose (`peer2peer` indicates device-to-device connections, `peer2net` indicates external connections)
- `links`: Link configuration.
  - `link_type`: Link type (`PEER2PEER` or `PEER2NET`)
  - `topo_type`: Topology type (`1DMESH`, `CLOS`, and so on)

**ranktable.json Structure**:

`ranktable.json` describes the devices and IP mappings used for this run:

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

**Field Description**:

- `server_count`: Number of servers.
- `device_count`: Total number of devices.
- `server_list`: Server and device list.
  - `device_ip`: Device IP address.
  - `port`: Device port number.

### 4.4 hccl\_config.sh File Description

The hccl\_config.sh file contains environment variable configurations required for HCCL\_Test case execution. The environment variables in this file match those used for HCCL\_Test cases in real hardware environments.
The user modifies the hccl\_config.sh script to configure HCCL case runtime environment variables according to case requirements.

```bash
#!/bin/bash
# hccl_config.sh - HCCL environment variable configuration

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

  # Return 0 regardless of whether files were deleted, to ensure the script continues execution
  return 0
}

# Clean redundant files in the data/ directory (temporary files generated in CCU mode)
cd "${HCCL_VM_INSTALL_DIR}/data" 2>/dev/null && {
  remove_files_by_prefix "sqe_info_rank_"
  remove_files_by_prefix "mc_instr_info_rank_"
  rm -f "all_rank_input_output.txt"
  cd "${HCCL_VM_INSTALL_DIR}"
}

# Set CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Disable the hccl heartbeat feature
export HCCL_DFS_CONFIG=cluster_heartbeat:off

# Set the HCCL-VM installation path, inferred from the script location (compatible with bin/ and script/ subdirectories)
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

# Configure LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH

# Set the ranktable.json file path (consistent with the mock-comm generation path)
export RANK_TABLE_FILE=${HCCL_VM_INSTALL_DIR}/data/ranktable.json

# Set the log level
export ASCEND_GLOBAL_LOG_LEVEL=1

# Set log output to screen
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# Set the HCCL runtime mode (CCU, AI_CPU, AIV expansion modes)
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# Or set HCCL runtime parameters (AI_CPU expansion mode). The AI_CPU mode environment variable cannot be set simultaneously with other modes
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# Or set HCCL runtime parameters (AIV expansion mode). The AIV mode environment variable cannot be set simultaneously with other modes
# export HCCL_OP_EXPANSION_MODE="AIV"

echo "HCCL-VM environment configured successfully!"

```

### 4.5 Tool Specification Constraints

**Supported Operator Types**:

The tool supports the following operator types: allgather/allreduce/alltoall/reduce/reduce\_scatter/scatter/alltoallv.

**Supported Data Types**:

The HCCL-VM tool supports the following data types: int8/int16/int32/fp16/fp32/uint8/uint16/uint32/bfp16/hif8/fp8e4m3/fp8e5m2/fp8e8m0.

The HCCL-VM Runner plugin supports the following data types:

| ReduceOp | DataType                           |
| -------- | ---------------------------------- |
| `ADD`    | `int8/int16/int32/uint8` |
| `MIN`    | `int8/int16/int32/uint8` |
| `MAX`    | `int8/int16/int32/uint8` |

**Hardware Specifications**:

This tool currently supports only the Ascend950 chip. A single server supports a maximum of 8 cards. More than 8 cards require cross-server execution.

### 4.6 HCCL-VM Plugin Features

#### 4.6.1 Runner Plugin

The Runner plugin simulates execution of tasks generated by HCCL business orchestration and outputs output data.
The simulator plugin is **disabled** by default during hccl\_test case execution. After the hccl\_test case calls the operator interface, it waits for operator task completion through the aclrtSynchronizeStream interface. The simulator tool waits until all ranks reach the waiting state, then starts simulated execution of tasks for all ranks. After execution completes, the tool notifies each rank to continue case execution.
After case execution completes, the user views the input buffer and output buffer data of each rank through the "all\_rank\_input\_output.txt" file in the execution directory. This feature is disabled by default. The user enables it with the corresponding command before case execution.

**Installation and Uninstallation**:

The Runner plugin supports installation and uninstallation through the `hccl-vm plugin install/uninstall` command. Install the runner plugin after entering the hccl-vm tool command line and before executing operator cases. Subsequent execution then runs the runner.

```bash
# Install the runner plugin
(hvm)$> hccl-vm plugin install @runner

# Uninstall the runner plugin
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker Plugin

The Checker plugin (algorithm analyzer plugin) forms a DAG from all tasks generated by hccl, analyzes the DAG for memory conflicts, and detects semantic errors through DAG simulation.
The user launches the algorithm analyzer plugin manually through commands.

The Checker plugin is in a transition phase between old and new versions. Checker V3 is a refactored version of the original Checker with improved verification performance. The new Checker (Checker V3) runs by default. Checker V3 big graph verification is enabled by default: multiple operators within each sync window merge into one big graph for cross-operator synchronous resource conflict verification. The big graph verification switch operates independently from the old Checker and new Checker switches. Adjust these through configuration parameters in the Checker `manifest.json` file.

```bash

# The configuration file is located at /pathto/hccl_vm_install/plugin/checker/manifest.json

{
  "name": "checker",        // Checker plugin name
  "version": "1.0.0",       // Checker plugin version
  "entry": "./checker",     // Checker plugin startup command
  "dependency": {
      "min_core_version": "1.0.0"
  },
  "setting": {              // Checker plugin configuration options
      "enable_new_checker": true,           // Enable the new Checker (Checker V3, enabled by default)
      "enable_old_checker": false,          // Enable the old Checker (disabled by default)
      "enable_big_graph_checker": true,     // Enable Checker V3 big graph verification (enabled by default, independent of old/new checker)
      "enable_insight_dump": false,         // Enable visualization data output (disabled by default, supports old Checker only)
      "enable_memory_snapshot_dump": false  // Enable visualization memory snapshot data output (disabled by default, supports old Checker only, requires "enable_insight_dump" to be enabled first)
  }
}
```

### 4.7 hccl_rootinfo.json File

The tool currently uses the ranktable.json file to initialize the communication domain. Therefore the hccl_rootinfo.json file serves only to provide the topo.json file path.
If the hccl_rootinfo.json file does not exist under the /etc path, the user creates the file with the following content:

```json
{
  "version": "2.0",
  "topo_file_path": "/home/workspace/hcomm/test/hccl_vm/hccl_vm_install/data/topo.json"
}
```

### 4.8 OpenMPI and MPICH Environment Case Execution Differences

Before running hccl_test cases, the user determines which mpirun the current environment uses through the which command.

#### 4.8.1 Environment Variable Configuration Differences

The environment typically configures OpenMPI by default. If the user runs cases with OpenMPI, no additional environment variable configuration is generally needed.
If the user runs cases with the MPICH environment, configure environment variables as follows:

```bash
# Configure mpich environment variables
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.8.2 mpirun Command Parameter Differences

In the OpenMPI environment, the user runs hccl_test cases with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter Description**:

 - --allow-run-as-root: An OpenMPI-specific parameter that allows running MPI processes as the root user, for use in environments without root permissions.
 - --oversubscribe: An OpenMPI-specific parameter that removes CPU slot limits, allowing a single node to launch [process count > CPU logical core count], that is, oversubscribed process execution.
 - -np 2: Specifies 2 processes, consistent with the node count.

In the MPICH environment, the user runs hccl_test cases with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter Description**:

 - -np 2: Specifies 2 processes, consistent with the node count.

### 4.9 Viewing Results

#### 4.9.1 Runner Plugin Results

If the user installs the plugin by executing hccl-vm plugin install @runner in the hccl-vm terminal, the runner plugin executes automatically after operator flow execution completes. The final result depends on hccl_test verification. The user checks the redirected log file for [error] level logs and the final verification result:

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.9.2 Checker Plugin Results

After the user executes hccl-vm plugin run @checker in the hccl-vm terminal, the Checker verification flow and results print in the terminal. The user checks for [error] level logs and the final verification result:

Big graph verification executes once per sync window. Big graph verification failure does not block other verification flows. Identify failures through `error` level logs such as `BigGraphCheckerV3 failed` or `Big graph sync-conflict check failed`.

```bash
[info][PID:144373][TID:144880][main.cc][RunChecker] [RunChecker] op[0] Checker Success.
```

---

### 4.10 Large Memory Block Reuse (Check-Only Mode)

Check-only mode applies to large-scale cluster scenarios that run only Checker verification. When enabled, large memory allocations from 200MB to 4GB reuse a single 4GB shared region `HcclCommPool`. All ranks share this region and may overwrite each other. This significantly reduces `/dev/shm` usage. Large block content is not guaranteed to be correct in this mode. This mode applies only to the Checker V3 verification path that does not read buffer data. Do not enable this mode when numerically correct results are required.

Check-only mode is a session-level switch. Append `--check-only` after the `start` subcommand to enable it explicitly. Without this flag, the default normal mode applies, and large blocks use real independent allocation with no correctness loss. Allocations smaller than 200MB always use real allocation. Single blocks larger than 4GB are rejected with an error in check-only mode. Check-only mode and Runner are not mutually exclusive. However, when check-only mode is enabled and Runner is installed, large block reuse still takes effect and may overwrite Runner data. The tool prints a warning in this case.

```bash

# Enable check-only mode when starting the tool
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only
```

---

## 5 Appendix

### Open-Source Third-Party Software Dependencies

The following third-party open-source software is required when compiling this project. For offline compilation scenarios, download and rename the software packages, then place them in the third_party directory within this project.

| Open-Source Software       | Version          | Download URL |
| ------------  | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI11         | 2.2.0         | [cli11-2.2.0.tar.gz](https://raw.gitcode.com/src-openeuler/cli11/blobs/58c912141164a5c0f0139bfa91343fefe151d525/cli11-2.2.0.tar.gz) |
| json          | 3.11.3        | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| spdlog        | 1.11.0        | [spdlog-v1.11.0.tar.gz](https://raw.gitcode.com/src-openeuler/spdlog/blobs/c2dfb1aca26c607393665c836155613ff283de66/v1.11.0.tar.gz) |
| yaml-cpp      | 0.8.0         | [yaml-cpp-0.8.0.tar.gz](https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz) |
| sqlite        | 3.51.0        | [sqlite-amalgamation-3510300.zip](https://www.sqlite.org/2026/sqlite-amalgamation-3510300.zip) |
| googletest    | 1.14.0        | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| cann-cmake    | master-044    | [cmake-master-044.tar.gz](https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-044.tar.gz) |

### Glossary

| Term       | Description                                                      |
| -------- | ------------------------------------------------------- |
| HCCL     | Huawei Collective Communication Library         |
| NPU      | Neural Processing Unit                          |
| CANN     | Compute Architecture for Neural Networks, Huawei Ascend AI processor software stack |
| MPI      | Message Passing Interface                        |
| CCU      | Collective Communication Unit                    |
| Topology | Device connection relationships                                               |
| Rank     | Process identifier in MPI                                         |

---

**Document Version**: v1.1.
**Last Updated**: 2026-06-30.

# Communicator Management - One NPU Device per Process (Initialize Communicator Based on Root Node Information)

## Sample Overview

This sample demonstrates how to use the `HcclCommInitRootInfoConfig()` API to initialize a communicator based on root node information. It covers the following features:

- Use rank0 as the root node and generate rootinfo identification information through the `HcclGetRootInfo()` API.

  > The rootinfo identification information mainly includes the Device IP, Device ID, and so on. This information must be broadcast to all ranks in the cluster to initialize the communicator.

- Launch multiple processes through MPI and broadcast the rootinfo identification information to all processes.
- Initialize the communicator in each process based on the rootinfo identification information using the `HcclCommInitRootInfoConfig()` API.
- Call the `HcclAllReduce()` operator and print the results.

## Directory Structure

```text
├── main.cc                   # Sample source file
├── Makefile                  # Build or configuration file
└── one_device_per_process    # Compiled executable
```

## Environment Preparation

### Requirements

This sample supports the following products with a single-node 2-card cluster topology:

- Ascend 950PR/Ascend 950DT

This sample supports the following products with a single-node 8-card cluster topology:

- Atlas A3 training products/Atlas A3 inference products
- Atlas A2 training products
- Atlas training products
- Atlas inference products

### Software Dependencies

Running this sample requires the CANN ops operator package to be installed. For detailed installation steps, see the "Install the CANN Software Package" section in [Source Code Build](../../../docs/en/build/build.md).

### Installing MPI

This sample depends on MPI to launch processes on each Device. Before running this sample, install MPI. For detailed installation steps, see the "MPI Installation and Configuration" section in the corresponding version of the [HCCL Performance Test Tool User Guide][1].

[1]: https://hiascend.com/document/redirect/CannCommunityToolHcclTest

### Disabling Signature Verification

The `cann-hcomm_<version>_linux-<arch>.run` software package generated from this source repository contains the following tar.gz subpackages:

  - `cann-hcomm-compat.tar.gz`: HCOMM compatibility upgrade package.
  - `cann-hccd-compat.tar.gz`: DataFlow compatibility upgrade package.
  - `aicpu_hcomm.tar.gz`: AI CPU communication base package.

These tar.gz packages are loaded to the Device when the service starts. During the loading process, the driver performs security signature verification by default to ensure the package is trusted. Because the tar.gz packages compiled from this source repository do not contain a signature header, the driver security signature verification mechanism needs to be disabled. For the method to disable signature verification, refer to the "Disable Signature Verification" section in [Source Code Build](../../../docs/en/build/build.md).

### Configuring Environment Variables

```bash
# Set CANN environment variables, using the root user default installation path as an example
source /usr/local/Ascend/cann/set_env.sh
# Set the MPI installation directory. Adjust it based on the actual situation.
export MPI_HOME=/usr/local/mpich
```

## Compiling and Running the Sample

Execute the following commands in the sample code directory:

```bash
make
make test N=${RANK_SIZE}
```

`RANK_SIZE` is the number of cluster devices. For Ascend 950PR/Ascend 950DT, `RANK_SIZE` is 2. For other product series, it is 8.

> Note: You can set the `HCCL_OP_EXPANSION_MODE` environment variable to configure the expansion mode of communication operators. For the range supported by different product models, refer to the usage of this environment variable in the [Environment Variable List](https://hiascend.com/document/redirect/CannCommunityEnvRef).
>
> ```bash
> # Set the expansion mode of communication operators to AI CPU communication engine
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## Sample Output

The data for each rank is initialized to 0 to 7. After the AllReduce operation, the result for each rank is the sum of the data at the corresponding positions across all ranks (the data from 8 ranks is summed).

```text
Found 8 NPU device(s) available
rankId: 0, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 1, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 2, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 3, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 4, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 5, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 6, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 7, output: [ 0 8 16 24 32 40 48 56 ]
```

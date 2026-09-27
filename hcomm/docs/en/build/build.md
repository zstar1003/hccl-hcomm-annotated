# Source Code Build

## Environment Preparation

This project supports source code building. Before compiling and running, complete the basic environment setup and source code download by following the steps below. Ensure that the NPU driver, firmware, and CANN software are installed.

### Prerequisites

The software dependencies required for compiling this project are listed below. Ensure that the version requirements are met.

- python >= 3.7.0
- pip3 >= 20.3.0
- setuptools >= 45.0.0
- wheel >= 0.34.0
- gcc and g++: 7.3.0 to 14.2.x
- cmake >= 3.16.0
- pkg-config >= 0.29.1 (for compiling rdma-core)
- patch >= 2.7.0 (for applying patch files)
- ccache (optional, for improving incremental compilation speed)
- lcov (optional, for generating UT or ST coverage reports)

### Installing the CANN Software Package

1. **Install the driver and firmware (runtime dependency)**

    For downloading and installing the driver and firmware, refer to the "Prepare Software Package" and "Install NPU Driver and Firmware" sections in the [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard). The driver and firmware are runtime dependencies. If you are only compiling the source code of this project, they do not need to be installed.

2. **Install the CANN software package**

    - **Scenario 1: Experience or develop based on the master version**

        Click the [download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/), select the latest version, and download the corresponding package based on the product model and environment architecture. The installation commands are as follows. For more guidance, refer to the [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard).

        1. Install the CANN Toolkit development kit package.

            ```bash
            # Ensure the installation package has executable permissions
            chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
            # Installation command
           ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
           ```

        2. Install the CANN ops operator package (runtime dependency).

            The ops operator package is a runtime dependency. If you are only compiling the source code of this project, this package does not need to be installed.

            ```bash
            # Ensure the installation package has executable permissions
            chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
            # Installation command
            ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
            ```

        - \$\{cann\_version\}: Indicates the CANN software package version number.
        - \$\{arch\}: Indicates the CPU architecture, for example, aarch64 or x86_64.
        - \$\{soc\_name\}: Indicates the NPU model name.
        - \$\{install\_path\}: Indicates the specified installation path. The CANN ops operator package must be installed in the same path as the CANN Toolkit development kit package. The default installation path for the root user is `/usr/local/Ascend`.

    - **Scenario 2: Experience or develop based on a released version**

        Visit the [CANN official download center](https://www.hiascend.com/cann/download), select a released version (only CANN 8.5.0 and later versions are supported), and download the corresponding package based on the product model and environment architecture. Finally, follow the commands provided on the webpage to complete the installation.

### Environment Verification

After installing the CANN software package, verify that the environment is functioning correctly.

- **Check the NPU device**:

    ```bash
    # Run npu-smi. If device information is displayed normally, the driver is working correctly.
    npu-smi info
    ```

- **Check the CANN software**:

   ```bash
    # View the version information provided by the version field of the CANN Toolkit development kit package (default installation path). <arch> indicates the CPU architecture (aarch64 or x86_64).
    cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
    # View the version information provided by the version field of the CANN ops operator package (default installation path).
    cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
   ```

### Environment Variable Configuration

Select the appropriate command to apply the environment variables.

```bash
# Default installation path, using the root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Specified installation path
# source ${install_path}/cann/set_env.sh
```

## Compilation and Installation

### Downloading the Source Code

Use the following command to download the source code. Replace \$\{tag\_version\} with the target branch tag name. For the mapping between source branch tags and CANN versions, refer to the [release repository](https://gitcode.com/cann/release-management).

```shell
# Download the source code for the corresponding project branch
git clone -b ${tag_version} https://gitcode.com/cann/hcomm.git
```

### Compiling the Source Code

This project provides a one-click build capability. Navigate to the root directory of the repository and execute the following commands:

```shell
# Compile the host package
bash build.sh --pkg
# Compile the host and device package
bash build.sh --pkg --full
```

During compilation, the dependency packages listed in [Open Source Third-Party Software Dependencies](#open-source-third-party-software-dependencies) are automatically downloaded. If the compilation environment does not have network access, download the required dependency packages in a networked environment, manually upload them to the compilation environment, and specify the dependency package storage path using the `--cann_3rd_lib_path` parameter.

```shell
# Specify the dependency package storage path. Default: ./third_party
bash build.sh --cann_3rd_lib_path={your_3rd_party_path}
```

After compilation, a `cann-hcomm_<version>_linux-<arch>.run` software package is generated in the `./build_out` directory.

`<version>` indicates the software version number, and `<arch>` indicates the operating system architecture. The values include x86_64 and aarch64.

### Installation

Install the compiled HCOMM software package:

```shell
bash ./build_out/cann-hcomm_<version>_linux-<arch>.run --full
```

During installation, replace the package name in the command with the actual software package name.

After installation, the HCOMM software package generated by the user replaces the HCOMM-related software in the installed CANN Toolkit development kit package.

### Uninstallation

To uninstall the compiled HCOMM software package and restore it to the state after installing the CANN Toolkit development kit package, use the following command:

```shell
bash ./build_out/cann-hcomm_<version>_linux-<arch>.run --uninstall
```

During uninstallation, replace the package name in the command with the actual software package name.

## Testing

### LLT Testing

After installing the compiled HCOMM software package, execute the following command to run LLT test cases:

```shell
bash build.sh --ut
```

### On-Board Testing

> [!Note]NOTE
> Before on-board testing, ensure that the driver, firmware, CANN Toolkit development kit package, and CANN ops operator package are installed.

Developers can use the HCCL Test tool for collective communication function and performance testing on the board. The workflow for using the HCCL Test tool is as follows:

1. Tool compilation

    Before using the HCCL Test tool, install the MPI dependency and compile the HCCL Test tool. For detailed instructions, refer to the "MPI Installation and Configuration" and "Tool Compilation" sections in the corresponding version of the [Ascend Documentation Center - HCCL Performance Test Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolHcclTest).

2. Disable signature verification

    The `cann-hcomm_<version>_linux-<arch>.run` software package generated from this source repository contains the following tar.gz subpackages:
      - `cann-hcomm-compat.tar.gz`: HCOMM compatibility upgrade package.
      - `cann-hccd-compat.tar.gz`: DataFlow compatibility upgrade package.
      - `aicpu_hcomm.tar.gz`: AI CPU communication base package.

    These tar.gz packages are loaded to the Device when the service starts. During the loading process, the driver performs security signature verification by default to ensure the package is trusted. Because the tar.gz packages compiled from this source repository do not contain a signature header, the driver security signature verification mechanism needs to be disabled.

   **Method to disable signature verification:**

       Use Ascend HDK 25.5.T2.B001 or later, and use the npu-smi tool provided with the Ascend HDK to disable signature verification. The following are reference commands. Execute them as the root user on the physical machine (using device 0 as an example).

      ```shell
      npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # Enable signature verification configuration
      npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # Disable custom signature verification
      ```

3. Execute the HCCL Test command to test the function and performance of collective communication.

    Using one compute node, 8 NPU devices, and testing the AllReduce operator performance as an example:

    ```shell
    # /usr/local/Ascend is the CANN software installation path for the root user under the default installation path. Replace it with the actual path.
    cd /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test

    # Data size (-b) from 8KB to 64MB, increment factor (-f) of 2, number of NPUs participating in training is 8
    mpirun -n 8 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
    ```

    For detailed usage instructions of the tool, refer to the "Tool Execution" section in the [Ascend Documentation Center - HCCL Performance Test Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolHcclTest).

4. View the results

    After executing the HCCL Test tool, the display output is as follows:

    ![hccltest_result](./figures/hccl_test_result.png)

    - "check_result" shows success, indicating that the communication operator executed successfully and the AllReduce operator function is correct.
    - "aveg_time": The execution time of the collective communication operator, in us.
    - "alg_bandwidth": The execution bandwidth of the collective communication operator, in GB/s.
    - "data_size": The amount of data participating in collective communication on a single NPU, in Bytes.

## Appendix

### Open Source Third-Party Software Dependencies

When compiling this project, the following third-party open source software dependencies are required:

| Open Source Software | Version | Download URL |
| ------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| json | 3.11.3 | [json-3.11.3.tar.gz](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz) |
| makeself | 2.5.0 | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
| openssl | 3.0.9 | [openssl-openssl-3.0.9.tar.gz](https://gitcode.com/cann-src-third-party/openssl/releases/download/openssl-3.0.9/openssl-openssl-3.0.9.tar.gz) |
| hcomm_utils | 9.0.0 (aarch64) | [cann-hcomm-utils_9.0.0_linux-aarch64.tar.gz](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260330_newest/cann-hcomm-utils_9.0.0_linux-aarch64.tar.gz) |
| hcomm_utils | 9.0.0 (x86_64) | [cann-hcomm-utils_9.0.0_linux-x86_64.tar.gz](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260330_newest/cann-hcomm-utils_9.0.0_linux-x86_64.tar.gz) |
| googletest | 1.14.0 | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| boost | 1.87.0 | [boost_1_87_0.tar.gz](https://gitcode.com/cann-src-third-party/boost/releases/download/v1.87.0/boost_1_87_0.tar.gz) |
| mockcpp | 2.7-h4 | [mockcpp-2.7.tar.gz](https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h4/mockcpp-2.7.tar.gz) |
| mockcpp-patch | 2.7-h4 | [mockcpp-2.7_py3.patch](https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h4/mockcpp-2.7_py3.patch) |
| abseil-cpp | 20250127.0 | [abseil-cpp-20250127.0.tar.gz](https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20250127.0/abseil-cpp-20250127.0.tar.gz) |
| protobuf | 25.1 | [protobuf-25.1.tar.gz](https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz) |
| rdma-core | v42.7-h1 | [rdma-core-42.7.tar.gz](https://gitcode.com/cann-src-third-party/rdma-core/releases/download/v42.7-h1/rdma-core-42.7.tar.gz) |
| rdma-core-patch | v42.7-h1 | [rdma-core-42.7.patch](https://gitcode.com/cann-src-third-party/rdma-core/releases/download/v42.7-h1/rdma-core-42.7.patch) |
| cann-cmake | master-051 | [cmake-master-051.tar.gz](https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-051.tar.gz) |

#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

function usage() {
  echo "Usage:"
  echo "  sh build.sh [-h | --help]"
  echo "              [--pkg]"
  echo "              [--package-path <PATH>] [--hcomm-path <PATH>]"
  echo "              [--version <VERSION>]"
  echo ""
  echo "Options:"
  echo "    -h, --help     Print usage"
  echo "    --pkg <PATH>    Set ascend package install path, default /usr/local/Ascend/cann"
  echo "    --package-path <PATH>"
  echo "                   Set ascend package install path, default /usr/local/Ascend/cann"
  echo "    --hcomm-path <PATH>"
  echo "                   Set hcomm code path, default /home/teamserver/xx/hcomm"
  echo "    --version <VERSION>"
  echo "                   Set build version to <VERSION>"
  echo ""
}

BUILD_MODE="HCCLVM"
aiv_build_enable="OFF"
package_enable=false

while [[ $# -gt 0 ]]; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
    --pkg)
        package_enable=true
        shift 1
        ;;
    --aicpu)
        BUILD_MODE="DEVICE"
        shift 1
        ;;
    --aiv)
        aiv_build_enable="ON"
        shift 1
        ;;
    --full)
        BUILD_MODE="FULL"
        aiv_build_enable="ON"
        shift 1
        ;;
    --package-path)
        ascend_package_path="$2"
        shift 2
        ;;
    --hcomm-path)
        hcomm_code_path="$2"
        shift 2
        ;;
    --hccl-path)
        hccl_code_path="$2"
        shift 2
        ;;
    --version)
        VERSION_INFO="$2"
        shift 2
        ;;
    *)
        echo "Error: Undefined option: $1"
        usage
        exit 1
        ;;
    esac
done

# 检查 HCOMM_CODE_HOME 是否配置
if [ -n "${hcomm_code_path}" ];then
    HCOMM_CODE_PATH=${hcomm_code_path}
    export HCOMM_CODE_HOME=${HCOMM_CODE_PATH}  # hcomm仓代码根路径
elif [[ -n "${HCOMM_CODE_HOME}" ]]; then
    HCOMM_CODE_PATH=${HCOMM_CODE_HOME}
else
    echo "Error: Please set the hcomm code path through parameter --hcomm-path or HCOMM_CODE_HOME environment variable."
    echo "Set by environment variable: export HCOMM_CODE_HOME=/path/to/hcomm"
    echo "Set by parameter: bash build.sh --hcomm-path=/path/to/hcomm"
    exit 1
fi

# 检查 HCOMM_CODE_HOME 路径是否存在
if [[ ! -d "${HCOMM_CODE_PATH}" ]]; then
    echo "Error: HCOMM_CODE_PATH directory does not exist: ${HCOMM_CODE_PATH}"
    exit 1
fi

# 当 BUILD_MODE 为 FULL/DEVICE 或 指定构建aiv时检查 HCCL_CODE_HOME 是否配置
if [[ "$BUILD_MODE" == "FULL" || "$BUILD_MODE" == "DEVICE" || "$aiv_build_enable" == "ON" ]]; then
    if [ -n "${hccl_code_path}" ]; then
        HCCL_CODE_PATH=${hccl_code_path}
        export HCCL_CODE_HOME=${HCCL_CODE_PATH}  # 设置 hccl 仓代码根路径
    elif [[ -n "${HCCL_CODE_HOME}" ]]; then
        HCCL_CODE_PATH=${HCCL_CODE_HOME}
    else
        echo "Error: BUILD_MODE is $BUILD_MODE. Please set the hccl code path through parameter --hccl-path or HCCL_CODE_HOME environment variable."
        echo "Set by environment variable: export HCCL_CODE_HOME=/path/to/hccl"
        echo "Set by parameter: bash build.sh --hccl-path=/path/to/hccl"
        exit 1
    fi

    # 检查 HCCL_CODE_PATH 路径是否存在
    if [[ ! -d "${HCCL_CODE_PATH}" ]]; then
        echo "Error: HCCL_CODE_PATH directory does not exist: ${HCCL_CODE_PATH}"
        exit 1
    fi
    
    echo "HCCL_CODE_PATH is set to: ${HCCL_CODE_PATH}"
fi

if [ -n "${ascend_package_path}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ascend_package_path}
    source ${ASCEND_CANN_PACKAGE_PATH}/set_env.sh
elif [ -n "${ASCEND_HOME_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
elif [ -n "${ASCEND_OPP_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=$(dirname ${ASCEND_OPP_PATH})
elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
elif [ -d "${DEFAULT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
else
    echo "Error: Please set the toolkit package installation directory through parameter --package-path."
    exit 1
fi

# 检查 source 是否成功（可选）
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to source set_env.sh of cann"
    exit 1
fi

build_host() {
    # 创建 build 目录（如果不存在）并进入
    mkdir -p build
    cd build

    # 执行编译命令
    echo "Running cmake and make..."
    cmake -DBUILD_DEVICE_ARM=OFF -DBUILD_AIV=$aiv_build_enable .. && make -j16 && make install

    # 检查编译是否成功
    if [[ $? -eq 0 ]]; then
        echo "Build completed successfully!"
    else
        echo "Build failed!"
        exit 1
    fi

    if [[ $package_enable == true ]]; then
        cd ..
        tar -czvf hccl_vm_install.tar.gz hccl_vm_install
        echo "Package completed successfully!"
    fi
}

build_device() {
    mkdir -p build_device
    mkdir -p build
    cd build
    cmake -DBUILD_DEVICE_ARM=ON .. && make -j16
    cd ..
}

if [[ "${BUILD_MODE}" == "HCCLVM" ]]; then
    build_host
elif [[ "${BUILD_MODE}" == "DEVICE" ]]; then
    build_device
elif [[ "${BUILD_MODE}" == "FULL" ]]; then
    build_device
    build_host
fi

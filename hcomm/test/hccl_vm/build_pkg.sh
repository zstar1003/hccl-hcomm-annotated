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

# 脚本所在目录的绝对路径（在任何 cd 之前计算，保证后续函数能找到 asset 目录下的工具脚本）
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==========================================
# 辅助函数: 原地解密并解压 AICPU 算子包
# ==========================================
# 从 ascen_package_load.ini 解析指定包的 package_path
function parse_ini_package_path() {
    # $1 = ini 路径
    # $2 = 要查询的包名（如 aicpu_hccl.tar.gz）
    local INI="$1"
    local NAME="$2"
    awk -v name="$NAME" '
        BEGIN { cur_name=""; cur_path="" }
        /^name:/        { cur_name=substr($0, 6); cur_path="" }
        /^package_path:/{ cur_path=substr($0, 14) }
        cur_name == name && cur_path != "" { print cur_path; exit }
    ' "$INI"
}

# 对指定 tar.gz 包调用 ci_img_headler.py 解密（有头剥头、无头直接拷出），
# 然后解压到指定目标目录（--strip-components=1 去掉 aicpu_kernels_device/ 前缀）
function decrypt_and_extract() {
    # $1 = 源 tar.gz 路径（可能带头也可能裸）
    # $2 = 解压目标目录（工具目录）
    local SRC="$1"
    local DEST_DIR="$2"
    if [ ! -f "$SRC" ]; then
        echo "错误: 找不到源文件 $SRC"
        return 1
    fi

    local HEADLER="$SCRIPT_DIR/third_party/cann-cmake/scripts/signtool/image_extract/ci_img_headler.py"
    if [ ! -f "$HEADLER" ]; then
        echo "错误: 找不到 $HEADLER，请先执行 bash build.sh --full 构建以拉取 cann-cmake 第三方依赖"
        return 1
    fi

    # 创建解密临时文件（在 /tmp 下，避免对只读 CANN 目录的写需求）
    local RAW_TMP
    RAW_TMP="$(mktemp /tmp/.aich_raw.XXXXXX)" || true
    if [ -z "$RAW_TMP" ] || [ ! -f "$RAW_TMP" ]; then
        echo "错误: mktemp 创建临时文件失败"
        return 1
    fi

    echo "解密 $(basename "$SRC") -> $(basename "$RAW_TMP")"
    python3 "$HEADLER" -img "$SRC" -raw "$RAW_TMP" --rcvr

    # 创建目标目录
    if ! mkdir -p "$DEST_DIR"; then
        rm -f "$RAW_TMP"
        echo "错误: 创建目标目录 $DEST_DIR 失败"
        return 1
    fi

    echo "解压 $(basename "$RAW_TMP") 到 $DEST_DIR (--strip-components=1)"
    if ! tar -zxf "$RAW_TMP" -C "$DEST_DIR" --strip-components=1; then
        rm -f "$RAW_TMP"
        echo "错误: 解压到 $DEST_DIR 失败"
        return 1
    fi

    rm -f "$RAW_TMP"
    return 0
}

# 部署单个 aicpu 算子包：从 CANN 安装目录读取 tarball，解密解压到指定目录
function deploy_aicpu_kernel_pkg() {
    local PKG_NAME="$1"
    local ASCEND_INSTALL_PATH="$2"
    local INI_PATH="$3"
    local DEST_DIR="$4"
    local SUB_PATH
    SUB_PATH=$(parse_ini_package_path "$INI_PATH" "$PKG_NAME")
    if [ -z "$SUB_PATH" ]; then
        echo "错误: $INI_PATH 中找不到 $PKG_NAME 的 package_path"
        return 1
    fi
    local SRC="$ASCEND_INSTALL_PATH/$SUB_PATH/$PKG_NAME"
    decrypt_and_extract "$SRC" "$DEST_DIR"
}

function usage() {
  echo "Usage:"
  echo "  sh build_pkg.sh [-h | --help]"
  echo "                  [--install <hccl|hcomm>]"
  echo "                  [--full]"
  echo "                  [--tool_path <PATH>]"
  echo ""
  echo "Options:"
  echo "    -h, --help     Print usage"
  echo "    --install <hccl|hcomm>"
  echo "                   Build/install the specified component, then decrypt+extract"
  echo "                   its aicpu kernel tarball to \$HCCL_VM_PATH/hccl_vm_install/lib/aarch64/"
  echo "    --full         Build/install both hccl and hcomm, then decrypt+extract both"
  echo "                   aicpu kernel tarballs to \$HCCL_VM_PATH/hccl_vm_install/lib/aarch64/"
  echo "                   (equivalent to \"--install hccl\" followed by \"--install hcomm\")"
  echo "    --tool_path <PATH>"
  echo "                   Set HCCL_VM_PATH, default: current directory"
  echo ""
  echo "Default (no option): only decrypt+extract aicpu_hccl.tar.gz and aicpu_hcomm.tar.gz"
  echo "                     (from CANN install root, per ascend_package_load.ini)"
  echo "                     to \$HCCL_VM_PATH/hccl_vm_install/lib/aarch64/."
  echo "                     No build/install is performed."
  echo ""
}

# ==========================================
# 第一步：解析脚本参数和环境变量
# ==========================================

# 1.默认值
# MODE 四态：deploy_only / install_hccl / install_hcomm / full
MODE=deploy_only
BUILD_HCCL=false
BUILD_HCOMM=false
TOOL_PATH=""

# 2.解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        --install)
            if [ "$2" == "hccl" ]; then
                MODE=install_hccl
                BUILD_HCCL=true
                BUILD_HCOMM=false
            elif [ "$2" == "hcomm" ]; then
                MODE=install_hcomm
                BUILD_HCCL=false
                BUILD_HCOMM=true
            else
                echo "错误: 无效的参数值，支持的参数值为 hccl 或 hcomm"
                exit 1
            fi
            shift 2
            ;;
        --full)
            MODE=full
            BUILD_HCCL=true
            BUILD_HCOMM=true
            shift
            ;;
        --tool_path)
            if [ -n "$2" ]; then
                TOOL_PATH="$2"
                shift 2
            else
                echo "错误: --tool_path 参数需要指定路径值"
                exit 1
            fi
            ;;
        *)
            echo "错误: 未知参数 $1"
            usage
            exit 1
            ;;
    esac
done

# 3.设置 HCCL_VM_PATH
if [ -n "$HCCL_VM_PATH" ]; then
    echo "使用环境变量中的 HCCL_VM_PATH: $HCCL_VM_PATH"
elif [ -n "$TOOL_PATH" ]; then
    export HCCL_VM_PATH="$TOOL_PATH"
    echo "使用 --tool_path 参数中的 HCCL_VM_PATH: $HCCL_VM_PATH"
else
    export HCCL_VM_PATH=$(pwd)
    echo "使用当前目录作为 HCCL_VM_PATH: $HCCL_VM_PATH"
fi

# 4.校验并获取 CANN 安装目录
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "错误: 环境变量 ASCEND_HOME_PATH 未设置，请参照source /home/workspace/Ascend/cann/set_env.sh设置"
    exit 1
fi

# 4.校验并获取 hccl 代码目录
if [ "$BUILD_HCCL" = true ] && [ -z "$HCCL_CODE_HOME" ]; then
    echo "错误: 环境变量 HCCL_CODE_HOME 未设置"
    exit 1
fi

# 5.校验并获取 hcomm 代码目录
if [ "$BUILD_HCOMM" = true ] && [ -z "$HCOMM_CODE_HOME" ]; then
    echo "错误: 环境变量 HCOMM_CODE_HOME 未设置"
    exit 1
fi

# 6.获得CANN安装目录: ASCEND_INSTALL_PATH
ASCEND_INSTALL_PATH=$(dirname "$ASCEND_HOME_PATH")

MACHINE_ARCH=$(uname -m)
if [ "$MACHINE_ARCH" = "aarch64" ] || [ "$MACHINE_ARCH" = "arm64" ]; then
    CANN_ARCH_DIR="aarch64-linux"
else
    CANN_ARCH_DIR="x86_64-linux"
fi

echo "CANN_ARCH_DIR: $CANN_ARCH_DIR (processor: $MACHINE_ARCH)"
echo "--- 环境变量解析成功 ---"
echo "HCCL_VM_PATH: $HCCL_VM_PATH"
echo "ASCEND_INSTALL_PATH: $ASCEND_INSTALL_PATH"
echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
if [ "$BUILD_HCCL" = true ]; then
    echo "HCCL_CODE_HOME: $HCCL_CODE_HOME"
fi
if [ "$BUILD_HCOMM" = true ]; then
    echo "HCOMM_CODE_HOME: $HCOMM_CODE_HOME"
fi
echo "构建配置:"
echo "  - MODE: $MODE"
echo "  - HCCL: $BUILD_HCCL"
echo "  - HCOMM: $BUILD_HCOMM"
echo "--------------------------"

# ==========================================
# 第二步：构建并安装 HCOMM
# ==========================================

# 关闭 Linux 系统对 Python 的保护锁，解决 pip 装包时报错
sudo rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

if [ "$BUILD_HCOMM" = true ]; then
    echo "正在开始构建 HCOMM..."
    cd "$HCOMM_CODE_HOME" || exit 1
    bash build.sh --full

    if [ $? -eq 0 ]; then
        echo "HCOMM 构建成功，准备安装..."
        # 自动匹配版本号run包
        CANN_HCOMM_PACKAGE=$(ls -t "$HCOMM_CODE_HOME"/build_out/cann-hcomm_*_linux-${MACHINE_ARCH}.run | head -n 1)
        
        # 检查是否找到安装包
        if [ ! -f "$CANN_HCOMM_PACKAGE" ]; then
            echo "错误: 未找到 HCOMM 安装包(.run文件)"
            exit 1
        fi
        
        echo "找到安装包: $CANN_HCOMM_PACKAGE"
        yes y | "$CANN_HCOMM_PACKAGE" --full --install-path="$ASCEND_INSTALL_PATH"
    else
        echo "错误: HCOMM 构建失败"
        exit 1
    fi
else
    echo "跳过 HCOMM 构建和安装..."
fi

# ==========================================
# 第三步：构建并安装 HCCL
# ==========================================
if [ "$BUILD_HCCL" = true ]; then
    echo "正在开始构建 HCCL..."
    cd "$HCCL_CODE_HOME" || exit 1
    bash build.sh --full

    if [ $? -eq 0 ]; then
        echo "HCCL 构建成功，准备安装..."
        # 自动匹配版本号run包
        CANN_HCCL_PACKAGE=$(ls -t "$HCCL_CODE_HOME"/build_out/cann-hccl_*_linux-${MACHINE_ARCH}.run | head -n 1)
        
        # 检查是否找到安装包
        if [ ! -f "$CANN_HCCL_PACKAGE" ]; then
            echo "错误: 未找到 HCCL 安装包(.run文件)"
            exit 1
        fi
        
        echo "找到安装包: $CANN_HCCL_PACKAGE"
        yes y | "$CANN_HCCL_PACKAGE" --full --install-path="$ASCEND_INSTALL_PATH"
    else
        echo "错误: HCCL 构建失败"
        exit 1
    fi
else
    echo "跳过 HCCL 构建和安装..."
fi

# ==========================================
# 第四步：从 CANN 安装目录解签名并解压 aicpu 包
# ==========================================
echo "正在处理 aicpu 相关的构建产物..."

INI_PATH="$ASCEND_HOME_PATH/conf/ascend_package_load.ini"
if [ ! -f "$INI_PATH" ]; then
    echo "错误: 找不到 $INI_PATH"
    exit 1
fi

# AICPU 算子解压目标目录
AICPU_DEPLOY_DIR="$HCCL_VM_PATH/hccl_vm_install/lib/aarch64"

# 默认模式和 install 模式都要从 CANN 根解压，解压到工具目录
if [ "$BUILD_HCCL" = true ] || [ "$MODE" = "deploy_only" ]; then
    echo "部署 aicpu_hccl.tar.gz -> $AICPU_DEPLOY_DIR"
    deploy_aicpu_kernel_pkg "aicpu_hccl.tar.gz" "$ASCEND_HOME_PATH" "$INI_PATH" "$AICPU_DEPLOY_DIR"
else
    echo "跳过 aicpu_hccl.tar.gz 处理..."
fi

if [ "$BUILD_HCOMM" = true ] || [ "$MODE" = "deploy_only" ]; then
    echo "部署 aicpu_hcomm.tar.gz -> $AICPU_DEPLOY_DIR"
    deploy_aicpu_kernel_pkg "aicpu_hcomm.tar.gz" "$ASCEND_HOME_PATH" "$INI_PATH" "$AICPU_DEPLOY_DIR"
else
    echo "跳过 aicpu_hcomm.tar.gz 处理..."
fi

# 修正目录和文件的权限，确保 .so 有足够权限被 dlopen 加载
sudo chmod -R 755 "$AICPU_DEPLOY_DIR"

echo "所有任务均已执行完成！"

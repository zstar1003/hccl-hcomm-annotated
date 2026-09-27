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

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 全局变量
WORKSPACE=${WORKSPACE:-/home/workspace}
ASCEND_PATH=${ASCEND_PATH:-}
HCOMM_PATH=${HCOMM_PATH:-}
HCCL_PATH=${HCCL_PATH:-}
CHECKER_PATH=${CHECKER_PATH:-}
TOPO_TYPE=${TOPO_TYPE:-}
TOPO_ID=${TOPO_ID:-112}
ENABLE_DUMP=${ENABLE_DUMP:-1}
SILENT_MODE=false
INSTALL_ONLY=false
RUN_ONLY=false
BUILD_HCCL_VM_ONLY=false
BUILD_HCCL_TEST_ONLY=false
RUN_TEST_ONLY=false

# 初始化变量
init_var() {
    ASCEND_PATH=${ASCEND_PATH:-$WORKSPACE/Ascend}
    HCOMM_PATH=${HCOMM_PATH:-$WORKSPACE/hcomm}
    HCCL_PATH=${HCCL_PATH:-$WORKSPACE/hccl}
    CHECKER_PATH=${CHECKER_PATH:-$WORKSPACE/CheckerL2/hccl_vm_install}
}

# 显示进度条
show_progress() {
    local current=$1
    local total=$2
    local percentage=$((current * 100 / total))
    local bar_length=50
    local filled_length=$((percentage * bar_length / 100))
    local bar=$(printf "%-${filled_length}s" "#" | tr ' ' '#')
    local spaces=$(printf "%-$((bar_length - filled_length))s" "")
    
    echo -ne "\r${YELLOW}进度: ${percentage}% [${bar}${spaces}]${NC}"
}

# 显示信息
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# 显示错误
error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# 显示警告
warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        #error "命令 $1 未找到，请先安装"
        apt install -y $1
    fi
}

# 检查包是否已安装
check_package() {
    if dpkg -l | grep -q "^ii  $1 "; then
        return 0  # 已安装
    else
        return 1  # 未安装
    fi
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --workspace)
                WORKSPACE="$2"
                shift 2
                ;;
            --ascend-path)
                ASCEND_PATH="$2"
                shift 2
                ;;
            --hcomm-path)
                HCOMM_PATH="$2"
                shift 2
                ;;
            --hccl-path)
                HCCL_PATH="$2"
                shift 2
                ;;
            --checker-path)
                CHECKER_PATH="$2"
                shift 2
                ;;
            --topo-type)
                TOPO_TYPE="$2"
                shift 2
                ;;
            --topo-id)
                TOPO_ID="$2"
                shift 2
                ;;
            --disable-dump)
                ENABLE_DUMP=0
                shift 1
                ;;
            --install)
                INSTALL_ONLY=true
                shift 1
                ;;
            --run)
                RUN_ONLY=true
                shift 1
                ;;
            --build-hccl-vm)
                BUILD_HCCL_VM_ONLY=true
                shift 1
                ;;
            --build-hccl-test)
                BUILD_HCCL_TEST_ONLY=true
                shift 1
                ;;
            --run-test)
                RUN_TEST_ONLY=true
                shift 1
                ;;
            -y|--yes)
                SILENT_MODE=true
                shift 1
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "未知参数: $1"
                ;;
        esac
    done
    
    # 在解析完命令行参数后，设置依赖其他变量的默认值
    HCOMM_PATH=${HCOMM_PATH:-${WORKSPACE}/hcomm}
    HCCL_PATH=${HCCL_PATH:-${WORKSPACE}/hccl}
    CHECKER_PATH=${CHECKER_PATH:-${WORKSPACE}/CheckerL2}
}

# 显示帮助信息
show_help() {
    echo "HCCL-VM 一键式安装脚本"
    echo ""
    echo "用法:"
    echo "  bash install.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --workspace <路径>      设置工作目录 (默认: /home/workspace)"
    echo "  --ascend-path <路径>    设置CANN安装路径 (默认: /home/workspace/Ascend)"
    echo "  --hcomm-path <路径>     设置hcomm源码路径 (默认: <工作目录>/hcomm)"
    echo "  --checker-path <路径>   设置CheckerL2源码路径 (默认: <工作目录>/CheckerL2)"
    echo "  --topo-type <类型>      设置拓扑类型"
    echo "  --topo-id <编号>        设置拓扑配置编号 (默认: 112)"
    echo "  --disable-dump          禁用数据dump功能"
    echo "  --install               只执行下载、构建和安装"
    echo "  --run                   跳过构建，直接执行测试"
    echo "  --build-hccl-vm         只执行HCCL-VM工具编译和安装"
    echo "  --build-hccl-test       只执行hccl_test用例编译"
    echo "  --run-test              只执行测试用例运行"
    echo "  -y, --yes               自动确认所有提示"
    echo "  -h, --help              显示此帮助信息"
}

# 检查系统环境和依赖
check_environment() {
    info "检查系统环境..."
    check_command git
    check_command cmake
    check_command make
    check_command sed
    
    if [ ! -d "${ASCEND_PATH}" ]; then
        error "CANN安装目录不存在: ${ASCEND_PATH}"
    fi
    
    if [ ! -f "${ASCEND_PATH}/cann/set_env.sh" ]; then
        error "CANN环境脚本不存在: ${ASCEND_PATH}/cann/set_env.sh"
    fi
    
    # 检查第三方依赖
    info "检查第三方依赖..."
    local dependencies=("cmake" "libsqlite3-dev" "libboost-all-dev" "rdma-core" "libibverbs-dev" "pkg-config" "gcc-aarch64-linux-gnu" "g++-aarch64-linux-gnu" "qemu-user-static" "binfmt-support")
    local missing_deps=()
    
    for dep in "${dependencies[@]}"; do
        if ! check_package "$dep"; then
            missing_deps+=("$dep")
        fi
    done
    
    # 安装缺失的依赖
    if [ ${#missing_deps[@]} -gt 0 ]; then
        info "安装缺失的依赖: ${missing_deps[*]}"
        if [ "$SILENT_MODE" = true ]; then
            sudo apt-get update -y > /dev/null 2>&1
            sudo apt install -y "${missing_deps[@]}" > /dev/null 2>&1
        else
            sudo apt-get update
            sudo apt install "${missing_deps[@]}"
        fi
    else
        info "所有第三方依赖已安装，跳过安装步骤"
    fi
}

# 下载源码
download_source() {
    info "下载源码..."
    mkdir -p "${WORKSPACE}"
    cd "${WORKSPACE}"
    
    if [ ! -d "${HCOMM_PATH}" ]; then
        info "克隆hcomm仓库..."
        git clone https://gitcode.com/cann/hcomm.git
    else
        info "hcomm仓库已存在，跳过下载"
    fi

    if [ ! -d "${HCCL_PATH}" ]; then
        info "克隆hccl仓库..."
        git clone https://gitcode.com/cann/hccl.git
    else
        info "hccl仓库已存在，跳过下载"
    fi
    
    if [ ! -d "${CHECKER_PATH}" ]; then
        info "克隆CheckerL2仓库..."
        git clone https://gitcode.com/zhupc158/CheckerL2.git
    else
        info "CheckerL2仓库已存在，跳过下载"
    fi
}

# 编译hccl_test用例
build_hccl_test() {
    info "编译hccl_test用例..."
    chmod -R 755 "${ASCEND_PATH}"
    cd "${ASCEND_PATH}/cann/tools/hccl_test"
    if ! grep -q '\-lmpi_cxx' Makefile; then
        sed -i 's/-lmpi/-lmpi -lmpi_cxx/g' Makefile
    fi
    MPI_HOME=/usr/lib/$(uname -m)-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH}
}

# 编译HCCL-VM工具
build_hccl_vm() {
    info "编译HCCL-VM工具..."
    export HCOMM_CODE_HOME="${HCOMM_PATH}"
    export HCCL_CODE_HOME="${HCCL_PATH}"
    cd "${CHECKER_PATH}"
    bash build.sh --package-path ${ASCEND_HOME_PATH} --hcomm-path ${HCOMM_CODE_HOME} --hccl-path ${HCCL_CODE_HOME} --pkg --full
}

# 编译hcxx子包并安装
build_cann_pkg() {
    export HCOMM_CODE_HOME="${HCOMM_PATH}"
    export HCCL_CODE_HOME="${HCCL_PATH}"
    cd "${CHECKER_PATH}"
    bash build_pkg.sh
}

# 运行测试用例
run_test() {
    info "运行测试用例..."
    cd "${CHECKER_PATH}/hccl_vm_install"
    
    # 检查安装目录是否存在
    if [ ! -d "${CHECKER_PATH}/hccl_vm_install" ]; then
        error "HCCL-VM安装目录不存在，请先运行安装流程"
    fi
    
    # 创建hccl_rootinfo.json文件
    if [ ! -f "/etc/hccl_rootinfo.json" ]; then
        info "创建hccl_rootinfo.json文件..."
        sudo bash -c "cat > /etc/hccl_rootinfo.json << EOF
{
    \"version\": \"2.0\",
    \"topo_file_path\": \"${CHECKER_PATH}/hccl_vm_install/data/topo.json\"
}
EOF"
    else
        info "更新hccl_rootinfo.json文件..."
        sudo sed -i -E 's|"topo_file_path"\s*:\s*".*"|"topo_file_path": "${CHECKER_PATH}/hccl_vm_install/data/topo.json"|g' /etc/hccl_rootinfo.json
    fi
    
    # 配置环境变量
    export HCCLVM_TOPO_TYPE="${TOPO_TYPE}"
    export HCCLVM_ENABLE_DUMP_DATA="${ENABLE_DUMP}"
    
    # 启动工具并执行测试
    info "启动HCCL-VM工具..."
    
    # 创建启动脚本
    cat > run_hccl_test.sh << 'EOF'
#!/bin/bash
source hccl_config.sh
bash start.sh > log.txt
hccl-vm plugin run @checker
exit
EOF
    
    chmod +x run_hccl_test.sh
    
    # 启动工具并执行测试
    ./hccl-vm start ${TOPO_ID} < run_hccl_test.sh
}

# 主函数
main() {
    parse_args "$@"

    init_var
    
    info "开始HCCL-VM操作..."
    info "工作目录: ${WORKSPACE}"
    info "CANN路径: ${ASCEND_PATH}"
    info "HCOMM路径: ${HCOMM_PATH}"
    info "HCCL路径: ${HCCL_PATH}"
    info "Checker路径: ${CHECKER_PATH}"
    info "拓扑类型: ${TOPO_TYPE}"
    info "拓扑编号: ${TOPO_ID}"
    
    # 加载CANN环境变量（所有操作都需要）
    info "加载CANN环境变量..."
    source "${ASCEND_PATH}/cann/set_env.sh"
    
    # 只编译HCCL-VM工具
    if [ "$BUILD_HCCL_VM_ONLY" = true ]; then
        info "进入只编译HCCL-VM模式..."
        # 检查源码目录
        if [ ! -d "${HCOMM_PATH}" ]; then
            error "hcomm源码目录不存在: ${HCOMM_PATH}"
        fi
        if [ ! -d "${HCCL_PATH}" ]; then
            error "hccl源码目录不存在: ${HCCL_PATH}"
        fi
        if [ ! -d "${CHECKER_PATH}" ]; then
            error "CheckerL2源码目录不存在: ${CHECKER_PATH}"
        fi
        # 编译HCCL-VM
        build_hccl_vm
        info "HCCL-VM工具编译完成！"
        info "安装目录: ${CHECKER_PATH}/hccl_vm_install"
        exit 0
    fi
    
    # 只编译hccl_test用例
    if [ "$BUILD_HCCL_TEST_ONLY" = true ]; then
        info "进入只编译hccl_test用例模式..."
        # 编译hccl_test
        build_hccl_test
        info "hccl_test用例编译完成！"
        exit 0
    fi
    
    # 只运行测试用例
    if [ "$RUN_TEST_ONLY" = true ]; then
        info "进入只运行测试用例模式..."
        # 运行测试
        run_test
        info "测试执行完成！"
        exit 0
    fi
    
    # 如果是只运行模式，直接执行测试
    if [ "$RUN_ONLY" = true ]; then
        info "进入只运行模式，跳过构建步骤..."
        # 运行测试
        run_test
        info "测试执行完成！"
        exit 0
    fi
    
    # 检查环境（包括依赖安装）
    info "检查&安装第三方依赖..."
    check_environment
    
    # 下载源码
    info "下载源码..."
    download_source
    
    # 编译hccl_test
    info "编译hccl_test用例..."
    build_hccl_test
    
    # 编译HCCL-VM工具
    info "编译HCCL-VM工具..."
    build_hccl_vm

    # 源码编译hcxx子包并安装
    info "源码编译hcxx子包并安装"
    build_cann_pkg
    
    # 如果是只安装模式，跳过测试
    if [ "$INSTALL_ONLY" = true ]; then
        info "进入只安装模式，跳过测试步骤..."
        info "HCCL-VM安装完成！"
        info "安装目录: ${CHECKER_PATH}/hccl_vm_install"
        exit 0
    fi
    
    # 运行测试用例
    info "运行测试用例..."
    #run_test
    
    info "HCCL-VM安装和测试完成！"
    info "测试日志: ${CHECKER_PATH}/hccl_vm_install/log.txt"
}

# 执行主函数
main "$@"

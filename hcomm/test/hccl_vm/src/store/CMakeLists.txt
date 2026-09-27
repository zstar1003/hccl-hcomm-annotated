# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE HCCL_SOURCES CONFIGURE_DEPENDS "*.cc")

# 添加静态库
add_library(store STATIC ${HCCL_SOURCES})
hcclvm_apply_common_flags(store)

# 设置头文件包含路径（改为绝对路径，基于项目根目录）
target_include_directories(store PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/runnerdb
    ${PROJECT_SOURCE_DIR}/include/store

    ${HCCL_VM_SRC}
    ${HCCL_VM_SRC}/store
    ${THRID_PARTY_DIR}/nlohmann_json

    ## cann包头文件
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/toolchain

    # cann包头文件
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)

target_link_libraries(store PRIVATE log loader)

# 日志染色: 模块 tag
target_compile_definitions(store PRIVATE HCCL_VM_MODULE="STORE")

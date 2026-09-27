# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(NOT BUILD_OPEN_PROJECT)
    set(INSTALL_CCL_KERNEL_JSON_DIR ${INSTALL_LIBRARY_DIR})
    set(INSTALL_DPU_KERNEL_JSON_DIR ${INSTALL_LIBRARY_DIR})
    message(STATUS "BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")
    message(STATUS "TOP_DIR=${TOP_DIR}")
    message(STATUS "CANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}")
    message(STATUS "INSTALL_LIBRARY_DIR=${INSTALL_LIBRARY_DIR}")
endif()

if(NOT DEFINED HCOMM_DIR)
    set(HCOMM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()

# AIV算子调测阶段可配置为1以打开AscendC打印功能
if(NOT DEFINED ASCENDC_DUMP)
    set(ASCENDC_DUMP "0")
endif()

message(STATUS "HCOMM_DIR=${HCOMM_DIR}")
message(STATUS "ASCENDC_DUMP=${ASCENDC_DUMP}")
message(STATUS "PRODUCT_SIDE=${PRODUCT_SIDE}")

# 组件间对外头文件目录
add_library(hcomm_headers INTERFACE)
target_include_directories(hcomm_headers INTERFACE
    $<BUILD_INTERFACE:${HCOMM_DIR}/include>
    $<BUILD_INTERFACE:${HCOMM_DIR}/include/hccl>
    $<BUILD_INTERFACE:${HCOMM_DIR}/pkg_inc>
    $<BUILD_INTERFACE:${HCOMM_DIR}/pkg_inc/hccl>
    $<BUILD_INTERFACE:${HCOMM_DIR}/include/hcomm>
    $<BUILD_INTERFACE:${HCOMM_DIR}/include/hcomm/ccu>
    $<BUILD_INTERFACE:${HCOMM_DIR}/pkg_inc/hcomm>
    $<BUILD_INTERFACE:${HCOMM_DIR}/pkg_inc/hcomm/ccu>
    $<BUILD_INTERFACE:${HCOMM_DIR}/pkg_inc/legacy>
    $<BUILD_INTERFACE:${HCOMM_DIR}/pkg_inc/legacy/hccl>
)

add_library(hccl_headers INTERFACE)
target_include_directories(hccl_headers INTERFACE
    $<BUILD_INTERFACE:${HCOMM_DIR}/include>
    $<BUILD_INTERFACE:${HCOMM_DIR}/include/hccl>
    $<BUILD_INTERFACE:${HCOMM_DIR}/pkg_inc>
    $<BUILD_INTERFACE:${HCOMM_DIR}/pkg_inc/hccl>
    $<BUILD_INTERFACE:${HCOMM_DIR}/pkg_inc/hcomm>
    $<BUILD_INTERFACE:${HCOMM_DIR}/pkg_inc/legacy>
    $<BUILD_INTERFACE:${HCOMM_DIR}/pkg_inc/legacy/hccl>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/hccl>
    $<INSTALL_INTERFACE:include/hccl/external>
    $<INSTALL_INTERFACE:include/hccl/external/hccl>
)

# 将子目录添加到构建过程
add_subdirectory(legacy)            # 包含 ascend910 和 ascend950
add_subdirectory(base_comm)         # 基础通信组件
add_subdirectory(coll_communicator_mgr) # 集合通信组件

# 定义 aicpu_custom 库，在 device 侧使用
# 注意：aicpu_custom 依赖 ccl_kernel 的源文件列表，因此需要在其后定义
if(PRODUCT_SIDE STREQUAL "device")
    include(legacy/ascend910/framework/aicpu_custom.cmake)
endif()

# 临时修改，后面做base_comm和hcomm二进制分离
if (TARGET hcomm AND TARGET hcomm_base_obj)
    target_link_libraries(hcomm hcomm_base_obj)
    add_dependencies(hcomm hcomm_base_obj)
endif()
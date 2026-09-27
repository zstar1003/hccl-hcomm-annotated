# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_compile_options(
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
)
# 日志染色: aiv_kernel 整片统一 [AIV]
add_compile_definitions(
        HCCL_VM_MODULE="AIV"
        _GLIBCXX_USE_CXX11_ABI=0
)

if(NOT DEFINED HCCL_CODE_ROOT)
    if(NOT DEFINED ENV{HCCL_CODE_HOME})
        message(FATAL_ERROR "Error: BUILD_AIV is ON, but HCCL_CODE_HOME environment variable is not set. Please set it to the HCCL source path.")
    endif()
    set(HCCL_CODE_ROOT $ENV{HCCL_CODE_HOME})
endif()
message(STATUS "HCCL_CODE_ROOT: ${HCCL_CODE_ROOT}")

set(AIV_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
MESSAGE(STATUS "AIV_KERNEL_DIR: " ${AIV_KERNEL_DIR})

set(HCCL_AIV_DEFINES
        ${HCCL_CODE_ROOT}/src/ops/op_common/algorithm/template/aiv/aiv_defines.h)
if(NOT EXISTS ${HCCL_AIV_DEFINES})
    message(FATAL_ERROR "HCCL aiv_defines.h not found: ${HCCL_AIV_DEFINES}")
endif()

set(AIV_SHADOW_DIR ${CMAKE_CURRENT_BINARY_DIR}/hccl_shadow)
file(MAKE_DIRECTORY ${AIV_SHADOW_DIR})
file(CREATE_LINK
        ${HCCL_AIV_DEFINES}
        ${AIV_SHADOW_DIR}/aiv_defines.h
        SYMBOLIC)
message(STATUS "AIV_SHADOW_DIR: ${AIV_SHADOW_DIR}")

set(AIV_KERNEL_STUB_INCLUDE_LIST
        ${AIV_KERNEL_DIR}/aiv_mode_stub
        ${AIV_KERNEL_DIR}/ascendc_stub
        ${AIV_KERNEL_DIR}/hccl_op_stub
        ${AIV_SHADOW_DIR}

        ${HCCL_VM_INCLUDE}

        ${THRID_PARTY_DIR}
        ${THRID_PARTY_DIR}/nlohmann_json
        ${THRID_PARTY_DIR}/dbg_macro   # for debug

        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
)

add_library(hccl_aiv_kernel SHARED
        $<TARGET_OBJECTS:hccl_aiv_kernel_base>

        $<TARGET_OBJECTS:hccl_aiv_all_gather_op>
        $<TARGET_OBJECTS:hccl_aiv_all_reduce_op>
        $<TARGET_OBJECTS:hccl_aiv_all_to_all_op>
        $<TARGET_OBJECTS:hccl_aiv_broadcast_op>
        $<TARGET_OBJECTS:hccl_aiv_reduce_op>
        $<TARGET_OBJECTS:hccl_aiv_reduce_scatter_op>
        $<TARGET_OBJECTS:hccl_aiv_recv_op>
        $<TARGET_OBJECTS:hccl_aiv_scatter_op>
        $<TARGET_OBJECTS:hccl_aiv_send_op>

        ${HCCL_VM_SRC}/utils/sim_common_api.cc
)

target_link_libraries(hccl_aiv_kernel PRIVATE log)

install(TARGETS hccl_aiv_kernel
        DESTINATION "lib/${CMAKE_SYSTEM_PROCESSOR}"
        PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

add_subdirectory(hccl_op_stub)

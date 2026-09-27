# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccl_ut_base_comm_nic_plugin_list
    main.cc
    ut_nic_plugin_manager.cc
)

add_executable(hccl_ut_base_comm_nic_plugin ${hccl_ut_base_comm_nic_plugin_list})

target_include_directories(hccl_ut_base_comm_nic_plugin PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
    # dlog 桩头文件(log.h 依赖 <dlog_pub.h>)
    ${HCCL_CODE_ROOT}/test/stub/depends/pkg_inc/base
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/api_c_adpt/nic_plugin
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/api_c_adpt
    ${HCCL_CODE_ROOT}/experimental/base_comm/nic_plugin
)

target_compile_definitions(hccl_ut_base_comm_nic_plugin PRIVATE
    HCOMM_CODE_ROOT="${HCCL_CODE_ROOT}"
)

target_compile_options(hccl_ut_base_comm_nic_plugin PRIVATE
    -U_FORTIFY_SOURCE
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
)

target_link_libraries(hccl_ut_base_comm_nic_plugin PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    -pthread
    hccl_llt
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lgcov
    -lrt
    -ldl
)

add_dependencies(hccl_ut_base_comm_nic_plugin hccl_llt)

enable_testing()
add_test(NAME hccl_ut_base_comm_nic_plugin COMMAND hccl_ut_base_comm_nic_plugin)

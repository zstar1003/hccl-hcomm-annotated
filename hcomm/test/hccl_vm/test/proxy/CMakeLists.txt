# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -D_GLIBCXX_USE_CXX11_ABI=0")

include_directories(
    ${CMAKE_SOURCE_DIR}/src/proxy
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/include/runnerdb
    ${CMAKE_SOURCE_DIR}/include/hccl
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    # From level1 library includes
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/asc/include/adv_api
    ${HCOMM_CODE_ROOT}/test/ut/depends/include
    ${HCOMM_CODE_ROOT}/include
    ${HCOMM_CODE_ROOT}/include/hccl
    ${HCOMM_CODE_ROOT}/pkg_inc
    ${HCOMM_CODE_ROOT}/pkg_inc/hccl
    ${HCOMM_CODE_ROOT}/src/pub_inc
    ${HCOMM_CODE_ROOT}/src/framework/communicator/impl/independent_op/rank_graph
    ${HCOMM_CODE_ROOT}/src/framework/inc
    ${CMAKE_SOURCE_DIR}/include
    ${HCOMM_CODE_ROOT}/test/ut/depends/pkg_inc/profiling
    ${HCOMM_CODE_ROOT}/test/ut/depends/pkg_inc/aicpu/tsd
    ${HCOMM_CODE_ROOT}/test/ut/depends/pkg_inc/aicpu
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/profiling
    ${HCOMM_CODE_ROOT}/src/platform/hccp/inc/network
    ${HCOMM_CODE_ROOT}/src/legacy/common
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/external_depends/ccu
    ${HCOMM_CODE_ROOT}/src/legacy/ascend910/pub_inc
)

link_libraries(
    gtest
    log
    store
    runnerdb
    sqlite3
    dl
    rt
)

# test_hccl_proxy_common
add_executable(test_hccl_proxy_common hccl_proxy_common_test.cc)
target_sources(test_hccl_proxy_common PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/hccl_proxy_common.cc
)
set_target_properties(test_hccl_proxy_common PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_hccl_proxy_common gtest_main)

# test_hccl_inner_stub
add_executable(test_hccl_inner_stub hccl_inner_stub_test.cc)
target_sources(test_hccl_inner_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/hccl_inner_stub.cc
)
set_target_properties(test_hccl_inner_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_hccl_inner_stub gtest_main)

# test_tsd_client_stub
add_executable(test_tsd_client_stub tsd_client_stub_test.cc)
target_sources(test_tsd_client_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/tsd_client_stub.cc
)
set_target_properties(test_tsd_client_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_tsd_client_stub gtest_main)

# test_scatter_stub
add_executable(test_scatter_stub scatter_stub_test.cc)
target_sources(test_scatter_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/scatter_stub.cc
)
set_target_properties(test_scatter_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_scatter_stub gtest_main)

# test_log_stub
add_executable(test_log_stub log_stub_test.cc)
target_sources(test_log_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/log_stub.cc
)
set_target_properties(test_log_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_log_stub gtest_main)

# test_aprofiling_stub
add_executable(test_aprofiling_stub aprofiling_stub_test.cc)
target_sources(test_aprofiling_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/aprofiling_stub.cc
)
set_target_properties(test_aprofiling_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_aprofiling_stub gtest_main)

# test_aclrt_stub
add_executable(test_aclrt_stub aclrt_stub_test.cc)
target_sources(test_aclrt_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/aclrt_stub.cc
)
set_target_properties(test_aclrt_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_aclrt_stub gtest_main)

# test_aclrt_memory_stub —— 主机侧大块引流/上界/FreeHost 行为单测
# InsertTaskToCollection 由 link_libraries 的 store 库提供，无需额外编 device_arm 源。
add_executable(test_aclrt_memory_stub aclrt_memory_stub_test.cc)
target_sources(test_aclrt_memory_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/aclrt_memory_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/hccl_proxy_common.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_sub_process_manager.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
)
set_target_properties(test_aclrt_memory_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_aclrt_memory_stub gtest_main)

add_executable(test_hccl_comm_stub hccl_comm_stub_test.cc)
target_sources(test_hccl_comm_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/hccl_comm_stub.cc
)
set_target_properties(test_hccl_comm_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_hccl_comm_stub gtest_main)

add_executable(test_ai_core_stub ai_core_stub_test.cc)
target_sources(test_ai_core_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/ai_core_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub/aiv_task.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/ascendc_stub/ascendc_base_stub.cc
)
target_include_directories(test_ai_core_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/aiv_mode_stub
    ${CMAKE_SOURCE_DIR}/src/proxy/aiv_kernel/ascendc_stub
)
set_target_properties(test_ai_core_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ai_core_stub gtest_main)

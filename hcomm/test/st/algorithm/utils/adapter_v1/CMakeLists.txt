# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(adaption_to_test_list
    # 依赖平台层的代码（这边最好不依赖平台层）
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/alg_adapt_v1_interface.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/device_info_recorder.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/orchestrate.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/transformer.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/utils_stub.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/communicator_stub.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/log_stub.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/slog_stub.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/network_manager_stub.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/error_manager_stub.cpp
    # ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/hccl_aiv_stub.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/sal_stub.cc

    # #aiv业务代码
    # ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/aiv/aiv_base_stub.cc
    # ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/aiv/aiv_copy_stub.cc
    # ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/aiv/aiv_memory_stub.cc
    # ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/aiv/aiv_sync_stub.cc
    # ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/aiv/sync_interface.cc

    ${HCOMM_CODE_PLATFORM}/resource/mem/mem_host.cc
    ${HCOMM_CODE_PLATFORM}/resource/mem/mem_device.cc
    ${HCOMM_CODE_PLATFORM}/common/workflow.cc
    ${HCOMM_CODE_PLATFORM}/common/device_capacity.cc

    ${HCOMM_CODE_PLATFORM}/common/externalinput.cc
    ${HCOMM_CODE_PLATFORM}/common/dlhal_function.cc
    ${HCOMM_CODE_ROOT}/src/base_comm/common/hccl_dl.cc
    ${HCOMM_CODE_PLATFORM}/common/hccl_ip_address.cc
    ${HCOMM_CODE_PLATFORM}/debug/traceinfo/hccl_trace_info.cc

    # 依赖框架层的代码
    ${HCOMM_CODE_FRAMEWORK}/common/src/config.cc
    ${HCOMM_CODE_FRAMEWORK}/common/src/config/env_config.cc
    ${HCOMM_CODE_FRAMEWORK}/common/src/config/env_config_host.cc
    ${HCOMM_CODE_FRAMEWORK}/common/src/topo/topoinfo_parse.cc
    ${HCOMM_CODE_FRAMEWORK}/common/src/topo/topoinfo_ranktableParser.cc
    ${HCOMM_CODE_FRAMEWORK}/common/src/topo/topoinfo_ranktableConcise.cc
    ${HCOMM_CODE_FRAMEWORK}/common/src/topo/topoinfo_ranktableHeterog.cc
    ${HCOMM_CODE_FRAMEWORK}/common/src/topo/topoinfo_ranktableStandard.cc
    ${HCOMM_CODE_FRAMEWORK}/common/src/topo/topoinfo_roletableParser.cc
    ${HCOMM_CODE_FRAMEWORK}/common/src/json_utils.cc
    ${HCOMM_CODE_FRAMEWORK}/communicator/impl/hccl_communicator_attrs.cc
    ${HCOMM_CODE_FRAMEWORK}/communicator/impl/hccl_communicator_attrs_host.cc

    # 依赖开源仓的代码
    ${HCOMM_CODE_ALGORITHM}/impl/resource_manager/ccl_buffer_manager.cc
    ${HCOMM_CODE_ALGORITHM}/impl/resource_manager/share_ccl_buffer_manager.cc
    ${HCOMM_CODE_ALGORITHM}/impl/resource_manager/stream_active_manager.cc
    ${HCOMM_CODE_COMMON}/debug/profiling/profiler_base.cc
    ${HCOMM_CODE_COMMON}/debug/config/config_log.cc
    ${HCOMM_CODE_COMMON}/launch_device/launch_device.cc
)

#aiv相关用c++17
set_source_files_properties(
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/aiv/aiv_base_stub.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/aiv/aiv_copy_stub.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/aiv/aiv_memory_stub.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/aiv/aiv_sync_stub.cc
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/aiv/sync_interface.cc
    PROPERTIES
    COMPILE_OPTIONS "-std=c++17"
)

add_library(hccl_alg_adaption SHARED ${adaption_to_test_list})

add_subdirectory(dispatcher)
add_subdirectory(transport)
add_subdirectory(notify)
add_subdirectory(runtime)
add_subdirectory(stream)

#头文件搜索路径
target_include_directories(hccl_alg_adaption PRIVATE
    ${UTILS_PUBLIC_INCLUDE_LIST}
    ${UTILS_INNER_INCLUDE_LIST}

    ${HCOMM_PUBLIC_INCLUDE_LIST}
    ${HCOMM_PLATFORM_INCLUDE_LIST}
    ${HCOMM_ALGORITHM_INCLUDE_LIST}
    ${HCOMM_COMMON_INCLUDE_LIST}
    ${HCOMM_FRAMEWORK_INCLUDE_LIST}
    ${HCOMM_LEGACY_INCLUDE_LIST}

    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto

    ${ASCEND_CANN_INCLUDE_LIST}
    ${CMAKE_SOURCE_DIR}/test/legacy/depends/toolchain/
    ${HCOMM_DIR}/test/stub/depends/pkg_inc/base
)

target_compile_definitions(hccl_alg_adaption PRIVATE
    google=ascend_private
    -DOPEN_HCCL_TEST
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(hccl_alg_adaption PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O2
    -pipe
)

target_link_libraries(hccl_alg_adaption PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    hccl_alg_test
    -pthread
    ascend_protobuf
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    -lrt
    -ldl
)

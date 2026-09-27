# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(include_list
    # 测试框架的代码
    ${UTILS_DIR}/inc
    ${UTILS_DIR}/checker
    ${UTILS_DIR}/checker/common/types
    ${UTILS_DIR}/checker/rank_info_recorder
    ${UTILS_DIR}/checker/mem_layout
    ${UTILS_DIR}/checker/link_type_recorder
    ${UTILS_DIR}/checker/task_collector
    ${UTILS_DIR}/checker/semantics_check
    ${UTILS_DIR}/checker/singletask_check
    ${UTILS_DIR}/checker/ui/data_dumper
    ${UTILS_DIR}/checker/param_trans
    ${UTILS_DIR}/checker/mem_conflict_check
    ${UTILS_DIR}/adapter_v1/aiv
    ${UTILS_DIR}/adapter_v1

    ${UTILS_DIR}/adapter_v2

    ${HCOMM_CODE_LEGACY}/common/types
    ${HCOMM_CODE_LEGACY}/common/utils
    ${HCOMM_CODE_LEGACY}/common/exception
    ${HCOMM_CODE_LEGACY}/include
    ${HCOMM_ROOT_PATH}/src/legacy/ascend910/pub_inc
    ${HCOMM_ROOT_PATH}/src/base_comm/common
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new
    ${HCOMM_DIR}/src/legacy/ascend910/common
    ${HCOMM_CODE_PLATFORM}/common

    ${HCOMM_CODE_LEGACY}/service/collective/primitive

    ${ASCEND_CANN_INCLUDE_LIST}

    ${HCOMM_CODE_ALGORITHM}/base/inc
    ${HCOMM_CODE_ALGORITHM}/impl
    ${HCOMM_CODE_ALGORITHM}/impl/task
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor
    ${HCOMM_CODE_ALGORITHM}/impl/resource_manager
    ${HCOMM_CODE_ALGORITHM}/base/alg_template
    ${HCOMM_CODE_ALGORITHM}/base/communicator
    ${HCOMM_CODE_ALGORITHM}/base/alg_template/component

    ${HCOMM_CODE_ALGORITHM}/impl/inc
    ${HCOMM_CODE_ALGORITHM}/impl/legacy
    ${HCOMM_CODE_ALGORITHM}/base/communicator/legacy

    ${HCOMM_CODE_COMMON}/debug/profiling
    ${HCOMM_CODE_COMMON}/debug/profiling/inc
    ${HCOMM_CODE_COMMON}/debug/profiling/inc/host
    ${HCOMM_CODE_COMMON}/debug/config

    # framework目录下的头文件
    ${HCOMM_CODE_FRAMEWORK}/inc
    ${HCOMM_CODE_FRAMEWORK}/common/src
    ${HCOMM_CODE_FRAMEWORK}/common/src/host
    ${HCOMM_CODE_FRAMEWORK}/common/src/config
    ${HCOMM_CODE_FRAMEWORK}/common/src/topo
    ${HCOMM_CODE_FRAMEWORK}/communicator/impl

    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto
)

set(checker_to_test_list

)

add_library(hccl_alg_analyzer_legacy SHARED ${checker_to_test_list})

add_subdirectory(checker)
add_subdirectory(adapter_v2)

# 需要针对protobuf自动编译生成的文件设置GENERATED属性，否则会编译报错
set_property(SOURCE ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto/analysis_result.pb.cc PROPERTY GENERATED 1)

#头文件搜索路径
target_include_directories(hccl_alg_analyzer_legacy PRIVATE
    ${include_list}
)

target_compile_definitions(hccl_alg_analyzer_legacy PRIVATE
    google=ascend_private
    -DOPEN_HCCL_TEST
    _GLIBCXX_USE_CXX11_ABI=0
    -DHCCL_ALG_ANALYZER_DAVID   #checker v2相关的编译宏
)

target_compile_options(hccl_alg_analyzer_legacy PRIVATE
    -Wall
    -Wfloat-equal
    -U_FORTIFY_SOURCE
    -Os
    -O2
    -pipe
)

target_link_libraries(hccl_alg_analyzer_legacy PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    hccl_alg_adaption_v2
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    ascend_protobuf
    -lrt
    -ldl
)
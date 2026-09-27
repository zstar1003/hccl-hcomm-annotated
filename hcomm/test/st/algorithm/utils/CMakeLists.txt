# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(checker_to_test_list)
add_library(hccl_alg_analyzer SHARED ${checker_to_test_list})

set(UTILS_INNER_INCLUDE_LIST
    ${HCOMM_ALG_UTILS_ROOT}/checker
    ${HCOMM_ALG_UTILS_ROOT}/checker/common/types
    ${HCOMM_ALG_UTILS_ROOT}/checker/rank_info_recorder
    ${HCOMM_ALG_UTILS_ROOT}/checker/mem_layout
    ${HCOMM_ALG_UTILS_ROOT}/checker/link_type_recorder
    ${HCOMM_ALG_UTILS_ROOT}/checker/task_collector
    ${HCOMM_ALG_UTILS_ROOT}/checker/semantics_check
    ${HCOMM_ALG_UTILS_ROOT}/checker/singletask_check
    ${HCOMM_ALG_UTILS_ROOT}/checker/mem_conflict_check
    ${HCOMM_ALG_UTILS_ROOT}/checker/ui/data_dumper
    ${HCOMM_ALG_UTILS_ROOT}/checker/param_trans
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1
    ${HCOMM_ALG_UTILS_ROOT}/adapter_v1/runtime
)

add_subdirectory(checker)
add_subdirectory(adapter_v1)

# 需要针对protobuf自动编译生成的文件设置GENERATED属性，否则会编译报错
set_property(SOURCE ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto/analysis_result.pb.cc PROPERTY GENERATED 1)

#头文件搜索路径
target_include_directories(hccl_alg_analyzer PRIVATE
    ${UTILS_PUBLIC_INCLUDE_LIST}
    ${UTILS_INNER_INCLUDE_LIST}
)

target_compile_definitions(hccl_alg_analyzer PRIVATE
    google=ascend_private
    -DOPEN_HCCL_TEST
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(hccl_alg_analyzer PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O2
    -pipe
)

target_link_libraries(hccl_alg_analyzer PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    hccl_alg_adaption
    -pthread
    ascend_protobuf
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    -lrt
    -ldl
)
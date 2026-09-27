# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_executable(executor_pipeline_hccl_test ${src_to_test_list})

#头文件搜索路径
target_include_directories(executor_pipeline_hccl_test PRIVATE
    ${HCOMM_ALG_TC_ROOT}/testcase

    ${UTILS_PUBLIC_INCLUDE_LIST}
    ${HCOMM_PUBLIC_INCLUDE_LIST}
    ${HCOMM_PLATFORM_INCLUDE_LIST}
    ${HCOMM_ALGORITHM_INCLUDE_LIST}
    ${HCOMM_COMMON_INCLUDE_LIST}
    ${HCOMM_FRAMEWORK_INCLUDE_LIST}
    ${HCOMM_LEGACY_INCLUDE_LIST}

    ${ASCEND_CANN_INCLUDE_LIST}
)

target_compile_definitions(executor_pipeline_hccl_test PRIVATE
    google=ascend_private
    _GLIBCXX_USE_CXX11_ABI=0
    -DOPEN_HCCL_TEST
)

target_compile_options(executor_pipeline_hccl_test PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O0 -g
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(executor_pipeline_hccl_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    hccl_alg_analyzer
    hccl_alg_adaption
    hccl_alg_test
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    ascend_protobuf
    -lrt
    -ldl
)

target_sources(executor_pipeline_hccl_test
    PUBLIC
    main.cc
    ../testcase/testcase_utils.cc
    testcase_all_to_all_A3_pipeline.cc
)

enable_testing()
add_test(NAME executor_pipeline_hccl_test COMMAND executor_pipeline_hccl_test)

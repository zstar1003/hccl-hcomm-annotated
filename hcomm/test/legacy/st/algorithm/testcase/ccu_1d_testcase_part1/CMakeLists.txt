# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_executable(legacy_alg_ccu_1d_testcase_part1 ${test_case_list})

#头文件搜索路径
target_include_directories(legacy_alg_ccu_1d_testcase_part1 PRIVATE
    ${include_list}
    ${HCOMM_DIR}/src/legacy/ascend910/common
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
)

target_compile_definitions(legacy_alg_ccu_1d_testcase_part1 PRIVATE
    google=ascend_private
    -DOPEN_HCCL_TEST
    _GLIBCXX_USE_CXX11_ABI=0
    -DCCL_KERNEL_AICPU
    -DHCCL_ALG_ANALYZER_DAVID
)

target_compile_options(legacy_alg_ccu_1d_testcase_part1 PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O0 -g
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

add_dependencies(legacy_alg_ccu_1d_testcase_part1 hccl_alg_analyzer_legacy hccl_orion_alg_test hccl_alg_adaption_v2)

target_link_libraries(legacy_alg_ccu_1d_testcase_part1 PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    hccl_alg_analyzer_legacy
    hccl_orion_alg_test
    hccl_alg_adaption_v2
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    ascend_protobuf
    -lrt
    -ldl
)

target_sources(legacy_alg_ccu_1d_testcase_part1
        PUBLIC
        main.cc
        testcase_utils.cc
        testcase_all_gather_ccu.cc
        testcase_all2all_ccu.cc
        testcase_all_reduce_ccu.cc
        # testcase_all_gather_ccu_detour.cc
        # testcase_all_reduce_ccu_detour.cc
        testcase_all_gather_v_ccu.cc
        testcase_broadcast_ccu.cc
        ${UTILS_DIR}/common/virtual_topo_stub/virtual_topo_stub.cc
)

enable_testing()
add_test(NAME legacy_alg_ccu_1d_testcase_part1 COMMAND legacy_alg_ccu_1d_testcase_part1)

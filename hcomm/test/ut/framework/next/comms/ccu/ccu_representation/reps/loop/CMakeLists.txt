# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_to_test_list
    main.cc
    ut_ccu_rep_loop_test.cc
)

add_executable(hccl_utest_comms_ccu_rep_loop ${src_to_test_list})

target_include_directories(hccl_utest_comms_ccu_rep_loop PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation/reps/loop
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/exception
    ${HCCL_CODE_ROOT}/pkg_inc/hcomm/ccu
)

target_compile_definitions(hccl_utest_comms_ccu_rep_loop PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    google=ascend_private
    -DOPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_utest_comms_ccu_rep_loop PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_utest_comms_ccu_rep_loop PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    hccl_llt
    -Wl,--whole-archive
    -Wl,--no-whole-archi
    -lgcov
    -lrt
    -ldl
)

add_dependencies(hccl_utest_comms_ccu_rep_loop hccl_llt)

enable_testing()
add_test(NAME hccl_utest_comms_ccu_rep_loop COMMAND hccl_utest_comms_ccu_rep_loop)
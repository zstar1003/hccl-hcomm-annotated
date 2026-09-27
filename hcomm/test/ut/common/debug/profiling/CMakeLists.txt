# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccl_ut_task_exception_list
    main.cc
    ut_task_exception.cc
    ut_task_profiling.cc
    ut_error_message.cc
)
add_executable(hccl_ut_task_exception ${hccl_ut_task_exception_list})

target_include_directories(hccl_ut_task_exception PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
)

target_compile_definitions(hccl_ut_task_exception PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    CCL_LLT
    OPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_ut_task_exception PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

target_link_libraries(hccl_ut_task_exception PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    hccl_llt
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lgcov
    -lrt
    -ldl
)
add_dependencies(hccl_ut_task_exception hccl_llt)

enable_testing()
add_test(NAME hccl_ut_task_exception COMMAND hccl_ut_task_exception)

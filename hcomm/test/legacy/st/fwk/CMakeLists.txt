# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccl_st_fwk SHARED "")

add_subdirectory(env_stub)
add_subdirectory(hccp_stub)
add_subdirectory(rts_stub)
add_subdirectory(hccl_v2)
add_subdirectory(log_stub)
add_subdirectory(context)
add_subdirectory(ranktable)


target_sources(hccl_st_fwk
    PRIVATE
        hccl_st_situation.cc
        hccl_st_test_case.cc
        hccl_st_prepare_verify.cc
)

target_compile_options(hccl_st_fwk PRIVATE
        -std=c++17
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -g
        -O0
)

target_link_libraries(hccl_st_fwk PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    c_sec_static
)
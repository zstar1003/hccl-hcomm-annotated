# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list
    adapter_rts.cc
    adapter_hal.cc
    adapter_tdt.cc
    adapter_hccp.cc
    adapter_verbs.cc
    adapter_trace.cc
)

# GCC 14 对 std::vector 按值拷贝内联展开后的 __builtin_memmove 存在 -Warray-bounds 误报
# (触发点在 stl_algobase.h，非用户代码真实越界)，对编译 adapter_hccp.cc 的 target 关闭该告警以解除 -Werror 阻断
if(TARGET hccl_plf)
    target_sources(hccl_plf   PRIVATE
        ${src_list}
    )
    target_compile_options(hccl_plf PRIVATE -Wno-array-bounds)
endif()

if(TARGET hccd)
    target_sources(hccd   PRIVATE
        adapter_hal.cc
        adapter_hccp.cc
        adapter_rts.cc
        adapter_tdt.cc
        adapter_verbs.cc
    )
    target_compile_options(hccd PRIVATE -Wno-array-bounds)
endif()

if(TARGET ccl_kernel_plf)
    target_sources(ccl_kernel_plf PRIVATE
        ${src_list}
    )
    target_compile_options(ccl_kernel_plf PRIVATE -Wno-array-bounds)
endif()

if(TARGET ccl_kernel_plf_a)
    target_sources(ccl_kernel_plf_a  PRIVATE
        ${src_list}
    )
    target_compile_options(ccl_kernel_plf_a PRIVATE -Wno-array-bounds)
endif()

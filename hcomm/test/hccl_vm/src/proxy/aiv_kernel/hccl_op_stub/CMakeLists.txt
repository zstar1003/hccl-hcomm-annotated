# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccl_aiv_kernel_base OBJECT
        ${AIV_KERNEL_DIR}/aiv_mode_stub/ai_core_stub.cc
        ${AIV_KERNEL_DIR}/aiv_mode_stub/aiv_task.cc
        ${AIV_KERNEL_DIR}/aiv_mode_stub/aiv_task_json.cc

        ${AIV_KERNEL_DIR}/ascendc_stub/ascendc_base_stub.cc

        ${AIV_KERNEL_DIR}/hccl_op_stub/aiv_model_init.cc
)

target_include_directories(hccl_aiv_kernel_base PRIVATE ${AIV_KERNEL_STUB_INCLUDE_LIST})

set_target_properties(hccl_aiv_kernel_base PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE
)

add_subdirectory(all_gather)
add_subdirectory(all_reduce)
add_subdirectory(all_to_all_v)
add_subdirectory(broadcast)
add_subdirectory(reduce)
add_subdirectory(reduce_scatter)
add_subdirectory(recv)
add_subdirectory(scatter)
add_subdirectory(send)

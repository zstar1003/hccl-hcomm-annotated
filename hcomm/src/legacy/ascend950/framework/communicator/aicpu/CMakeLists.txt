# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(daemon)
add_subdirectory(one_sided_component)

if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        communicator_impl_lite_manager.cc
        communicator_impl_lite.cc
        aicpu_hdc_handler.cc
        kernel_entrance.cc
        aicpu_res_package_helper.cc
        ns_recovery_handler_func.cc
        aicpu_comm_destroy_func.cc
        hdc_lite.cc
        aicpu_mc2_handler.cc
        hccl_mc2_ex.cc
        aicpu_utils.cc
        launch_device.cc
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        communicator_impl_lite_manager.cc
        communicator_impl_lite.cc
        kernel_entrance.cc
        aicpu_res_package_helper.cc
        aicpu_hdc_handler.cc
        hdc_lite.cc
        ns_recovery_handler_func.cc
        aicpu_comm_destroy_func.cc
        aicpu_mc2_handler.cc
        hccl_mc2_ex.cc
        aicpu_utils.cc
    )
endif()

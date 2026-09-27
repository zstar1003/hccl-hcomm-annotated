# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(ccu)

set(src_list
    hcomm_c_adpt.cc
    hcomm_endpoint_c_adpt.cc
    hcomm_mem_c_adpt.cc
    hcomm_channel_c_adpt.cc
    hcomm_thread_c_adpt.cc
    hcomm_engine_ctx_c_adpt.cc
    hccl_comm_socket_c_adpt.cc
)

add_subdirectory(nic_plugin)
 
if(TARGET hcomm_base_obj)
    target_sources(hcomm_base_obj PRIVATE
        ${src_list}
        cpu_primitives_c_adpt.cc
    )
    target_include_directories(hcomm_base_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src/base_comm/resources/endpoint_pairs/channels/aiv
        ${PROJECT_SOURCE_DIR}/src/base_comm/resources/endpoints/dfx
    )
    if(ENABLE_EXPERIMENTAL)
        target_compile_definitions(hcomm_base_obj PRIVATE ENABLE_EXPERIMENTAL)
        target_include_directories(hcomm_base_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/nic_plugin
        )
    endif()
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        aicpu_ts_primitives_c_adpt.cc
        aicpu_ts_sync_data_c_adpt.cc
        aicpu_ts_diag_c_adapt.cc
        plf_debug_config_aicpu.cc
        hcomm_exception_mgr_c_adpt.cc
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/nic_plugin
    )
endif()

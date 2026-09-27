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
    roce_reged_mem_mgr.cc
    aicpu_ts_roce_reged_mem_mgr.cc
    ub_reged_mem_mgr.cc
    ub_mem_reged_mem_mgr.cc
    hccs_reged_mem_mgr.cc
    proc_reged_mem_mgr_cache.cc
)

if(TARGET hcomm_base_obj)
    target_sources(hcomm_base_obj PRIVATE
        ${src_list}
    )
    target_include_directories(hcomm_base_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
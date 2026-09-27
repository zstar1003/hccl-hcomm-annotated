# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# HCCL 层：HcclTeam 进程级管理器、UB Memory Window 控制面及 C 适配
# 依赖方向：依赖同层 hcomm 子目录和 base_comm，不反向被 hcomm/base_comm 依赖
set(src_list
    hccl_team_mgr.cc
    hccl_team_c_adpt.cc
    ubmem_symmetric_memory.cc
    ubmem_symmetric_memory_agent.cc
)

if(TARGET hcomm)
    target_sources(hcomm PRIVATE
        ${src_list}
    )
    target_include_directories(hcomm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

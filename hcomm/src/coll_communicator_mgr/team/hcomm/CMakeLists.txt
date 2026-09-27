# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# HCOMM 层：HcommTeam 资源管理器 + C 适配
# 依赖方向：仅依赖同层或更下层（base_comm/include/外部头），不反向依赖 hccl 子目录
set(src_list
    hcomm_team_c_adpt.cc
    hcomm_team_mgr.cc
)

if(TARGET hcomm)
    target_sources(hcomm PRIVATE
        ${src_list}
    )
    target_include_directories(hcomm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccl_aiv_scatter_op OBJECT
        aiv_communication_v2.cc
)

target_include_directories(hccl_aiv_scatter_op PRIVATE
        ${HCCL_CODE_ROOT}/src/ops/scatter/algorithm/template/aiv/kernel # hccl算子源码
)

target_include_directories(hccl_aiv_scatter_op PRIVATE
        ${AIV_KERNEL_STUB_INCLUDE_LIST} # AscendC & AIV stub
)

set_target_properties(hccl_aiv_scatter_op PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE
)

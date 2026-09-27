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
    rma_conn_lite.cc
    ub_conn_lite.cc
    ub_conn_lite_mgr.cc
    rdma_conn_lite_v2.cc
)

if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        ${src_list}
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${src_list}
    )
endif()

if(TARGET hcomm)
    target_sources(hcomm PRIVATE
        ${src_list}
    )
    target_include_directories(hcomm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
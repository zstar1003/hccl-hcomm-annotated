# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(TARGET hccl_v2)
    target_sources(hccl_v2
            PRIVATE
            buffer.cc
            local_ipc_rma_buffer.cc
            local_rdma_rma_buffer.cc
            local_ub_rma_buffer.cc
            remote_rma_buffer.cc
            dev_buffer.cc
            host_buffer.cc
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
            buffer.cc
    )
endif()

add_subdirectory(aicpu)
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(srcs
    aicpu_kfc_process.cc
    aicpu_kfc_deprecated_process.cc
    aicpu_kfc_batchwrite_process.cc
    aicpu_kfc_retry_process.cc
    aicpu_kfc_rpc_server.cc
    aicpu_kfc_rpc_serverv2.cc
    aicpu_kfc_prof.cc
)

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE ${srcs})
endif()

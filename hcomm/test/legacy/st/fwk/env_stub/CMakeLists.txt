# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

target_include_directories(hccl_st_fwk PRIVATE
        ${TOP_DIR}/abl/qos
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/inc/aicpu/tsd
        ${TOP_DIR}/inc/aicpu
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/inc/external
)


target_sources(hccl_st_fwk
        PRIVATE
        qos_stub.cc
        secure_stub.cc
        tsd_stub.cc
        driver_stub.cc    
)
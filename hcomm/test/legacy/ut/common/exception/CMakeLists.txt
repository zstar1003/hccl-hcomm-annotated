# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

target_sources(hccl_v2_utest 
    PUBLIC
    #TODO 暂时关闭此用例：主要功能为比对错误信息，等错误信息稳定后再打开。
    #ut_hccl_exception.cc
    ut_trace.cc
    
    ${HCCL_V2_CODE_ROOT}/common/exception/hccl_exception.cc
    ${HCCL_V2_CODE_ROOT}/common/exception/exception_definition.cc
)
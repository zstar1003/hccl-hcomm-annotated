/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "ccu_datatype.h"
#include "ccu_context_alg_base.h"

namespace Hccl {

void CcuContextAlgBase::GroupBroadcastV2(
    const std::vector<CcuTransport*>& transports, const std::vector<CcuRep::Memory>& dst, const CcuRep::Memory& src,
    const GroupOpSizeV2& goSize) const
{
    (void)transports;
    (void)dst;
    (void)src;
    (void)goSize;
    THROW<NotSupportException>("Unimplemented Interface: %s", __func__);
}

void CcuContextAlgBase::GroupReduceV2(
    const std::vector<CcuTransport*>& transports, const CcuRep::Memory& dst, const std::vector<CcuRep::Memory>& src,
    const GroupOpSizeV2& goSize, DataType dataType, DataType outputDataType, ReduceOp reduceType) const
{
    (void)transports;
    (void)dst;
    (void)src;
    (void)goSize;
    (void)dataType;
    (void)outputDataType;
    (void)reduceType;
    THROW<NotSupportException>("Unimplemented Interface: %s", __func__);
}
} // namespace Hccl

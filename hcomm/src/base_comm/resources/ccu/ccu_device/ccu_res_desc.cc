/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res_desc.h"

namespace hcomm {
static_assert(
    CCU_RES_TYPE_COUNT == static_cast<size_t>(ResType::__COUNT__), "CCU_RES_TYPE_COUNT must match ResType::__COUNT__");

CcuResult CcuResDesc::SetResNum(ResType resType, uint32_t num)
{
    if (!IsValidResType(resType)) {
        return CcuResult::CCU_E_PARA;
    }

    resNum[ResTypeIndex(resType)] = num;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuResDesc::QueryResNum(ResType resType, uint32_t& num) const
{
    if (!IsValidResType(resType)) {
        return CcuResult::CCU_E_PARA;
    }

    num = resNum[ResTypeIndex(resType)];
    return CcuResult::CCU_SUCCESS;
}

bool CcuResDesc::IsValidResType(ResType resType) { return resType < ResType::__COUNT__; }

size_t CcuResDesc::ResTypeIndex(ResType resType) { return static_cast<size_t>(static_cast<ResType::Value>(resType)); }
} // namespace hcomm

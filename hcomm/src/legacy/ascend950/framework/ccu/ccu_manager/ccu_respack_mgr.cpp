/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_respack_mgr.h"
#include "internal_exception.h"

namespace Hccl {

// 新增空的CcuResPack，同时内部做标记，用于回退
void CcuResPackMgr::PrepareAlloc(u32 size)
{
    if (resPacks.size() < size) {
        unConfirmedNum = size - resPacks.size();
        resPacks.resize(size);
    }

    HCCL_INFO(
        "[CcuResPackMgr][PrepareAlloc] CcuResPack need size[%u], current resPacks size[%zu], unConfirmedNum[%u]", size,
        resPacks.size(), unConfirmedNum);
}

// 确认新增的资源，未来不可回退（对之前调用PrepareXxx()新增的资源，取消标记）
void CcuResPackMgr::Confirm() { unConfirmedNum = 0; }

// 对新增的资源进行回退
void CcuResPackMgr::Fallback()
{
    for (u32 idx = 0; idx < unConfirmedNum; ++idx) {
        if (resPacks[idx].handles.size() != 0) {
            HCCL_WARNING(
                "[CcuResPackMgr][Fallback]resPacks idx[%u] has handles size[%u], not empty", idx,
                resPacks[idx].handles.size());
        }
        resPacks.pop_back();
    }
    unConfirmedNum = 0;
}

// 获取idx对应的CcuResPack
CcuResPack& CcuResPackMgr::GetCcuResPack(u32 idx)
{
    u32 size = resPacks.size();
    if (size <= idx) {
        THROW<InternalException>(
            StringFormat("[CcuResPackMgr][GetCcuResPack] idx[%u] is bigger than resPacks size[%u]", idx, size));
    }
    return resPacks[idx];
}

} // namespace Hccl

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_adapter_rts.h"

#include "log.h"

namespace hcomm {

HcclResult RtsUbDevQueryInfo(const rtUbDevQueryCmd cmd, rtMemUbTokenInfo& devInfo)
{
    if (cmd != QUERY_PROCESS_TOKEN) {
        HCCL_ERROR("[%s] error cmd[%d].", __func__, cmd);
        return HcclResult::HCCL_E_PARA;
    }

    auto ret = rtUbDevQueryInfo(cmd, &devInfo);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s] failed[%d], va[0x%llx] size[%llu].", __func__, ret, devInfo.va, devInfo.size);
        return HcclResult::HCCL_E_RUNTIME;
    }

    constexpr u32 TOKEN_ID_RIGHT_SHIF = 8;
    devInfo.tokenId = devInfo.tokenId >> TOKEN_ID_RIGHT_SHIF;
    return HcclResult::HCCL_SUCCESS;
}

} // namespace hcomm

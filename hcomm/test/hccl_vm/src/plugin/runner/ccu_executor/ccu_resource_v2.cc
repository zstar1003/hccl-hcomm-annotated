/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "ccu_microcode_common_v2.h"
#include "ccu_resource_manager.h"

void CcuResourceV2::Reset()
{
    for (int i = 0; i < HcclSim::DIE_NUM; i++) {
        xn_[i].fill(0);
        gsa_[i].fill(0);
        cke_[i].fill(0);
        std::fill(ms_[i].begin(), ms_[i].end(), 0);
        ccuTaskInfos_[i] = {};
        instrSpace_[i] = {};
        simulators_[i].reset();
    }
    channelId2RmtRankMap_ = {};
}

CcuResourceV2::CcuResourceV2(int rankId, uint32_t rankSize)
{
    rankId_ = rankId;
    rankSize_ = rankSize;
    for (int i = 0; i < HcclSim::DIE_NUM; i++) {
        ms_[i].resize(SimCcuV2::CCU_RESOURCE_MS_SIZE);
    }
}

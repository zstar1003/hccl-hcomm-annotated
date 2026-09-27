/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_eid_info.h"

#include "hccl_common_v2.h"
#include "orion_adapter_rts.h"

namespace Hccl {

CcuEidInfo& CcuEidInfo::GetInstance(int32_t logicDeviceId)
{
    static CcuEidInfo ccuEidInfo[MAX_MODULE_DEVICE_NUM + 1];

    if (static_cast<u32>(logicDeviceId) > MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[CcuEidInfo] GetInstance failed, logicDeviceId=%d, ret=%d", logicDeviceId, HCCL_E_PARA);
        return ccuEidInfo[0];
    }

    return ccuEidInfo[logicDeviceId];
}

CcuEidInfo::CcuEidInfo() {}

CcuEidInfo::~CcuEidInfo() { initflag_ = false; }

HcclResult CcuEidInfo::GetEidInfo(int32_t logicDeviceId, std::vector<HrtDevEidInfo>& eidInfo)
{
    if (!initflag_) {
        HRaInfo info(HrtNetworkMode::HDC, HrtGetDevicePhyIdByUserDevId(logicDeviceId));
        vector<HrtDevEidInfo> eidInfoList = HrtRaGetDevEidInfoList(info);

        if (eidInfoList.empty()) {
            HCCL_WARNING("[GetEidInfo] Get EidInfo failed, logicDeviceId=%d", logicDeviceId);
            return HCCL_E_DRV;
        }

        eidInfoList_.assign(eidInfoList.begin(), eidInfoList.end());
        initflag_ = true;
    }

    if (eidInfoList_.empty()) {
        HCCL_WARNING("[GetEidInfo] EidInfo is empty, logicDeviceId=%d", logicDeviceId);
        return HCCL_E_DRV;
    }

    eidInfo.assign(eidInfoList_.begin(), eidInfoList_.end());

    HCCL_INFO("[GetEidInfo] Get EidInfo success, logicDeviceId=%d, eidInfo size=%u", logicDeviceId, eidInfo.size());

    return HCCL_SUCCESS;
}

}; // namespace Hccl

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EID_INFO_MGR_H
#define EID_INFO_MGR_H

#include <mutex>
#include <unordered_map>

#include "hccl_res.h"
#include "hccl_types.h"
#include "hcomm_adapter_hccp.h"

// 暂时使用orion ipaddr
#include "ip_address.h"

namespace hcomm {

class EidInfoMgr {
public:
    static EidInfoMgr& GetInstance(const uint32_t devicePhyId);
    HcclResult GetEidInfos(std::vector<DevEidInfo>& eidInfos);
    HcclResult GetEidInfoByAddr(const CommAddr& commAddr, DevEidInfo& eidInfo);

private:
    explicit EidInfoMgr() = default;
    ~EidInfoMgr() = default;
    EidInfoMgr(const EidInfoMgr& that) = delete;
    EidInfoMgr& operator=(const EidInfoMgr& that) = delete;

    HcclResult Init();

private:
    uint32_t devPhyId_{0};
    bool initflag_{false};
    std::mutex innerMutex_{};

    std::vector<DevEidInfo> eidInfos_{};
    std::unordered_map<Hccl::IpAddress, DevEidInfo> eidInfoMap_{};
};

}; // namespace hcomm

#endif // EID_INFO_MGR_H

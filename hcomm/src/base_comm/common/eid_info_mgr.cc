/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "eid_info_mgr.h"

#include "hccl_common.h"
#include "orion_adpt_utils.h"

namespace hcomm {

EidInfoMgr& EidInfoMgr::GetInstance(const uint32_t devicePhyId)
{
    static EidInfoMgr eidInfoMgr[MAX_MODULE_DEVICE_NUM + 1];

    uint32_t devPhyId = devicePhyId;
    if (devPhyId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[EidInfoMgr][%s] use the backup device, devPhyId[%u] should be "
            "less than %u.",
            __func__, devPhyId, MAX_MODULE_DEVICE_NUM);
        devPhyId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }

    eidInfoMgr[devPhyId].devPhyId_ = devPhyId;
    return eidInfoMgr[devPhyId];
}

HcclResult EidInfoMgr::Init()
{
    const RaInfo raInfo{NetworkMode::NETWORK_OFFLINE, devPhyId_};
    CHK_RET(RaGetDevEidInfos(raInfo, eidInfos_));

    initflag_ = true;
    if (eidInfos_.empty()) {
        HCCL_ERROR(
            "[EidInfoMgr][%s] failed to find any eid info, "
            "devPhyId[%u].",
            __func__, devPhyId_);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    for (const auto& eidInfo : eidInfos_) {
        Hccl::IpAddress ipAddr{}; // 暂时使用orion ip addr 用作索引
        CHK_RET(CommAddrToIpAddress(eidInfo.commAddr, ipAddr));
        eidInfoMap_[ipAddr] = eidInfo;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult EidInfoMgr::GetEidInfos(std::vector<DevEidInfo>& eidInfos)
{
    std::unique_lock<std::mutex> lock(innerMutex_);
    if (!initflag_) {
        CHK_RET(Init());
    }

    // 不允许外部修改eidInfo，传递拷贝结果
    eidInfos.assign(eidInfos_.begin(), eidInfos_.end());

    HCCL_INFO("[EidInfoMgr][%s] found %zu eid info, devPhyId[%u].", __func__, eidInfos.size(), devPhyId_);

    return HCCL_SUCCESS;
}

HcclResult EidInfoMgr::GetEidInfoByAddr(const CommAddr& commAddr, DevEidInfo& eidInfo)
{
    std::unique_lock<std::mutex> lock(innerMutex_);
    if (!initflag_) {
        CHK_RET(Init());
    }

    Hccl::IpAddress ipAddr{}; // 暂时使用orion ipaddr 用作索引
    CHK_RET(CommAddrToIpAddress(commAddr, ipAddr));
    // 兼容上层传递ipv4查询eid
    const auto eidAddr = Hccl::IpAddress(ipAddr.GetEid());

    const auto addrIter = eidInfoMap_.find(eidAddr);
    if (addrIter == eidInfoMap_.end()) {
        HCCL_ERROR(
            "[EidInfoMgr][%s] failed to find eid info by ip addr[%s], "
            "devPhyId[%u].",
            __func__, ipAddr.Describe().c_str(), devPhyId_);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    eidInfo = addrIter->second; // 触发深拷贝，避免外部修改值
    return HcclResult::HCCL_SUCCESS;
}

}; // namespace hcomm

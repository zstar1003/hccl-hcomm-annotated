/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_tlv_hdc_mgr.h"

#include "hccl_common.h"

#include "hccp_tlv.h"

namespace hcomm {

HccpTlvHdcMgr& HccpTlvHdcMgr::GetInstance(const uint32_t devicePhyId)
{
    static HccpTlvHdcMgr hccpTlvHdcMgr[MAX_MODULE_DEVICE_NUM + 1];

    uint32_t devPhyId = devicePhyId;
    if (devPhyId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[HccpTlvHdcMgr][%s] use the backup device, devPhyId[%u] should be "
            "less than %u.",
            __func__, devPhyId, MAX_MODULE_DEVICE_NUM);
        devPhyId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }

    hccpTlvHdcMgr[devPhyId].devPhyId_ = devPhyId;
    return hccpTlvHdcMgr[devPhyId];
}

inline TlvInitInfo GetCfgInfo(const uint32_t devPhyId)
{
    constexpr u32 tlvVersion = 1;
    struct TlvInitInfo tlvInfo {};
    tlvInfo.phyId = devPhyId;
    tlvInfo.nicPosition = NetworkMode::NETWORK_OFFLINE;
    tlvInfo.version = tlvVersion;

    return tlvInfo;
}

static HcclResult HccpTlvInit(const uint32_t devPhyId, TlvHandle& tlvHandle)
{
    TlvInitInfo cfgInfo = GetCfgInfo(devPhyId);
    unsigned int bufferSize{0}; // 当前未使用

    int32_t ret = RaTlvInit(&cfgInfo, &bufferSize, &tlvHandle);
    if (ret != 0 || tlvHandle == nullptr) {
        HCCL_ERROR(
            "[Init][RaTlv]errNo[0x%016llx] ra tlv init fail. params: mode[%u]. return: ret[%d]",
            static_cast<unsigned long long>(HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK)), cfgInfo.nicPosition, ret);
        return HcclResult::HCCL_E_NETWORK;
    }

    HCCL_INFO("[%s] success, device id[%u] tlv handle[%p]", __func__, cfgInfo.phyId, tlvHandle);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HccpTlvHdcMgr::Init()
{
    std::unique_lock<std::mutex> lock(innerMutex_);
    if (initFlag_) {
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(HccpTlvInit(devPhyId_, tlvHandle_));
    initFlag_ = true;
    return HcclResult::HCCL_SUCCESS;
}

TlvHandle HccpTlvHdcMgr::GetHandle() { return tlvHandle_; }

static HcclResult HccpTlvDeinit(const TlvHandle tlvHandle)
{
    int32_t ret = RaTlvDeinit(tlvHandle);
    if (ret != 0) {
        HCCL_ERROR(
            "[DeInit][RaTlv]errNo[0x%016llx] ra tlv deinit fail. return: ret[%d]",
            static_cast<unsigned long long>(HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK)), ret);
        return HcclResult::HCCL_E_NETWORK;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult HccpTlvHdcMgr::Deinit()
{
    std::unique_lock<std::mutex> lock(innerMutex_);
    if (!initFlag_) {
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(HccpTlvDeinit(tlvHandle_));
    tlvHandle_ = nullptr;
    initFlag_ = false;
    return HcclResult::HCCL_SUCCESS;
}

HccpTlvHdcMgr::~HccpTlvHdcMgr() { (void)Deinit(); }

} // namespace hcomm

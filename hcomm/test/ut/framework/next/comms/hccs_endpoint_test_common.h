/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCS_ENDPOINT_TEST_COMMON_H
#define HCCS_ENDPOINT_TEST_COMMON_H

namespace {
static s32 deviceCurLogicId_ = 0;
static s32 deviceCurPhyId_ = 0;

void StubSetDevice(s32 deviceLogicId)
{
    deviceCurLogicId_ = deviceLogicId;
    deviceCurPhyId_ = deviceLogicId;
}

HcclResult StubHrtGetDevice(s32* deviceLogicId)
{
    if (deviceLogicId != nullptr) {
        *deviceLogicId = deviceCurLogicId_;
    }
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDeviceRefresh(s32* deviceLogicId)
{
    if (deviceLogicId != nullptr) {
        *deviceLogicId = deviceCurLogicId_;
    }
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDevicePhyIdByUserDevId(u32 deviceLogicId, u32& devicePhyId, bool isRefresh)
{
    devicePhyId = deviceLogicId;
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDeviceIndexByPhyId(u32 devicePhyId, u32& deviceLogicId)
{
    deviceLogicId = devicePhyId;
    return HCCL_SUCCESS;
}

HcclResult StubHcclSocketAcceptForEp(
    hccl::HcclSocket* /*self*/, const std::string& /*tag*/, std::shared_ptr<hccl::HcclSocket>& socket,
    u32 /*acceptTimeOut*/)
{
    socket = std::make_shared<hccl::HcclSocket>(static_cast<HcclNetDevCtx>(nullptr), 16666);
    return HCCL_SUCCESS;
}

HcclResult StubGetDeviceVnicIP(u32 devicePhyId, u32 superDeviceId, hccl::HcclIpAddress& vnicIP)
{
    std::string ip = "127.0.0." + std::to_string(devicePhyId + 1);
    (void)vnicIP.SetReadableAddress(ip);
    return HCCL_SUCCESS;
}

HcclResult StubHcclNetOpenDev(
    HcclNetDevCtx* netDevCtx, NicType nicType, s32 devicePhyId, s32 deviceLogicId, hccl::HcclIpAddress localIp,
    hccl::HcclIpAddress backupIp)
{
    static hccl::NetDevContext kNetDevCtx[MAX_MODULE_DEVICE_NUM];
    static bool initialized[MAX_MODULE_DEVICE_NUM] = {false};
    if (!initialized[devicePhyId]) {
        hccl::HcclIpAddress localIp;
        std::string ip = "127.0.0." + std::to_string(devicePhyId + 1);
        (void)localIp.SetReadableAddress(ip);
        kNetDevCtx[devicePhyId].Init(NicType::VNIC_TYPE, 0, 0, localIp);
        initialized[devicePhyId] = true;
    }
    *netDevCtx = reinterpret_cast<HcclNetDev>(&kNetDevCtx[devicePhyId]);
    return HCCL_SUCCESS;
}

void StubHcclNetCloseDev(HcclNetDevCtx netDevCtx) {}
} // namespace

#endif // HCCS_ENDPOINT_TEST_COMMON_H

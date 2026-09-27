/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_connection.h"
#include "adapter_rts.h"
#include "rdma_handle_manager.h"

namespace Hccl {

CcuConnection::CcuConnection(
    const IpAddress& locAddr, const IpAddress& rmtAddr, const CcuChannelInfo& channelInfo,
    const std::vector<CcuJetty*>& ccuJettys)
    : locAddr_(locAddr),
      rmtAddr_(rmtAddr),
      channelInfo_(channelInfo),
      ccuJettys_(ccuJettys)
{}

CcuTpConnection::CcuTpConnection(
    const IpAddress& locAddr, const IpAddress& rmtAddr, const CcuChannelInfo& channelInfo,
    const std::vector<CcuJetty*>& ccuJettys)
    : CcuConnection(locAddr, rmtAddr, channelInfo, ccuJettys)
{
    tpProtocol = TpProtocol::TP;
}

CcuCtpConnection::CcuCtpConnection(
    const IpAddress& locAddr, const IpAddress& rmtAddr, const CcuChannelInfo& channelInfo,
    const std::vector<CcuJetty*>& ccuJettys)
    : CcuConnection(locAddr, rmtAddr, channelInfo, ccuJettys)
{
    tpProtocol = TpProtocol::CTP;
}

HcclResult CcuConnection::Init()
{
    devLogicId = HrtGetDevice();
    uint32_t devPhyId = HrtGetDevicePhyIdByUserDevId(devLogicId);

    auto& rdmaHandleMgr = RdmaHandleManager::GetInstance();
    rdmaHandle = rdmaHandleMgr.GetByIp(devPhyId, locAddr_);
    auto dieIdAndFuncId = HraGetDieAndFuncId(rdmaHandle);
    dieId = channelInfo_.dieId;
    status = CcuConnStatus::INIT;
    innerStatus = InnerStatus::INIT;
    return HcclResult::HCCL_SUCCESS;
}

IpAddress CcuConnection::GetLocAddr() { return locAddr_; }

IpAddress CcuConnection::GetRmtAddr() { return rmtAddr_; }

HcclResult CcuConnection::ReleaseConnRes() { return HcclResult::HCCL_SUCCESS; }

CcuConnection::~CcuConnection() {}

uint32_t CcuConnection::GetChannelId() const { return channelInfo_.channelId; }

int32_t CcuConnection::GetDevLogicId() const { return devLogicId; }

uint32_t CcuConnection::GetDieId() const { return channelInfo_.dieId; }

} // namespace Hccl

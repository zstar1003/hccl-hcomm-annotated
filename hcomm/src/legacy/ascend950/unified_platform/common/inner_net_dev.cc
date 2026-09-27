/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inner_net_dev.h"
#include "null_ptr_exception.h"
#include "exception_util.h"
#include "network_api_exception.h"

namespace Hccl {
InnerNetDev::InnerNetDev(const NetDevInfo& info)
{
    IpAddress localIp = info.addr;

    RaInterface intf{};
    intf.address = localIp;
    intf.phyId = info.devId;

    HCCL_DEBUG("InnerNetDev::Init, devPhyId[%u]", info.devId);

    localProto_ = info.protoType;
    if (info.type == PortDeploymentType::HOST_NET) {
        netMode_ = HrtNetworkMode::PEER;
    }

    try {
        if (localProto_ == LinkProtoType::RDMA) {
            rdmaHandle_ = HrtRaRdmaInit(netMode_, intf);
            auto dieAndFuncId = HraGetDieAndFuncId(rdmaHandle_);
            dieId_ = dieAndFuncId.first;
            funcId_ = dieAndFuncId.second;

            auto tokenIdHandlePair = RaUbAllocTokenIdHandle(rdmaHandle_);
            tokenHandle_ = tokenIdHandlePair.first;
            tokenId_ = tokenIdHandlePair.second;
        } else if (localProto_ == LinkProtoType::UB) {
            HrtRaUbCtxInitParam in(HrtNetworkMode::HDC, info.devId, localIp);
            rdmaHandle_ = HrtRaUbCtxInit(in);
            tokenInfoManager_ = make_unique<TokenInfoManager>(info.devId, rdmaHandle_);
        }
        isValid_ = true;
    } catch (const NetworkApiException& e) {
        HCCL_ERROR("%s", e.what());
        isValid_ = false;
    }
}

JfcHandle InnerNetDev::getUbJfcHandle(HrtUbJfcMode jfcMode)
{
    if (rdmaHandle_ == nullptr) {
        THROW<NullPtrException>("[InnerNetDev::%s] rdmaHandle_ is nullptr", __func__);
    }
    CqCreateInfo cqInfo;
    ubJfcHandle_ = HrtRaUbCreateJfc(rdmaHandle_, cqInfo, jfcMode);
    return ubJfcHandle_;
}

std::pair<TokenIdHandle, uint32_t> InnerNetDev::getTokenIdInfo(const BufferKey<uintptr_t, u64>& bufKey)
{
    if (tokenInfoManager_ == nullptr) {
        THROW<NullPtrException>("[InnerNetDev::%s] tokenInfoManager_ is nullptr", __func__);
    }

    return tokenInfoManager_->GetTokenInfo(bufKey);
}

void InnerNetDev::putTokenIdInfo(const BufferKey<uintptr_t, u64>& bufKey, TokenIdHandle tokenIdHandle)
{
    if (tokenInfoManager_ == nullptr) {
        HCCL_WARNING("[InnerNetDev::%s]tokenInfoManager_ is nullptr", __func__);
        return;
    }
    tokenInfoManager_->PutTokenInfo(bufKey, tokenIdHandle);
}

InnerNetDev::~InnerNetDev()
{
    if (ubJfcHandle_ != 0) {
        DECTOR_TRY_CATCH("jfc handle destroy", HrtRaUbDestroyJfc(rdmaHandle_, ubJfcHandle_));
    }
    if (localProto_ == LinkProtoType::RDMA) {
        if (tokenHandle_ != 0) {
            RaUbFreeTokenIdHandle(rdmaHandle_, tokenHandle_);
        }
        if (rdmaHandle_ != nullptr) {
            HrtRaRdmaDeInit(rdmaHandle_, netMode_);
        }
    } else if (localProto_ == LinkProtoType::UB) {
        if (tokenInfoManager_ != nullptr) {
            tokenInfoManager_->Destroy();
        }
        if (rdmaHandle_ != nullptr) {
            HrtRaUbCtxDestroy(rdmaHandle_);
        }
    }
}

} // namespace Hccl

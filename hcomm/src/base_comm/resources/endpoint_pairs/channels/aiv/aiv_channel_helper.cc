/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "aiv_channel_helper.h"
#include "channel_process.h"
#include "channel.h"
#include "aicpu_ts_roce_channel_v2.h"
#include "aiv_urma_channel.h"
#include "comm_engine_utils.h"

using namespace hcomm;

HcclResult AivChannelHelper::FillDevEntities(
    const ChannelHandle* channelList, uint32_t listNum, const HcommChannelDesc* channelDescs,
    const int32_t* linkStatusList)
{
    CHK_PTR_NULL(channelList);
    CHK_PTR_NULL(linkStatusList);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s] Invalid listNum, listNum[%u]", __func__, listNum), HCCL_E_PARA);

    for (uint32_t i = 0; i < listNum; i++) {
        if (linkStatusList[i] != HCOMM_CHANNEL_STATUS_READY) {
            continue;
        }
        CommProtocol protocol = channelDescs[i].remoteEndpoint.protocol;
        void* channelPtr = nullptr;
        CHK_RET(ChannelProcess::ChannelGet(channelList[i], &channelPtr));
        CHK_PTR_NULL(channelPtr);
        auto* channel = static_cast<Channel*>(channelPtr);
        if (channel->IsDeviceEntityReady()) {
            continue;
        }

        if (protocol == COMM_PROTOCOL_ROCE) {
            auto* roceChannel = static_cast<AicpuTsRoceChannelV2*>(channelPtr);
            CHK_RET(roceChannel->FillDevChannelEntity());
        }

        if (protocol == COMM_PROTOCOL_UB_CTP || protocol == COMM_PROTOCOL_UBC_TP || protocol == COMM_PROTOCOL_UB_RTP) {
            auto* aivChannel = static_cast<AivUrmaChannel*>(channelPtr);
            CHK_RET(aivChannel->FillChannelEntityToDevice());
        }

        channel->SetDeviceEntityReady();
        HCCL_INFO("[%s] channel[%u] fill dev entity success.", __func__, i);
    }
    return HCCL_SUCCESS;
}

HcclResult AivChannelHelper::HandleStatus(
    const ChannelHandle* channelList, uint32_t listNum, const HcommChannelDesc* channelDescs,
    const std::vector<int32_t>& linkStatusList, int32_t* statusList)
{
    HcclResult fillRet = FillDevEntities(channelList, listNum, channelDescs, linkStatusList.data());
    if (fillRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] FillDevEntities failed, ret[%d]", __func__, fillRet);
        return HCCL_E_INTERNAL;
    }
    for (uint32_t i = 0; i < listNum; i++) {
        statusList[i] = linkStatusList[i];
    }
    return HCCL_SUCCESS;
}

HcclResult AivChannelHelper::PreAllocChannels(
    ChannelHandle* targetChannels, ChannelHandle* userChannels, const HcommChannelDesc* channelDescs,
    uint32_t channelNum)
{
    CHK_PTR_NULL(targetChannels);
    CHK_PTR_NULL(userChannels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s] Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);

    bool needD2HMap = false;
    for (uint32_t i = 0; i < channelNum; i++) {
        CommProtocol protocol = channelDescs[i].remoteEndpoint.protocol;

        if (protocol == COMM_PROTOCOL_ROCE) {
            needD2HMap = true;
            auto* channel = ReinterpretAs<AicpuTsRoceChannelV2*>(targetChannels[i]);
            CHK_PTR_NULL(channel);
            CHK_RET(channel->PreAllocDevChannelEntity(&userChannels[i]));
            HCCL_INFO(
                "[%s] channel[%u] pre-alloc dev entity success, devEntityPtr[%p]", __func__, i,
                ReinterpretAs<void*>(static_cast<uintptr_t>(userChannels[i])));
        } else if (
            protocol == COMM_PROTOCOL_UB_CTP || protocol == COMM_PROTOCOL_UBC_TP || protocol == COMM_PROTOCOL_UB_RTP) {
            needD2HMap = true;
            auto* channel = ReinterpretAs<AivUrmaChannel*>(targetChannels[i]);
            CHK_PTR_NULL(channel);

            void* devChannelEntity = nullptr;
            HcclResult ret = channel->PreAllocChannelEntityToDevice(&devChannelEntity);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[%s] channel[%u] PreAllocChannelEntityToDevice failed, ret[%d]", __func__, i, ret), ret);
            CHK_PTR_NULL(devChannelEntity);
            userChannels[i] = static_cast<ChannelHandle>(ReinterpretAs<uintptr_t>(devChannelEntity));
            HCCL_INFO(
                "[%s] channel[%u] pre-alloc dev entity success, devEntityPtr[%p]", __func__, i,
                ReinterpretAs<void*>(static_cast<uintptr_t>(userChannels[i])));
        } else {
            userChannels[i] = targetChannels[i];
            HCCL_INFO(
                "[%s] AIV engine channel protocol not supported pre-alloc, idx[%u], protocol[%d]. "
                "Return host channel handle.",
                __func__, i, static_cast<int>(protocol));
        }
    }

    if (needD2HMap) {
        CHK_RET(ChannelProcess::FillChannelD2HMap(userChannels, targetChannels, channelNum));
    }
    return HCCL_SUCCESS;
}

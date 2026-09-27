/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_res.h"
#include "log.h"
#include "hccl_comm_pub.h"
#include "independent_op.h"
#include "channel_manager.h"
#include "hcomm_c_adpt.h"
#include "param_check_pub.h"
#include "hccl_one_sided_conn.h"
#include <array>
#include <vector>

using namespace hccl;

// [中文导读] 查询通道可用的通知槽位数量，不创建通知、不发送通知；V2和兼容域由不同实现提供结果。
HcclResult HcclChannelGetNotifyNum(HcclComm comm, ChannelHandle channel, uint32_t* notifyNum)
{
    CHK_PTR_NULL(notifyNum);
    CHK_PTR_NULL(comm);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        ret = static_cast<HcclResult>(HcommChannelGetNotifyNum(channel, notifyNum));
    } else {
        auto& channelMgr = hcclComm->GetIndependentOp().GetChannelManager();
        ret = channelMgr.ChannelCommGetNotifyNum(channel, notifyNum);
    }

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to get channel notifyNum, group[%s], channel[%llu], ret[%d]", __func__,
            hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), ret);
        return ret;
    }

    HCCL_RUN_INFO(
        "[%s] get channel notifyNum success, group[%s], channel[%llu], notifyNum[%u], ret[%d]", __func__,
        hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), *notifyNum, ret);
    return HCCL_SUCCESS;
}

HcclResult CommChannelDestroy(HcclComm comm, ChannelHandle* channelList, uint32_t channelNum)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(channelList);
    CHK_PRT_RET(
        channelNum == 0, HCCL_ERROR("[%s]Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        ChannelManager* channelMgr = collComm->GetChannelManager();
        CHK_PTR_NULL(channelMgr);
        ret = channelMgr->ChannelCommDestroy(channelList, channelNum);
    } else {
        auto& channelMgr = hcclComm->GetIndependentOp().GetChannelManager();
        ret = channelMgr.ChannelCommDestroy(channelList, channelNum);
    }

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to destroy channel, group[%s], channelList[%p], channelNum[%u], ret[%d]", __func__,
            hcclComm->GetIdentifier().c_str(), channelList, channelNum, ret);
        return ret;
    }

    HCCL_RUN_INFO(
        "[%s] destroy channel success, group[%s], channelList[%p], channelNum[%u], ret[%d]", __func__,
        hcclComm->GetIdentifier().c_str(), channelList, channelNum, ret);
    return HCCL_SUCCESS;
}

// [中文导读] 取得此Channel对应Peer的CCL区地址和容量，供算法构造远端读写切片。
// [中文导读] 返回的是远端内存描述，不是把对端CCL内容复制回本机，也不是重新分配一块本地缓冲区。
HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void** buffer, uint64_t* size)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(buffer);
    CHK_PTR_NULL(size);
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        auto myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        CHK_RET(myRank->ChannelGetHcclBuffer(channel, buffer, size));
        return HCCL_SUCCESS;
    }());
#endif
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CollComm* collComm = hcclComm->GetCollComm();
    hccl::MyRank* myRank = nullptr;
    if (collComm != nullptr) {
        myRank = collComm->GetMyRank();
    }
    if (collComm != nullptr && hcclComm->GetConnectMode() != 0 && myRank != nullptr) {
        CHK_RET(myRank->ChannelGetHcclBuffer(channel, buffer, size));
        return HCCL_SUCCESS;
    }

    CommBuffer commBuffer;
    auto& channelMgr = hcclComm->GetIndependentOp().GetChannelManager();
    HcclResult ret = channelMgr.ChannelCommGetHcclBuffer(channel, &commBuffer);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to get channel hccl buffer, group[%s], channel[%llu], ret[%d]", __func__,
            hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), ret);
        return ret;
    }
    *buffer = commBuffer.addr;
    *size = commBuffer.size;

    HCCL_RUN_INFO(
        "[%s] get channel hccl buffer success, group[%s], channel[%llu], "
        "buffer[type:%d, addr:%p, size:%llu], ret[%d]",
        __func__, hcclComm->GetIdentifier().c_str(), static_cast<unsigned long long>(channel), commBuffer.type,
        commBuffer.addr, static_cast<unsigned long long>(commBuffer.size), ret);
    return HCCL_SUCCESS;
}

// [中文导读] 查询建链交换得到的远端内存集合及标签；AIV等路径用它定位对端的通信信息/标记区。
// [中文导读] 本函数并不把这些内存交给调用方独立拥有，不应自行free返回的内部列表。
// [中文导读] 旧实现可用固定标签代替真实tag，读取代码时必须区分V2与兼容分支。
HcclResult
HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems, char*** memTags)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(remoteMems);
    CHK_PTR_NULL(memTags);
    CHK_PTR_NULL(memNum);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        auto myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        CHK_RET(myRank->ChannelGetRemoteMems(channel, memNum, remoteMems, memTags));
        return HCCL_SUCCESS;
    }());
#endif

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO("legacy Entry-%s:comm[%s]", __func__, commId.c_str());
    HcclMem* remoteMem = nullptr;
    HcclResult ret
        = hcclComm->GetIndependentOp().GetChannelManager().ChannelCommGetRemoteMem(channel, &remoteMem, memNum);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclChannelGetRemoteMems]legacy failed. channel[%llu], ret[%d]", channel, ret), ret);
    *remoteMems = reinterpret_cast<CommMem*>(remoteMem);
    if (*memNum > 0) {
        // A2/A3 tag为非真实tag，无法获取到，统一使用固定字符串。
        static const char* HCCL_BUFFER_TAG = "HcclBuffer";
        static thread_local std::array<char*, MAX_REMOTE_MEM_NUM> tagPtrs;
        CHK_PRT_RET(
            *memNum > MAX_REMOTE_MEM_NUM,
            HCCL_ERROR("[HcclChannelGetRemoteMems] memNum[%u] exceeds max[%u]", *memNum, MAX_REMOTE_MEM_NUM),
            HCCL_E_PARA);
        for (uint32_t i = 0; i < *memNum; ++i) {
            tagPtrs[i] = const_cast<char*>(HCCL_BUFFER_TAG);
        }
        *memTags = tagPtrs.data();
    } else {
        *memTags = nullptr;
    }
    HCCL_INFO("[HcclChannelGetRemoteMems]legacy success: memNum[%u]", *memNum);
    return HCCL_SUCCESS;
}

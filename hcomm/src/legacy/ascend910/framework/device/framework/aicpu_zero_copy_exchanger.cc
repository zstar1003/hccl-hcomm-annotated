/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_zero_copy_exchanger.h"
#include "ascend_hal.h"
#include "sal_pub.h"

namespace hccl {
ZeroCopyAddressMgr AicpuZeroCopyExchanger::globalAddrMgr_;

AicpuZeroCopyExchanger::AicpuZeroCopyExchanger(
    u32 rank, u32 rankSize, const HcclOpResParam* resParam, std::function<bool()> needStop, u32 timeoutSec,
    u32 deviceNumPerAggregation, u32 taskMonitorInterval)
    : rankId_(rank),
      rankSize_(rankSize),
      resParam_(resParam),
      needStop_(needStop),
      timeoutSec_(timeoutSec),
      deviceNumPerAggregation_(deviceNumPerAggregation),
      taskMonitorInterval_(taskMonitorInterval)
{
    HCCL_INFO("Construct AicpuZeroCopyExchanger complete.");
}

AicpuZeroCopyExchanger::~AicpuZeroCopyExchanger() {}

HcclResult AicpuZeroCopyExchanger::ExchangeAddress(
    const std::string& tag, void* localInput, void* localOutput, AlgResourceResponse* algResResponse)
{
    if (localInput == nullptr || localOutput == nullptr || algResResponse == nullptr) {
        HCCL_ERROR("[AicpuZeroCopyExchanger][ExchangeAddress] Invalid input params, maybe nullptr");
        return HCCL_E_PARA;
    }

    CHK_PRT_RET(
        needStop_ == nullptr, HCCL_ERROR("[AicpuZeroCopyExchanger][ExchangeAddress] needStop function is nullptr"),
        HCCL_E_PARA);
    HcclUs startut = TIME_NOW();
    HCCL_INFO(
        "[AicpuZeroCopyExchanger][ExchangeAddress] rank[%u] input[%p] output[%p]", rankId_, localInput, localOutput);
    CHK_RET(PrepareTagRes(tag, algResResponse->opTransportResponse));
    CHK_PTR_NULL(current_);

    if (!IsAllIpcAddressValid()) {
        HCCL_ERROR("[AicpuZeroCopyExchanger][ExchangeAddress] may some ipc address invalid");
        return HCCL_E_PARA;
    }

    CHK_RET(BatchSetLocalAddrToRemote(localInput, localOutput));

    HcclResult ret = GetRemoteAddr();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuZeroCopyExchanger][ExchangeAddress] tag[%s], rank[%u]", tag.c_str(), rankId_);
        return ret;
    }

    CHK_RET(UpdateTransportAddress());
    HcclUs endut = TIME_NOW();
    auto timeVal = DURATION_US(endut - startut).count();
    constexpr u64 MS_TO_US = 1000;
    if (taskMonitorInterval_ != 0 && static_cast<u64>(timeVal) >= taskMonitorInterval_ * MS_TO_US) {
        std::string endInfo;
        const int kLogMessageBufferSize = 100;
        endInfo.reserve(kLogMessageBufferSize);
        endInfo = "task time: " + std::to_string(timeVal) + " us," + "taskMonitor"
                  + std::to_string(taskMonitorInterval_ * MS_TO_US) + " us";
        HCCL_RUN_INFO("[ExchangeAddress] %s, %s", tag.c_str(), endInfo.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuZeroCopyExchanger::PrepareRemoteUserMemRanges(
    const uint64_t inputSize, const uint64_t outputSize, std::vector<OpUnfoldMemRange>& userInputMemRanges,
    std::vector<OpUnfoldMemRange>& userOutputMemRanges) const
{
    // 注意: 不能直接使用inAddrs_和outAddrs_, 保存的是remote ranks' user input/output memory在远端的virtual addr
    // 需要使用current_->links中的input/output memory, 才是remote ranks' user input/output memory在本端的virtual addr

    HCCL_INFO("[AicpuZeroCopyExchanger][PrepareRemoteUserMemRanges] prepare remote input/output memory ranges");

    const uint32_t rankSize = userInputMemRanges.size(); // 获取通信域内的rank数量
    const std::vector<LINK>& links = current_->links;
    for (size_t linkIdx = 0; linkIdx < links.size(); ++linkIdx) {
        const LINK& curLink = links[linkIdx];

        // 对端在通信域内的rank id
        const uint32_t remoteRank = curLink->GetRemoteRank();
        CHK_PRT_RET(
            remoteRank >= rankSize,
            HCCL_ERROR(
                "[AicpuZeroCopyExchanger][PrepareRemoteUserMemRanges] remoteRank %u >= rankSize %u", remoteRank,
                rankSize),
            HCCL_E_INTERNAL);

        HCCL_INFO(
            "[AicpuZeroCopyExchanger][PrepareRemoteUserMemRanges] prepare memory range of remote rank %u", remoteRank);

        // 获取remote user input memory addr
        void* remoteUserInputBaseAddr = nullptr;
        CHK_RET(curLink->GetRemoteMem(UserMemType::INPUT_MEM, &remoteUserInputBaseAddr));
        CHK_PTR_NULL(remoteUserInputBaseAddr);

        // 更新remote user input memory range
        OpUnfoldMemRange& remoteInputMemRange = userInputMemRanges[remoteRank];
        remoteInputMemRange.isValid = true;
        remoteInputMemRange.baseAddr = reinterpret_cast<uint64_t>(remoteUserInputBaseAddr);
        remoteInputMemRange.memSize = inputSize;

        // 获取remote user output memory addr
        void* remoteUserOutputBaseAddr = nullptr;
        CHK_RET(curLink->GetRemoteMem(UserMemType::OUTPUT_MEM, &remoteUserOutputBaseAddr));
        CHK_PTR_NULL(remoteUserOutputBaseAddr);

        // 更新remote user output memory range
        OpUnfoldMemRange& remoteOutputMemRange = userOutputMemRanges[remoteRank];
        remoteOutputMemRange.isValid = true;
        remoteOutputMemRange.baseAddr = reinterpret_cast<uint64_t>(remoteUserOutputBaseAddr);
        remoteOutputMemRange.memSize = outputSize;
    }

    return HCCL_SUCCESS;
}

bool AicpuZeroCopyExchanger::IsAllIpcAddressValid()
{
    // 目前只判断所有的共享内存是否Ok，映射部分校验放到后面check
    if (resParam_->zeroCopyIpcPtrs[rankId_ % deviceNumPerAggregation_] == 0) {
        HCCL_ERROR("[AicpuZeroCopyExchanger][IsAllIpcAddressValid] self rank %u ipc addrs is nullptr", rankId_);
        return false;
    }

    for (auto rank : current_->remoteRanks) {
        CHK_PRT_RET(
            resParam_->zeroCopyIpcPtrs[rank % deviceNumPerAggregation_] == 0,
            HCCL_ERROR("[AicpuZeroCopyExchanger][IsAllIpcAddressValid] rank %u ipc addrs is nullptr", rank), false);
    }

    return true;
}

bool AicpuZeroCopyExchanger::IsSupportZeroCopyLinkType(LinkType linkType)
{
    return linkType == LinkType::LINK_HCCS || linkType == LinkType::LINK_SIO || linkType == LinkType::LINK_HCCS_SW;
}

HcclResult AicpuZeroCopyExchanger::TryToRead(FlagData& data, u64& in, u64& out)
{
    u64 flag = data.flag;
    CHK_PRT_RET(
        flag != INVALID_DATA && flag != VALID_DATA,
        HCCL_ERROR("[AicpuZeroCopyExchanger][TryToRead] flag is [%lu] corruption", flag), HCCL_E_INTERNAL);

    // 必须是有效的才能读
    if (data.flag != VALID_DATA) {
        return HCCL_E_AGAIN;
    }

    // 先读取数据，再修改flag
    in = data.inAddr;
    out = data.outAddr;
    if (in == 0 || out == 0) {
        return HCCL_E_AGAIN;
    }

    MemFence();

    data.flag = INVALID_DATA;
    data.inAddr = 0;
    data.outAddr = 0;

    MemFence();

    return HCCL_SUCCESS;
}

HcclResult AicpuZeroCopyExchanger::GetRemoteRanks(TagRes& tagRes, OpCommTransport& opTransportResponse)
{
    CHK_PRT_RET(
        opTransportResponse.size() == 0,
        HCCL_ERROR("[AicpuZeroCopyExchanger][GetRemoteRanks] opTransportResponse size is 0"), HCCL_E_PARA);
    // 先清空已有的数据
    tagRes.remoteRanks.clear();
    tagRes.links.clear();

    for (auto& singleSubCommTransport : opTransportResponse[COMM_LEVEL0]) {
        for (u64 i = 0; i < singleSubCommTransport.links.size(); ++i) {
            LINK link = singleSubCommTransport.links[i];
            if (link == nullptr || !singleSubCommTransport.transportRequests[i].isValid
                || !IsSupportZeroCopyLinkType(link->GetLinkType())) {
                // 无效或者不支持的链路
                continue;
            }
            tagRes.remoteRanks.insert(link->GetRemoteRank());
            tagRes.links.emplace_back(link);
        }
    }

    // 校验交换地址的buffer长度是足够，目前是固定使用16个
    u32 maxDeviceNum;
    CHK_RET(GetMaxDevNum(maxDeviceNum));
    u64 actualUseLen = maxDeviceNum * sizeof(FlagData);
    CHK_PRT_RET(
        actualUseLen > ZERO_COPY_IPC_BUFFER_LENGTH,
        HCCL_ERROR(
            "[AicpuZeroCopyExchanger][GetRemoteRanks] invalid ipc buffer length [%lu] max [%lu]", actualUseLen,
            ZERO_COPY_IPC_BUFFER_LENGTH),
        HCCL_E_PARA);

    return HCCL_SUCCESS;
}

HcclResult AicpuZeroCopyExchanger::PrepareTagRes(const std::string& tag, OpCommTransport& opTransportResponse)
{
    // 清理一下当前正在使用的tag资源
    current_ = nullptr;

    // 查找是否已经配置过
    HCCL_INFO("[%s] tag[%s]", __func__, tag.c_str());
    auto it = tagRes_.find(tag);
    if (it != tagRes_.end()) {
        current_ = &it->second;
        return HCCL_SUCCESS;
    }

    TagRes tagRes;
    CHK_RET(GetRemoteRanks(tagRes, opTransportResponse));

    tagRes_[tag] = tagRes;
    current_ = &tagRes_[tag];

    // 初始化batchSdma的数据
    auto peerCount = current_->remoteRanks.size();
    current_->remotePtrs.resize(peerCount, nullptr);
    current_->selfPtrs.resize(peerCount, nullptr);
    current_->selfData.resize(peerCount);
    current_->sizes.resize(peerCount, sizeof(FlagData));
    current_->rankIds.resize(peerCount);

    // 准备batch sdma的输入输出地址
    int index = 0;
    for (auto remoteRank : current_->remoteRanks) {
        FlagData* datas
            = reinterpret_cast<FlagData*>(resParam_->zeroCopyIpcPtrs[remoteRank % deviceNumPerAggregation_]);
        current_->remotePtrs[index] = &datas[rankId_ % deviceNumPerAggregation_];
        current_->selfPtrs[index] = &current_->selfData[index];
        current_->rankIds[index] = remoteRank;
        ++index;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuZeroCopyExchanger::GetRemoteAddr()
{
    // 遍历所有对端，读取出自己所拥有的地址即可
    std::set<u32> doneRanks;
    HcclResult ret = HCCL_SUCCESS;

    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(timeoutSec_);
    while (doneRanks.size() < current_->remoteRanks.size()) {
        CHK_PRT_RET(
            needStop_(), HCCL_ERROR("AicpuZeroCopyExchanger][GetRemoteAddr] we need stop now"), HCCL_E_SUSPENDING);
        for (auto remoteRank : current_->remoteRanks) {
            CHK_PRT_RET(
                ((std::chrono::steady_clock::now() - startTime) > timeout && timeoutSec_ != 0),
                HCCL_ERROR(
                    "[AicpuZeroCopyExchanger][GetRemoteAddr] get remote addr timeout [%ld s], %s", timeout,
                    DumpLinkInfo(doneRanks).c_str()),
                HCCL_E_TIMEOUT);

            if (doneRanks.find(remoteRank) != doneRanks.end()) {
                continue;
            }

            FlagData* datas
                = reinterpret_cast<FlagData*>(resParam_->zeroCopyIpcPtrs[rankId_ % deviceNumPerAggregation_]);
            ret = TryToRead(
                datas[remoteRank % deviceNumPerAggregation_], inAddrs_[remoteRank % deviceNumPerAggregation_],
                outAddrs_[remoteRank % deviceNumPerAggregation_]);
            if (ret == HCCL_E_AGAIN) {
                continue;
            } else if (ret == HCCL_SUCCESS) {
                HCCL_INFO(
                    "[AicpuZeroCopyExchanger][GetRemoteAddr] success read from rank[%u], remoteInput[0x%lx] "
                    "remoteOutput[0x%lx]",
                    remoteRank, inAddrs_[remoteRank % deviceNumPerAggregation_],
                    outAddrs_[remoteRank % deviceNumPerAggregation_]);
                doneRanks.insert(remoteRank);
            } else {
                HCCL_ERROR(
                    "[AicpuZeroCopyExchanger][GetRemoteAddr] failed read from rank[%u] ipcPtr[%p] data[%p]", remoteRank,
                    datas, &datas[remoteRank % deviceNumPerAggregation_]);
                return ret;
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuZeroCopyExchanger::BatchSetLocalAddrToRemote(void* in, void* out)
{
    CHK_PTR_NULL(in);
    CHK_PTR_NULL(out);
    CHK_PTR_NULL(current_);

    size_t peerCount = current_->remoteRanks.size();
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(timeoutSec_);
    std::set<u32> doneRanks;
    while (true) {
        CHK_PRT_RET(
            needStop_(), HCCL_ERROR("AicpuZeroCopyExchanger][BatchSetLocalAddrToRemote] we need stop now"),
            HCCL_E_SUSPENDING);

        CHK_PRT_RET(
            ((std::chrono::steady_clock::now() - startTime) > timeout && timeoutSec_ != 0),
            HCCL_ERROR(
                "[AicpuZeroCopyExchanger][BatchSetLocalAddrToRemote] Set to remote addr timeout [%ld s], %s", timeout,
                DumpLinkInfo(doneRanks).c_str()),
            HCCL_E_TIMEOUT);

        DVresult ret = halSdmaBatchCopy(
            current_->selfPtrs.data(), current_->remotePtrs.data(), current_->sizes.data(), peerCount);
        CHK_PRT_RET(
            ret != 0,
            HCCL_ERROR(
                "[AicpuZeroCopyExchanger][BatchSetLocalAddrToRemote] sdma batch copy "
                "failed, ret[%u]",
                ret),
            HCCL_E_INTERNAL);

        size_t readyCount = 0;
        for (u64 i = 0; i < peerCount; ++i) {
            FlagData* data = reinterpret_cast<FlagData*>(current_->selfPtrs[i]);
            u64 flag = data->flag;
            CHK_PRT_RET(
                flag != INVALID_DATA && flag != VALID_DATA,
                HCCL_ERROR(
                    "[AicpuZeroCopyExchanger][BatchSetLocalAddrToRemote] rank[%lu]'s flag is [%lu] corruption",
                    current_->rankIds[i], flag),
                HCCL_E_INTERNAL);

            if (flag != INVALID_DATA) {
                break;
            }
            readyCount++;
            data->inAddr = reinterpret_cast<u64>(in);
            data->outAddr = reinterpret_cast<u64>(out);
            data->flag = VALID_DATA;
            doneRanks.insert(current_->rankIds[i]);
        }

        if (readyCount != peerCount) {
            continue;
        }

        ret = halSdmaBatchCopy(
            current_->remotePtrs.data(), current_->selfPtrs.data(), current_->sizes.data(), peerCount);
        CHK_PRT_RET(
            ret != 0,
            HCCL_ERROR(
                "[AicpuZeroCopyExchanger][BatchSetLocalAddrToRemote] sdma batch copy "
                "failed, ret[%u]",
                ret),
            HCCL_E_INTERNAL);
        break;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuZeroCopyExchanger::UpdateTransportAddress()
{
    u32* head = reinterpret_cast<u32*>(resParam_->zeroCopyHeadPtr);
    u32* tail = reinterpret_cast<u32*>(resParam_->zeroCopyTailPtr);
    ZeroCopyRingBufferItem* ringBuffer = reinterpret_cast<ZeroCopyRingBufferItem*>(resParam_->zeroCopyRingBuffer);

    CHK_PRT_RET(
        head == nullptr || tail == nullptr || ringBuffer == nullptr,
        HCCL_ERROR("[AicpuZeroCopyExchanger][UpdateTransportAddress] ring buffer ptr is nullptr"), HCCL_E_INTERNAL);

    // RingBuffer中有东西，所以先去处理一下，更新一下mgr的值
    if (*head != *tail) {
        CHK_RET(globalAddrMgr_.ProcessRingBuffer(ringBuffer, head, tail));
    }

    u64 remoteIns[MAX_MODULE_DEVICE_NUM]{};
    u64 remoteOuts[MAX_MODULE_DEVICE_NUM]{};
    for (auto remoteRank : current_->remoteRanks) {
        u32 devicePhyId = resParam_->zeroCopyDevicePhyId[remoteRank % deviceNumPerAggregation_];

        // remote in addr
        LocalIpc2RemoteAddr inMapAddr;
        CHK_RET(globalAddrMgr_.GetLocalIpc2RemoteAddr(
            devicePhyId, reinterpret_cast<void*>(inAddrs_[remoteRank % deviceNumPerAggregation_]), inMapAddr));
        remoteIns[remoteRank % deviceNumPerAggregation_]
            = inMapAddr.localIpcAddr + (inAddrs_[remoteRank % deviceNumPerAggregation_] - inMapAddr.remoteAddr);
        CHK_PRT_RET(
            !globalAddrMgr_.IsActivateCommMemoryAddr(
                reinterpret_cast<void*>(remoteIns[remoteRank % deviceNumPerAggregation_]), 1),
            HCCL_ERROR(
                "[AicpuZeroCopyExchanger][UpdateTransportAddress] rank[%u] ptr[0x%lx] is not activate", remoteRank,
                remoteIns[remoteRank % deviceNumPerAggregation_]),
            HCCL_E_PARA);

        // remote out addr
        LocalIpc2RemoteAddr outMapAddr;
        CHK_RET(globalAddrMgr_.GetLocalIpc2RemoteAddr(
            devicePhyId, reinterpret_cast<void*>(outAddrs_[remoteRank % deviceNumPerAggregation_]), outMapAddr));
        remoteOuts[remoteRank % deviceNumPerAggregation_]
            = outMapAddr.localIpcAddr + (outAddrs_[remoteRank % deviceNumPerAggregation_] - outMapAddr.remoteAddr);
        CHK_PRT_RET(
            !globalAddrMgr_.IsActivateCommMemoryAddr(
                reinterpret_cast<void*>(remoteOuts[remoteRank % deviceNumPerAggregation_]), 1),
            HCCL_ERROR(
                "[AicpuZeroCopyExchanger][UpdateTransportAddress] rank[%u] ptr[0x%lx] is not activate", remoteRank,
                remoteOuts[remoteRank % deviceNumPerAggregation_]),
            HCCL_E_PARA);

        HCCL_INFO(
            "[AicpuZeroCopyExchanger][UpdateTransportAddress] remoteRank[%u] localInBase[0x%lx] remoteInBase[0x%lx] "
            "remoteIn [0x%lx] localOutBase [0x%lx] remoteOutBase [0x%lx] remoteOut [0x%lx]",
            remoteRank, inMapAddr.localIpcAddr, inMapAddr.remoteAddr, remoteIns[remoteRank % deviceNumPerAggregation_],
            outMapAddr.localIpcAddr, outMapAddr.remoteAddr, remoteOuts[remoteRank % deviceNumPerAggregation_]);
    }

    // 因此同一个对端可能有多条p2p链路
    for (auto& link : current_->links) {
        u32 remoteRank = link->GetRemoteRank();
        void* remoteIn = reinterpret_cast<void*>(remoteIns[remoteRank % deviceNumPerAggregation_]);
        void* remoteOut = reinterpret_cast<void*>(remoteOuts[remoteRank % deviceNumPerAggregation_]);

        CHK_PRT_RET(
            remoteIn == nullptr || remoteOut == nullptr,
            HCCL_ERROR(
                "[AicpuZeroCopyExchanger][UpdateTransportAddress] remoteRank in[%p] out[%p] is invalid", remoteIn,
                remoteOut),
            HCCL_E_INTERNAL);
        CHK_RET(link->UpdateRemoteAddr(remoteIn, remoteOut));
    }

    return HCCL_SUCCESS;
}

std::string AicpuZeroCopyExchanger::DumpLinkInfo(std::set<u32>& doneRanks)
{
    std::string msg = "Expect:[";
    for (auto remoteRank : current_->remoteRanks) {
        msg += std::to_string(remoteRank) + " ";
    }

    msg += "] actual:[";
    for (auto remoteRank : doneRanks) {
        msg += std::to_string(remoteRank) + " ";
    }
    msg += "]";

    return msg;
}

} // namespace hccl

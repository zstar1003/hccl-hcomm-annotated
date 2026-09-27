/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <arpa/inet.h>
#include <securec.h>
#include <chrono>
#include <memory>
#include "network/hccp_common.h"
#include "device_capacity.h"
#include "dlhns_function.h"
#include "adapter_verbs.h"
#include "transport_device_ibverbs.h"
#include "new/hccl_dispatcher_ctx.h"

constexpr u32 RDMA_QP_EXPECT_STATUS_PAUSE = 5;
constexpr u32 RDMA_QP_EXPECT_STATUS_CONNECTED = 1;
constexpr s32 RDMA_QP_NO_MEM = -12;

constexpr u32 RDMA_WRITE_NOTIFY_OFFSET_MASK = 0xffffff;
constexpr u32 RDMA_WRITE_NOTIFY_VALUE_RECORD = 0x1000000;

// 内存屏障，确保wqe下到HBM里
#if defined(__x86_64__)
#define HCOMM_DSB() asm volatile("" ::: "memory")
#elif defined(__aarch64__)
#define HCOMM_DSB() asm volatile("dsb st" ::: "memory")
#else
#define HCOMM_DSB()
#endif

namespace hccl {
namespace {
    inline BufferKey<uintptr_t, u64> MakeMemLookupKey(u64 logicalStartVa, u64 size)
    {
        return BufferKey<uintptr_t, u64>(static_cast<uintptr_t>(logicalStartVa), size);
    }

    inline BufferKey<uintptr_t, u64> MakeMemLookupKey(const void* logicalPtr, u64 size)
    {
        return MakeMemLookupKey(static_cast<u64>(reinterpret_cast<uintptr_t>(logicalPtr)), size);
    }

    inline void* LogicalPtrToDevPtr(const RoceMemDetails& md, const void* logicalPtr)
    {
        const u64 logicalVa = static_cast<u64>(reinterpret_cast<uintptr_t>(logicalPtr));
        const u64 offset = logicalVa - md.addr;
        const u64 devVa = md.devAddr + offset;
        return reinterpret_cast<void*>(static_cast<uintptr_t>(devVa));
    }
} // namespace

std::atomic<u64> TransportDeviceIbverbs::wrIdOffset_ = {0};

TransportDeviceIbverbs::TransportDeviceIbverbs(
    DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
    std::chrono::milliseconds timeout, const TransportDeviceIbverbsData& transDevIbverbsData)
    : TransportIbverbs(dispatcher, notifyPool, machinePara, timeout),
      transDevIbverbsData_(transDevIbverbsData)
{}

TransportDeviceIbverbs::~TransportDeviceIbverbs()
{
    HCCL_DEBUG("~TransportDeviceIbverbs Enter!");

    (void)DeInit();

    if (machinePara_.deviceLogicId >= 0 && (static_cast<u32>(machinePara_.deviceLogicId) < MAX_MODULE_DEVICE_NUM)) {
        if (instanceRef_[machinePara_.deviceLogicId].Unref() == 0) {
            std::unique_lock<std::mutex> lock(notifyValueMutex_[machinePara_.deviceLogicId]);
            notifyValueMem_[machinePara_.deviceLogicId].free();
        }
    }
    HCCL_DEBUG("~TransportDeviceIbverbs Success!");
}

HcclResult TransportDeviceIbverbs::InitDrainNotifyInfo()
{
    HCCL_DEBUG(
        "[%s] RemoteNotifyAddr[%llu], remoteNotifyKey[%u], localDataNotifyAddr[%llu], localDataNotifyKey[%u],"
        "notifySize[%u]",
        __func__, transDevIbverbsData_.remoteNotifyValueAddr, transDevIbverbsData_.remoteNotifyValueKey,
        transDevIbverbsData_.localDataNotifyAddr, transDevIbverbsData_.localDataNotifyKey,
        transDevIbverbsData_.notifySize);
    CHK_PRT_RET(
        (transDevIbverbsData_.localDataNotifyAddr == 0 || transDevIbverbsData_.remoteNotifyValueAddr == 0),
        HCCL_ERROR(
            "[%s] Notify addr is nullptr, RemoteNotifyAddr[%llu], localDataNotifyAddr[%llu]",
            transDevIbverbsData_.remoteNotifyValueAddr, transDevIbverbsData_.localDataNotifyAddr),
        HCCL_E_PTR);
    memMsg_[MemType::DATA_NOTIFY_MEM].addr = reinterpret_cast<void*>(transDevIbverbsData_.localDataNotifyAddr);
    memMsg_[MemType::DATA_NOTIFY_MEM].lkey = transDevIbverbsData_.localDataNotifyKey;
    memMsg_[MemType::DATA_NOTIFY_MEM].len = transDevIbverbsData_.notifySize;
    remoteMemMsg_[MemType::NOTIFY_SRC_MEM].addr = reinterpret_cast<void*>(transDevIbverbsData_.remoteNotifyValueAddr);
    remoteMemMsg_[MemType::NOTIFY_SRC_MEM].lkey = transDevIbverbsData_.remoteNotifyValueKey;
    remoteMemMsg_[MemType::NOTIFY_SRC_MEM].len = transDevIbverbsData_.notifySize;
    CHK_RET(SignalInit(transDevIbverbsData_.dataNotify, dataNotify_));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::Init()
{
    HCCL_DEBUG("TransportDeviceIbverbs Init Enter! notifyNum[%u]", machinePara_.notifyNum);
    if (transDevIbverbsData_.useMemDetailsMgr) {
        return InitMemDetails();
    }
    CHK_RET(SignalInit(transDevIbverbsData_.ackNotify, ackNotify_));
    CHK_RET(SignalInit(transDevIbverbsData_.dataNotify, dataNotify_));
    CHK_RET(SignalInit(transDevIbverbsData_.dataAckNotify, dataAckNotify_));
    constexpr u32 QPINFO_SIZE_MAX = 33;
    constexpr u32 QPINFO_SIZE_MIN = 1;
    constexpr u32 QP_PERCONNECTION_MAX = 32;
    constexpr u32 QP_PERCONNECTION_MIN = 1;
    u32 qpInfoSize = transDevIbverbsData_.qpInfo.size();
    if (transDevIbverbsData_.qpsPerConnection + static_cast<u32>(qpInfoSize > 1) != qpInfoSize
        || qpInfoSize > QPINFO_SIZE_MAX || qpInfoSize < QPINFO_SIZE_MIN
        || transDevIbverbsData_.qpsPerConnection > QP_PERCONNECTION_MAX
        || transDevIbverbsData_.qpsPerConnection < QP_PERCONNECTION_MIN) {
        HCCL_ERROR(
            "[TransportDeviceIbverbs][Init]QPNum[%d] or qpInfos size[%u] is invalid",
            transDevIbverbsData_.qpsPerConnection, qpInfoSize);
        return HCCL_E_INTERNAL;
    }
    combineAiQpInfo_.aiQpInfo.aiQpAddr = transDevIbverbsData_.qpInfo[0].qpPtr;
    combineAiQpInfo_.aiQpInfo.sqIndex = transDevIbverbsData_.qpInfo[0].sqIndex;
    combineAiQpInfo_.aiQpInfo.dbIndex = transDevIbverbsData_.qpInfo[0].dbIndex;
    combineAiQpInfos_.resize(transDevIbverbsData_.qpsPerConnection);
    for (u32 i = 1, j = 0; i < qpInfoSize; i++, j++) {
        combineAiQpInfos_[j].aiQpInfo.aiQpAddr = transDevIbverbsData_.qpInfo[i].qpPtr;
        combineAiQpInfos_[j].aiQpInfo.sqIndex = transDevIbverbsData_.qpInfo[i].sqIndex;
        combineAiQpInfos_[j].aiQpInfo.dbIndex = transDevIbverbsData_.qpInfo[i].dbIndex;
        HCCL_DEBUG(
            "TransportDeviceIbverbs Init multiQp[%u], aiQpAddr[%llu] sqIndex[%u] dbIndex[%u]", j,
            transDevIbverbsData_.qpInfo[i].qpPtr, transDevIbverbsData_.qpInfo[i].sqIndex,
            transDevIbverbsData_.qpInfo[i].dbIndex);
    }
    notifySize_ = transDevIbverbsData_.notifySize;
    remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].addr = transDevIbverbsData_.inputBufferPtr;
    remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].lkey = transDevIbverbsData_.remoteInputKey;

    remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].addr = transDevIbverbsData_.outputBufferPtr;
    remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].lkey = transDevIbverbsData_.remoteOutputKey;

    u32 ackNotifyIdx = static_cast<u32>(MemType::ACK_NOTIFY_MEM);
    remoteMemMsg_[ackNotifyIdx].addr = reinterpret_cast<void*>(transDevIbverbsData_.remoteAckNotifyDetails.addr);
    remoteMemMsg_[ackNotifyIdx].notifyId = transDevIbverbsData_.remoteAckNotifyDetails.notifyId;
    remoteMemMsg_[ackNotifyIdx].lkey = transDevIbverbsData_.remoteAckNotifyDetails.key;

    u32 dataNotifyIdx = static_cast<u32>(MemType::DATA_NOTIFY_MEM);
    remoteMemMsg_[dataNotifyIdx].addr = reinterpret_cast<void*>(transDevIbverbsData_.remoteDataNotifyDetails.addr);
    remoteMemMsg_[dataNotifyIdx].notifyId = transDevIbverbsData_.remoteDataNotifyDetails.notifyId;
    remoteMemMsg_[dataNotifyIdx].lkey = transDevIbverbsData_.remoteDataNotifyDetails.key;

    u32 dataAckIdx = static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM);
    remoteMemMsg_[dataAckIdx].addr = reinterpret_cast<void*>(transDevIbverbsData_.remoteDataAckNotifyDetails.addr);
    remoteMemMsg_[dataAckIdx].notifyId = transDevIbverbsData_.remoteDataAckNotifyDetails.notifyId;
    remoteMemMsg_[dataAckIdx].lkey = transDevIbverbsData_.remoteDataAckNotifyDetails.key;

    HCCL_INFO(
        "%s ACK:addr[0x%llx] notifyId[%d] lkey[%u], DATA:addr[0x%llx] notifyId[%d] lkey[%u], "
        "DATA_ACK:addr[0x%llx] notifyId[%d] lkey[%u]",
        __func__, remoteMemMsg_[ackNotifyIdx].addr, remoteMemMsg_[ackNotifyIdx].notifyId,
        remoteMemMsg_[ackNotifyIdx].lkey, remoteMemMsg_[dataNotifyIdx].addr, remoteMemMsg_[dataNotifyIdx].notifyId,
        remoteMemMsg_[dataNotifyIdx].lkey, remoteMemMsg_[dataAckIdx].addr, remoteMemMsg_[dataAckIdx].notifyId,
        remoteMemMsg_[dataAckIdx].lkey);

    memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr
        = reinterpret_cast<void*>(transDevIbverbsData_.localNotifyValueAddr);
    memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey = transDevIbverbsData_.notifyValueKey;
    localInputMem_ = transDevIbverbsData_.localInputMem;
    memMsg_[MemType::USER_INPUT_MEM].addr = reinterpret_cast<void*>(transDevIbverbsData_.localInputMem.addr);
    memMsg_[MemType::USER_INPUT_MEM].len = transDevIbverbsData_.localInputMem.size;
    memMsg_[MemType::USER_INPUT_MEM].lkey = transDevIbverbsData_.localInputMem.key;

    localOutputMem_ = transDevIbverbsData_.localOutputMem;
    memMsg_[MemType::USER_OUTPUT_MEM].addr = reinterpret_cast<void*>(transDevIbverbsData_.localOutputMem.addr);
    memMsg_[MemType::USER_OUTPUT_MEM].len = transDevIbverbsData_.localOutputMem.size;
    memMsg_[MemType::USER_OUTPUT_MEM].lkey = transDevIbverbsData_.localOutputMem.key;

    notifyValueAddr_ = reinterpret_cast<void*>(transDevIbverbsData_.localNotifyValueAddr);
    CHK_RET(CheckDeviceId());
    CHK_RET(DlHnsFunction::GetInstance().DlHnsFunctionInit());
    transportAttr_.linkType = LinkType::LINK_ROCE;
    multiQpThreshold_ = transDevIbverbsData_.multiQpThreshold;
    qpsPerConnection_ = transDevIbverbsData_.qpsPerConnection;
    if (transDevIbverbsData_.userLocalNotify.size() != qpsPerConnection_
        || transDevIbverbsData_.userRemoteNotifyDetails.size() != qpsPerConnection_) {
        HCCL_ERROR(
            "[TransportDeviceIbverbs][Init]userLocalNotify size[%u] is not equal to qpsPerConnection[%u]",
            transDevIbverbsData_.userLocalNotify.size(), qpsPerConnection_);
        return HCCL_E_INTERNAL;
    }

    userMultiQpLocalNotify_.resize(transDevIbverbsData_.qpsPerConnection);
    u32 multiQpExtNotifyLength = transDevIbverbsData_.qpsPerConnection > 1 ? transDevIbverbsData_.qpsPerConnection : 0;
    multiQpDataNotify_.resize(multiQpExtNotifyLength);
    for (u32 i = 0; i < transDevIbverbsData_.qpsPerConnection; ++i) {
        CHK_PRT_RET(
            transDevIbverbsData_.userLocalNotify[i].empty() && transDevIbverbsData_.qpsPerConnection > 1,
            HCCL_ERROR(
                "[TransportDeviceIbverbs][Init]userLocalNotify[%u] is empty, qpsPerConnection[%u]", i,
                transDevIbverbsData_.qpsPerConnection),
            HCCL_E_INTERNAL);
        u32 singleQpNotifyNum = transDevIbverbsData_.qpsPerConnection > 1 ?
                                    transDevIbverbsData_.userLocalNotify[i].size() - 1 :
                                    transDevIbverbsData_.userLocalNotify[i].size();
        CHK_PRT_RET(
            singleQpNotifyNum != notifyNum_,
            HCCL_ERROR(
                "[TransportDeviceIbverbs][Init] qpIdx[%u] userLocalNotify notifynum[%u] is not equal to notifyNum_[%u]",
                i, singleQpNotifyNum, notifyNum_),
            HCCL_E_INTERNAL);
        userMultiQpLocalNotify_[i].resize(singleQpNotifyNum);
        for (u32 j = 0; j < singleQpNotifyNum; ++j) {
            CHK_RET(SignalInit(transDevIbverbsData_.userLocalNotify[i][j], userMultiQpLocalNotify_[i][j]));
        }
        if (transDevIbverbsData_.qpsPerConnection > 1) {
            CHK_RET(SignalInit(transDevIbverbsData_.userLocalNotify[i][singleQpNotifyNum], multiQpDataNotify_[i]));
        }
    }

    userMultiQpRemoteNotifyMsg_.resize(transDevIbverbsData_.qpsPerConnection);
    multiQpDataNotifyRemoteMemMsg_.resize(multiQpExtNotifyLength);
    for (u32 i = 0; i < transDevIbverbsData_.qpsPerConnection; ++i) {
        CHK_PRT_RET(
            transDevIbverbsData_.userRemoteNotifyDetails[i].empty() && transDevIbverbsData_.qpsPerConnection > 1,
            HCCL_ERROR(
                "[TransportDeviceIbverbs][Init]userLocalNotify[%u] is empty, qpsPerConnection[%u]", i,
                transDevIbverbsData_.qpsPerConnection),
            HCCL_E_INTERNAL);
        u32 singleQpNotifyNum = transDevIbverbsData_.qpsPerConnection > 1 ?
                                    transDevIbverbsData_.userRemoteNotifyDetails[i].size() - 1 :
                                    transDevIbverbsData_.userRemoteNotifyDetails[i].size();
        CHK_PRT_RET(
            singleQpNotifyNum != notifyNum_,
            HCCL_ERROR(
                "[TransportDeviceIbverbs][Init] qpIdx[%u] userLocalNotify notifynum[%u] is not equal to notifyNum_[%u]",
                i, singleQpNotifyNum, notifyNum_),
            HCCL_E_INTERNAL);
        userMultiQpRemoteNotifyMsg_[i].resize(singleQpNotifyNum);
        u32 j = 0;
        for (; j < singleQpNotifyNum; ++j) {
            userMultiQpRemoteNotifyMsg_[i][j].addr
                = reinterpret_cast<void*>(transDevIbverbsData_.userRemoteNotifyDetails[i][j].addr);
            userMultiQpRemoteNotifyMsg_[i][j].notifyId = transDevIbverbsData_.userRemoteNotifyDetails[i][j].notifyId;
            userMultiQpRemoteNotifyMsg_[i][j].lkey = transDevIbverbsData_.userRemoteNotifyDetails[i][j].key;
            HCCL_INFO(
                "userMultiQpRemoteNotifyMsg_[%u][%u] addr[0x%llx] notifyId[%u] lkey[%u]", i, j,
                userMultiQpRemoteNotifyMsg_[i][j].addr, userMultiQpRemoteNotifyMsg_[i][j].notifyId,
                userMultiQpRemoteNotifyMsg_[i][j].lkey);
        }
        if (transDevIbverbsData_.qpsPerConnection > 1) {
            multiQpDataNotifyRemoteMemMsg_[i].addr
                = reinterpret_cast<void*>(transDevIbverbsData_.userRemoteNotifyDetails[i][j].addr);
            multiQpDataNotifyRemoteMemMsg_[i].notifyId = transDevIbverbsData_.userRemoteNotifyDetails[i][j].notifyId;
            multiQpDataNotifyRemoteMemMsg_[i].lkey = transDevIbverbsData_.userRemoteNotifyDetails[i][j].key;
            HCCL_INFO(
                "multiQpDataNotifyRemoteMemMsg_[%u] addr[0x%llx] notifyId[%u] lkey[%u]", i,
                multiQpDataNotifyRemoteMemMsg_[i].addr, multiQpDataNotifyRemoteMemMsg_[i].notifyId,
                multiQpDataNotifyRemoteMemMsg_[i].lkey);
        }
    }
    useAtomicWrite_ = transDevIbverbsData_.useAtomicWrite;
    HCCL_USER_CRITICAL_LOG(
        "create hccl transport:communicator[%s], local rank[%u], remote rank[%u],"
        "transporttype[%s], atomicWrite[%d]",
        machinePara_.tag.c_str(), machinePara_.localUserrank, machinePara_.remoteUserrank,
        GetLinkTypeEnumStr(GetLinkType()).c_str(), useAtomicWrite_);

    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::InitMemDetails()
{
    constexpr u32 QPINFO_SIZE_MAX = 33;
    constexpr u32 QPINFO_SIZE_MIN = 1;
    constexpr u32 QP_PERCONNECTION_MAX = 32;
    constexpr u32 QP_PERCONNECTION_MIN = 1;
    u32 qpSize = transDevIbverbsData_.qpInfo.size();
    if (transDevIbverbsData_.qpsPerConnection + static_cast<u32>(qpSize > 1) != qpSize || qpSize > QPINFO_SIZE_MAX
        || qpSize < QPINFO_SIZE_MIN || transDevIbverbsData_.qpsPerConnection > QP_PERCONNECTION_MAX
        || transDevIbverbsData_.qpsPerConnection < QP_PERCONNECTION_MIN) {
        HCCL_ERROR(
            "[TransportDeviceIbverbs][InitMemDetails]QPNum[%d] or qpInfos size[%u] is invalid",
            transDevIbverbsData_.qpsPerConnection, qpSize);
        return HCCL_E_INTERNAL;
    }
    combineAiQpInfo_.aiQpInfo.aiQpAddr = transDevIbverbsData_.qpInfo[0].qpPtr;
    combineAiQpInfo_.aiQpInfo.sqIndex = transDevIbverbsData_.qpInfo[0].sqIndex;
    combineAiQpInfo_.aiQpInfo.dbIndex = transDevIbverbsData_.qpInfo[0].dbIndex;
    combineAiQpInfos_.resize(transDevIbverbsData_.qpsPerConnection);
    for (u32 i = 1, j = 0; i < qpSize; i++, j++) {
        combineAiQpInfos_[j].aiQpInfo.aiQpAddr = transDevIbverbsData_.qpInfo[i].qpPtr;
        combineAiQpInfos_[j].aiQpInfo.sqIndex = transDevIbverbsData_.qpInfo[i].sqIndex;
        combineAiQpInfos_[j].aiQpInfo.dbIndex = transDevIbverbsData_.qpInfo[i].dbIndex;
    }

    CHK_RET(CheckDeviceId());
    CHK_RET(DlHnsFunction::GetInstance().DlHnsFunctionInit());
    transportAttr_.linkType = LinkType::LINK_ROCE;
    multiQpThreshold_ = transDevIbverbsData_.multiQpThreshold;
    qpsPerConnection_ = transDevIbverbsData_.qpsPerConnection;
    useAtomicWrite_ = transDevIbverbsData_.useAtomicWrite;
    HCCL_USER_CRITICAL_LOG(
        "create hccl transport:communicator[%s], local rank[%u], remote rank[%u],"
        "transporttype[%s], atomicWrite[%d]",
        machinePara_.tag.c_str(), machinePara_.localUserrank, machinePara_.remoteUserrank,
        GetLinkTypeEnumStr(GetLinkType()).c_str(), useAtomicWrite_);
    CHK_RET(BuildMemDetailsRmaMgrs());
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::BuildMemDetailsRmaMgrs()
{
    localMemDetailsRmaMgr_.reset();
    remoteMemDetailsRmaMgr_.reset();
    useMemDetailsLookup_ = false;
    localMemDetailsRmaMgr_ = std::make_unique<DeviceMemDetailsRmaMgr>();
    remoteMemDetailsRmaMgr_ = std::make_unique<DeviceMemDetailsRmaMgr>();
    for (const auto& md : transDevIbverbsData_.localRoceMemDetailsList) {
        if (md.size == 0U) {
            continue;
        }
        auto ent = std::make_shared<RoceMemDetails>(md);
        auto pr = localMemDetailsRmaMgr_->Add(MakeMemLookupKey(md.addr, md.size), ent);
        if (pr.first == localMemDetailsRmaMgr_->End()) {
            HCCL_ERROR(
                "[TransportDeviceIbverbs][BuildMemDetailsRmaMgrs] add local mem range failed, "
                "logical[0x%llx, +%llu) devBase[0x%llx] key[%u]",
                static_cast<unsigned long long>(md.addr), static_cast<unsigned long long>(md.size),
                static_cast<unsigned long long>(md.devAddr), md.key);
            return HCCL_E_INTERNAL;
        }
        HCCL_DEBUG(
            "[TransportDeviceIbverbs][BuildMemDetailsRmaMgrs] add local MR logical[0x%llx, +%llu) "
            "devBase[0x%llx] key[%u]",
            static_cast<unsigned long long>(md.addr), static_cast<unsigned long long>(md.size),
            static_cast<unsigned long long>(md.devAddr), md.key);
    }
    for (const auto& md : transDevIbverbsData_.remoteRoceMemDetailsList) {
        if (md.size == 0U) {
            continue;
        }
        auto ent = std::make_shared<RoceMemDetails>(md);
        auto pr = remoteMemDetailsRmaMgr_->Add(MakeMemLookupKey(md.addr, md.size), ent);
        if (pr.first == remoteMemDetailsRmaMgr_->End()) {
            HCCL_ERROR(
                "[TransportDeviceIbverbs][BuildMemDetailsRmaMgrs] add remote mem range failed, "
                "logical[0x%llx, +%llu) devBase[0x%llx] key[%u]",
                static_cast<unsigned long long>(md.addr), static_cast<unsigned long long>(md.size),
                static_cast<unsigned long long>(md.devAddr), md.key);
            return HCCL_E_INTERNAL;
        }
        HCCL_DEBUG(
            "[TransportDeviceIbverbs][BuildMemDetailsRmaMgrs] add remote MR logical[0x%llx, +%llu) "
            "devBase[0x%llx] key[%u]",
            static_cast<unsigned long long>(md.addr), static_cast<unsigned long long>(md.size),
            static_cast<unsigned long long>(md.devAddr), md.key);
    }
    HCCL_INFO(
        "[TransportDeviceIbverbs][BuildMemDetailsRmaMgrs] indexed localMR[%zu] remoteMR[%zu]",
        localMemDetailsRmaMgr_->size(), remoteMemDetailsRmaMgr_->size());
    useMemDetailsLookup_ = true;
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::AddWrList(
    void* dstMemPtr, const void* srcMemPtr, u64 srcMemSize, u32 srcKey, u32 dstKey, WqeType wqeType, WrAuxInfo& aux,
    std::vector<WrInformation>& wrInfoVec, bool isLast)
{
    HCCL_DEBUG("TransportDeviceIbverbs AddWrList start");
    if (srcMemSize == 0) {
        return HCCL_SUCCESS;
    }
    WrInformation wrInfoTmp;
    wrInfoTmp.wrData.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(dstMemPtr));
    wrInfoTmp.wrData.rkey = dstKey;
    wrInfoTmp.wrData.sendFlags = fence_ ? (RA_SEND_SIGNALED | RA_SEND_FENCE) : (isLast ? RA_SEND_SIGNALED : 0);
    fence_ = false;
    wrInfoTmp.wrData.immData = 0;
    wrInfoTmp.wrData.wrId = 0;
    wrInfoTmp.wrData.memList.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(srcMemPtr));
    wrInfoTmp.wrData.memList.len = srcMemSize;
    wrInfoTmp.wrData.memList.lkey = srcKey;

    switch (wqeType) {
        case WqeType::WQE_TYPE_DATA:
        case WqeType::WQE_TYPE_DATA_NOTIFY:
        case WqeType::WQE_TYPE_ACK_NOTIFY:
        case WqeType::WQE_TYPE_DATA_ACK_NOTIFY:
        case WqeType::WQE_TYPE_DATA_WITH_NOTIFY:
            wrInfoTmp.wrData.op = RA_WR_RDMA_WRITE;
            wrInfoTmp.type = static_cast<u64>(wqeType);
            break;
        case WqeType::WQE_TYPE_DATA_WITH_REDUCE:
            wrInfoTmp.wrData.op = RA_WR_RDMA_REDUCE_WRITE;
            wrInfoTmp.wrData.aux = aux;
            // REDUCE WRITE 作为特殊的DATA
            wrInfoTmp.type = static_cast<u64>(WqeType::WQE_TYPE_DATA);
            break;
        case WqeType::WQE_TYPE_READ_DATA:
            wrInfoTmp.wrData.op = RA_WR_RDMA_READ;
            wrInfoTmp.type = static_cast<u64>(wqeType);
            break;
        default:
            HCCL_ERROR("error wqeType[%d]", wqeType);
            return HCCL_E_INTERNAL;
    }
    CHK_RET(GetWrDataAddr(dstMemPtr, wqeType, wrInfoTmp.wrDataAddr, wrInfoTmp.notifyId));
    HCCL_DEBUG(
        "wrInfoTmp dst_addr[0x%llx] memList addr[0x%llx] len[%llu] isLast[%u]", wrInfoTmp.wrData.dstAddr,
        wrInfoTmp.wrData.memList.addr, srcMemSize, isLast);
    wrInfoVec.push_back(wrInfoTmp);
    HCCL_DEBUG("TransportDeviceIbverbs AddWrList end");
    return HCCL_SUCCESS;
}

HcclResult
TransportDeviceIbverbs::GetMemInfo(UserMemType memType, void** dstMemPtr, unsigned int* dstKey, u64& dstMemSize)
{
    CHK_PTR_NULL(dstMemPtr);
    CHK_PTR_NULL(dstKey);

    switch (memType) {
        case UserMemType::INPUT_MEM: {
            *dstMemPtr = remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].addr;
            dstMemSize = remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].len;
            *dstKey = remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].lkey;
            break;
        }

        case UserMemType::OUTPUT_MEM: {
            *dstMemPtr = remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].addr;
            dstMemSize = remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].len;
            *dstKey = remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].lkey;
            break;
        }

        default: {
            HCCL_ERROR("[Get][MemInfo]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::ConstructPayLoadWqe(
    void* dstMemPtr, u32 dstKey, const void* src, u32 srcKey, u64 len, WqeType wqeType, WrAuxInfo& aux,
    std::vector<WrInformation>& wrInfoVec, u32 txSendDataTimes, bool isLast)
{
    HcclResult ret;
    // 发送数据Wqe
    for (u32 txSendDataIdx = 0; txSendDataIdx < txSendDataTimes; txSendDataIdx++) {
        u64 txSendDataOffset = txSendDataIdx * RDMA_SEND_MAX_SIZE;
        u64 txSendDataSize = (txSendDataIdx == (txSendDataTimes - 1)) ? len - txSendDataOffset : RDMA_SEND_MAX_SIZE;

        void* txdstMemPtr = reinterpret_cast<void*>(reinterpret_cast<char*>(dstMemPtr) + txSendDataOffset);

        const void* txsrcMemPtr = reinterpret_cast<const void*>(reinterpret_cast<const char*>(src) + txSendDataOffset);
        ret = AddWrList(txdstMemPtr, txsrcMemPtr, txSendDataSize, srcKey, dstKey, wqeType, aux, wrInfoVec, isLast);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TransportDeviceIbverbs][TxAsync]errNo[0x%016llx] In lbv exp, add wqe list failed."
                "srcMemSize[%llu]",
                HCCL_ERROR_CODE(ret), txSendDataSize),
            ret);
    }
    HCCL_DEBUG("TransportDeviceIbverbs TxPayLoad end");

    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::TxPayLoad(
    UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, WqeType wqeType, WrAuxInfo& aux,
    std::vector<WrInformation>& wrInfoVec)
{
    HCCL_DEBUG("TransportDeviceIbverbs TxPayLoad start");
    void* dstMemPtr = nullptr;
    unsigned int dstKey;
    unsigned int srcKey;
    u64 dstMemSize = 0;
    // 为保证单算子下不同数据量下子图的结构相同，zero byte message 时也需要下发task
    u32 txSendDataTimes = (len + RDMA_SEND_MAX_SIZE - 1) / RDMA_SEND_MAX_SIZE;

    // 当前len不可用，无法校验dstOffset > dstMemSize
    CHK_RET(GetMemInfo(dstMemType, &dstMemPtr, &dstKey, dstMemSize));

    u64 srcAddr = reinterpret_cast<u64>(src);
    if (srcAddr >= localInputMem_.addr && srcAddr < localInputMem_.addr + localInputMem_.size) {
        srcKey = localInputMem_.key;
    } else if (srcAddr >= localOutputMem_.addr && srcAddr <= localOutputMem_.addr + localOutputMem_.size) {
        srcKey = localOutputMem_.key;
    } else {
        HCCL_ERROR(
            "[TransportDeviceIbverbs][TxAsync]src_ptr=%p is out of range, inputmem src[%p], size[%llu];"
            " outputmem src[%p] size[%llu]",
            src, localInputMem_.addr, localInputMem_.size, localOutputMem_.addr, localOutputMem_.size);
        return HCCL_E_INTERNAL;
    }

    dstMemPtr = reinterpret_cast<void*>(reinterpret_cast<char*>(dstMemPtr) + dstOffset);
    CHK_RET(ConstructPayLoadWqe(dstMemPtr, dstKey, src, srcKey, len, wqeType, aux, wrInfoVec, txSendDataTimes));

    return HCCL_SUCCESS;
}

HcclResult
TransportDeviceIbverbs::TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream)
{
    CHK_SMART_PTR_NULL(stream);
    std::vector<WrInformation> wrInfoVec;
    struct WrAuxInfo aux = {};
    HCCL_DEBUG("TX src[%p] len[%llu] dstOffset[%llu]", src, len, dstOffset);

    if (len > 0) {
        CHK_PTR_NULL(src);
        CHK_RET(TxPayLoad(dstMemType, dstOffset, src, len, WqeType::WQE_TYPE_DATA, aux, wrInfoVec));
    }

    CHK_RET(TxSendDataAndNotify(wrInfoVec, stream, GetUseOneDoorbellValue()));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::TxWithReduce(
    UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, const HcclDataType datatype, HcclReduceOp redOp,
    Stream& stream)
{
    CHK_SMART_PTR_NULL(stream);
    std::vector<WrInformation> wrInfoVec;
    struct WrAuxInfo aux = {};
    aux.dataType = RDMA_REDUCE_DATA_TYPE_TABLE[datatype];
    aux.reduceType = RDMA_REDUCE_OP_TYPE_TABLE[redOp];
    if (aux.dataType == static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID)
        || aux.reduceType == static_cast<uint8_t>(RdmaReduceOpType::RDMA_REDUCE_OP_INVALID)) {
        HCCL_ERROR(
            "unsupported data type [%s] or Reduce type [%s]", GetDataTypeEnumStr(datatype).c_str(),
            GetReduceOpEnumStr(redOp).c_str());
        return HCCL_E_INTERNAL;
    }
    if (len > 0) {
        CHK_PTR_NULL(src);
        CHK_RET(TxPayLoad(dstMemType, dstOffset, src, len, WqeType::WQE_TYPE_DATA_WITH_REDUCE, aux, wrInfoVec));
    }

    CHK_RET(TxSendDataAndNotify(wrInfoVec, stream, GetUseOneDoorbellValue()));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::TxWithReduce(
    const std::vector<TxMemoryInfo>& txWithReduceMems, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream)
{
    CHK_SMART_PTR_NULL(stream);
    std::vector<WrInformation> wrInfoVec;
    struct WrAuxInfo aux = {};
    aux.dataType = RDMA_REDUCE_DATA_TYPE_TABLE[datatype];
    aux.reduceType = RDMA_REDUCE_OP_TYPE_TABLE[redOp];
    if (aux.dataType == static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID)
        || aux.reduceType == static_cast<uint8_t>(RdmaReduceOpType::RDMA_REDUCE_OP_INVALID)) {
        HCCL_ERROR(
            "unsupported data type [%s] or Reduce type [%s]", GetDataTypeEnumStr(datatype).c_str(),
            GetReduceOpEnumStr(redOp).c_str());
        return HCCL_E_INTERNAL;
    }

    for (const TxMemoryInfo& txWithReduceMem : txWithReduceMems) {
        if (txWithReduceMem.len == 0) {
            continue;
        }
        CHK_PTR_NULL(txWithReduceMem.src);
        CHK_RET(TxPayLoad(
            txWithReduceMem.dstMemType, txWithReduceMem.dstOffset, txWithReduceMem.src, txWithReduceMem.len,
            WqeType::WQE_TYPE_DATA_WITH_REDUCE, aux, wrInfoVec));
    }

    CHK_RET(TxSendDataAndNotify(wrInfoVec, stream, GetUseOneDoorbellValue()));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::TxSendDataAndNotifyWithSingleQP(
    std::vector<WrInformation>& wrInfoVec, Stream& stream, bool useOneDoorbell)
{
    // 发送data notify同步信息
    struct WrAuxInfo aux = {};
    void* remoteNotifyaddr = remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].addr;
    ;
    CHK_RET(AddWrList(
        remoteNotifyaddr, notifyValueAddr_, notifySize_, memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey,
        remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].lkey, WqeType::WQE_TYPE_DATA_NOTIFY, aux, wrInfoVec));

    CHK_RET(RdmaSendAsync(wrInfoVec, stream, useOneDoorbell));
    return HCCL_SUCCESS;
}

HcclResult
TransportDeviceIbverbs::TxSendDataAndNotify(std::vector<WrInformation>& wrInfoVec, Stream& stream, bool useOneDoorbell)
{
    u32 maxLength = 0;
    for (u32 i = 0; i < wrInfoVec.size(); i++) {
        if (wrInfoVec[i].wrData.memList.len > maxLength) {
            maxLength = wrInfoVec[i].wrData.memList.len;
        }
    }
    u32 actualMultiQpNum = GetActualQpNum(maxLength);
    HCCL_DEBUG(
        "[TransportDeviceIbverbs][TxSendDataAndNotify] UseMultiQp[%d] MultiQpNum[%u] actualMultiQpNum[%u] "
        "maxLength[%u]",
        UseMultiQp(), qpsPerConnection_, actualMultiQpNum, maxLength);
    if (UseMultiQp() && actualMultiQpNum != 1 && actualMultiQpNum <= qpsPerConnection_ && maxLength != 0) {
        CHK_RET(TxSendDataAndNotifyWithMultiQP(wrInfoVec, actualMultiQpNum, stream, useOneDoorbell));
    } else {
        CHK_RET(TxSendDataAndNotifyWithSingleQP(wrInfoVec, stream, useOneDoorbell));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream)
{
    CHK_SMART_PTR_NULL(stream);

    std::vector<WrInformation> wrInfoVec;
    struct WrAuxInfo aux = {};

    for (auto& mem : txMems) {
        HCCL_DEBUG("TX src[%p] len[%llu] dstOffset[%llu]", mem.src, mem.len, mem.dstOffset);
        if (mem.len == 0) {
            continue;
        }
        CHK_PTR_NULL(mem.src);
        CHK_RET(TxPayLoad(mem.dstMemType, mem.dstOffset, mem.src, mem.len, WqeType::WQE_TYPE_DATA, aux, wrInfoVec));
    }

    CHK_RET(TxSendDataAndNotify(wrInfoVec, stream, GetUseOneDoorbellValue()));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::TxWrList(
    std::vector<WrInformation>& wrInfoVec, Stream& stream, std::vector<struct SendWrRsp>& opRspVec, u32 multiQpIndex)
{
    (void)stream;

    u32 totalWqeCount = wrInfoVec.size();
    WrInformation* wrlist = wrInfoVec.data();
    struct SendWrRsp* opRsp = opRspVec.data();

    // HCCP会校验 zero byte messages 的内存地址是否已注册MR。对于 zero byte messages 不下发WR，将opRsp设置为特殊值。
    // 下发rdmasend task时检查该特殊值，如果zero byte message则不下发rdmasend task。
    bool batchSendWr = true;
    for (u32 i = 0; i < totalWqeCount; i++) {
        if (wrInfoVec[i].wrData.memList.len == 0) {
            batchSendWr = false;
            break;
        }
    }

    if (batchSendWr) {
        CHK_RET(SendWrList(totalWqeCount, wrlist, opRsp, multiQpIndex));
    } else {
        for (u32 i = 0; i < totalWqeCount; i++) {
            if (wrInfoVec[i].wrData.memList.len > 0) {
                CHK_RET(SendWrList(1U, &wrlist[i], &opRsp[i], multiQpIndex));
            } else {
                opRsp[i].wqeTmp.sqIndex = INVALID_UINT;
                opRsp[i].wqeTmp.wqeIndex = INVALID_UINT;
                opRsp[i].db.dbIndex = INVALID_UINT;
                opRsp[i].db.dbInfo = INVALID_U64;
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult
TransportDeviceIbverbs::SendWrList(u32 wrNum, WrInformation* wrlist, struct SendWrRsp* opRsp, u32 multiQpIndex)
{
    unsigned int completeNum = 0;
    HcclResult ret = SendWrlistExt(wrlist, opRsp, wrNum, &completeNum, multiQpIndex);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[TransportDeviceIbverbs][SendWrList]In ibv send wq list, SendWrlistExt failed.ret[%d]", ret),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::SendWrlistExt(
    WrInformation wr[], struct SendWrRsp opRsp[], unsigned int sendNum, unsigned int* completeNum, u32 multiQpIndex)
{
    HcclResult ret = HCCL_SUCCESS;
    auto startTime = std::chrono::steady_clock::now();
    u32 remainNum = sendNum;
    unsigned int completeNumLocal = 0;
    *completeNum = 0;
    while (true) {
        if (remainNum > sendNum) {
            HCCL_ERROR(
                "[Aicpu][Send][Wr]wr list send async fail. return[%d], remainNum[%u], "
                "sendNum[%u].",
                HCCL_E_ROCE_TRANSFER, remainNum, sendNum);
            return HCCL_E_ROCE_TRANSFER;
        }
        if (remainNum == 0) {
            break;
        }
        ret = TxSendWrlistExt(
            wr + (sendNum - remainNum), remainNum, opRsp + (sendNum - remainNum), &completeNumLocal, multiQpIndex);
        *completeNum += completeNumLocal;
        if (ret == HCCL_SUCCESS && *completeNum == sendNum) {
            break; // 成功跳出
        }

        if (ret == HCCL_E_AGAIN || *completeNum < sendNum) {
            remainNum -= completeNumLocal;
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout_);
            CHK_PRT_RET(
                bTimeout,
                HCCL_ERROR(
                    "[Aicpu][Send][Wr]errNo[0x%016llx] wrlist send async timeout[%d]ms. "
                    "return[%d], params: send_wrAddr[%p], opRspAddr[%p]",
                    HCCL_ERROR_CODE(HCCL_E_ROCE_TRANSFER), timeout_, ret, wr, opRsp),
                HCCL_E_ROCE_TRANSFER);
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            HCCL_ERROR(
                "[Aicpu][Send][Wr]wrlist send async fail. return[%d], para: send_wrAddr[%p], "
                "opRspAddr[%p].",
                ret, wr, opRsp);
            return HCCL_E_ROCE_TRANSFER; // 非-2/-11场景错误，不轮询，直接退出
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::TxSendWrlistExt(
    WrInformation wrList[], u32 sendNum, struct SendWrRsp opRsp[], unsigned int* completeNum, u32 multiQpIndex)
{
    u32 i = 0;
    s32 ret = 0;
    struct ibv_send_wr ib_wr = {};
    struct ibv_sge list = {};
    struct ibv_send_wr* bad_wr = nullptr;
    struct WrExpRsp exp_rsp = {};
    struct IbvPostSendExtResp ext_rsp = {};
    struct IbvPostSendExtAddt ext_attr = {};
    for (; i < sendNum; i++) {
        if (wrList[i].wrData.memList.len > IBV_SGLIST_LEN_MAX) {
            HCCL_ERROR("sg list len is more than 2G, len[%u]", wrList[i].wrData.memList.len);
            return HCCL_E_PARA;
        }
        u64 wrIdoffset = wrIdOffset_++;

        // 910B和910_93，reduce的下一个notify要设置为atomic write
        u32& preWrOpcode = multiQpIndex == RDMA_INVALID_QP_INDEX ? combineAiQpInfo_.preWrOpcode :
                                                                   combineAiQpInfos_[multiQpIndex].preWrOpcode;
        ModifyAtomicWriteAfterReduce(preWrOpcode, wrList[i].type, wrList[i].wrData.op, wrList[i].wrData.immData);

        if (wrList[i].wrData.op != RA_WR_SEND && wrList[i].wrData.op != RA_WR_SEND_WITH_IMM) {
            HCCL_DEBUG("remote wr dst addr is 0x%llx", wrList[i].wrData.dstAddr);
            list.addr = wrList[i].wrData.memList.addr;
            list.length = wrList[i].wrData.memList.len;
            list.lkey = wrList[i].wrData.memList.lkey;

            ib_wr.sg_list = &list;
            ib_wr.opcode = static_cast<enum ibv_wr_opcode>(wrList[i].wrData.op);
            ib_wr.send_flags = static_cast<unsigned int>(wrList[i].wrData.sendFlags);
            ib_wr.imm_data = wrList[i].wrData.immData;

            ib_wr.num_sge = 1; /* only support one sge */
            ib_wr.wr_id = wrList[i].wrData.wrId += wrIdoffset;
            ib_wr.wr.rdma.rkey = wrList[i].wrData.rkey;
            ib_wr.wr.rdma.remote_addr = wrList[i].wrData.dstAddr;
        } else {
            list.addr = wrList[i].wrData.memList.addr;
            list.length = wrList[i].wrData.memList.len;
            list.lkey = wrList[i].wrData.memList.lkey;

            ib_wr.sg_list = &list;
            ib_wr.opcode = static_cast<enum ibv_wr_opcode>(wrList[i].wrData.op);
            ib_wr.send_flags = static_cast<unsigned int>(wrList[i].wrData.sendFlags);
            ib_wr.imm_data = wrList[i].wrData.immData;

            ib_wr.num_sge = 1; /* only support one sge */
            ib_wr.wr_id = wrList[i].wrData.wrId += wrIdoffset;
        }
        unsigned long long aiQpAddr = multiQpIndex == RDMA_INVALID_QP_INDEX ?
                                          combineAiQpInfo_.aiQpInfo.aiQpAddr :
                                          combineAiQpInfos_[multiQpIndex].aiQpInfo.aiQpAddr;
        struct ibv_qp* qp = reinterpret_cast<struct ibv_qp*>(aiQpAddr);
        HCCL_DEBUG(
            "ib_wr.sglist[%u].addr[%p], ib_wr.sglist[%u].length[%u], ib_wr.sglist[%u], "
            "ib_wr.wr_id[%llu], raddr[%p], opcode[%d], imm_data[0x%llx]",
            i, list.addr, i, list.length, i, ib_wr.wr_id, ib_wr.wr.rdma.remote_addr, wrList[i].wrData.op,
            ib_wr.imm_data);
        if (wrList[i].wrData.op == RA_WR_RDMA_ATOMIC_WRITE) {
            ext_attr.reduce_op = wrList[i].wrData.aux.reduceType;
            ext_attr.reduce_type = wrList[i].wrData.aux.dataType;
            ret = DlHnsFunction::GetInstance().dlHnsIbvExtPostSend(qp, &ib_wr, &bad_wr, &ext_attr, &ext_rsp);
            HCOMM_DSB();
            exp_rsp.wqe_index = ext_rsp.wqe_index;
            exp_rsp.db_info = ext_rsp.db_info;
            HCCL_DEBUG(
                "ibv_ext_post_send, op = [0x%x], imm_data = [0x%lx], reduce_op = [%d], reduceType = [%d]",
                wrList[i].wrData.op, ib_wr.imm_data, ext_attr.reduce_op, ext_attr.reduce_type);
        } else if (
            wrList[i].wrData.op == RA_WR_RDMA_WRITE_WITH_NOTIFY || wrList[i].wrData.op == RA_WR_RDMA_REDUCE_WRITE
            || wrList[i].wrData.op == RA_WR_RDMA_REDUCE_WRITE_WITH_NOTIFY) {
            ib_wr.imm_data = htobe32(
                (wrList[i].wrData.aux.notifyOffset & RDMA_WRITE_NOTIFY_OFFSET_MASK) | RDMA_WRITE_NOTIFY_VALUE_RECORD);
            ext_attr.reduce_op = wrList[i].wrData.aux.reduceType;
            ext_attr.reduce_type = wrList[i].wrData.aux.dataType;
            ret = DlHnsFunction::GetInstance().dlHnsIbvExtPostSend(qp, &ib_wr, &bad_wr, &ext_attr, &ext_rsp);
            HCOMM_DSB();
            exp_rsp.wqe_index = ext_rsp.wqe_index;
            exp_rsp.db_info = ext_rsp.db_info;
            HCCL_DEBUG(
                "ibv_ext_post_send, op = [0x%x], imm_data = [0x%lx], reduce_op = [%d],reduceType = [%d]",
                wrList[i].wrData.op, ib_wr.imm_data, ext_attr.reduce_op, ext_attr.reduce_type);
        } else {
            ret = DlHnsFunction::GetInstance().dlHnsIbvExpPostSend(qp, &ib_wr, &bad_wr, &exp_rsp);
            HCOMM_DSB();
            HCCL_DEBUG(
                "ibv_exp_post_send, op = [0x%x], remote_addr = [0x%llx], size = [%d]", wrList[i].wrData.op,
                ib_wr.wr.rdma.remote_addr, ib_wr.sg_list->length);
        }
        if (ret != 0) {
            HCCL_WARNING("[TxSendWrlistExt]ibv_post_send failed ret %d, i[%u]", ret, i);
            break;
        }
        unsigned long long dbIndex = multiQpIndex == RDMA_INVALID_QP_INDEX ?
                                         combineAiQpInfo_.aiQpInfo.dbIndex :
                                         combineAiQpInfos_[multiQpIndex].aiQpInfo.dbIndex;
        opRsp[i].db.dbIndex = (unsigned int)dbIndex;
        HCCL_DEBUG("opRsp.db.dbIndex = [%d]", opRsp[i].db.dbIndex);
        opRsp[i].db.dbInfo = exp_rsp.db_info;
    }

    HCCL_DEBUG("completeNum[%d], ret[%d]", i, ret);
    *completeNum = i;
    if ((ret == SOCK_ENOENT) || (ret == ROCE_EAGAIN)
        || (workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && ret == ROCE_ENOMEM)) {
        return HCCL_E_AGAIN;
    } else if (ret == 0) {
        return HCCL_SUCCESS;
    } else if (ret == RDMA_QP_NO_MEM) { // 表示qp已满，内存不足，需要重发
        ib_wr.wr_id = wrList[i].wrData.wrId -= wrIdOffset_;
        // 可能出现主流没有launch，但从流一直下发导致卡死超时的问题，所以这里将所有流都下发
        CHK_RET(dispatcher_->LaunchAllTasks());
        return HCCL_E_AGAIN;
    } else {
        return HCCL_E_ROCE_TRANSFER;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::RdmaSendAsync(
    std::vector<WrInformation>& wrInfoVec, Stream& stream, bool useOneDoorbell, u32 multiQpIndex)
{
    HcclResult ret;

    std::vector<struct SendWrRsp> opRspVec(wrInfoVec.size());
    CHK_RET(TxWrList(wrInfoVec, stream, opRspVec, multiQpIndex));

    for (u32 i = 0; i < wrInfoVec.size(); i++) {
        if (useOneDoorbell && i != wrInfoVec.size() - 1) {
            // 如果useOneDoorbell为true，只敲最后一次doorbell
            continue;
        }

        RdmaTaskInfo taskInfo = {};
        taskInfo.remoteRank = machinePara_.remoteWorldRank;
        taskInfo.rdmaType = (wrInfoVec[i].type == static_cast<u64>(WqeType::WQE_TYPE_DATA)) ?
                                RdmaType::RDMA_SEND_PAYLOAD :
                                RdmaType::RDMA_SEND_NOTIFY;

        if (useOneDoorbell) {
            // 如果useOneDoorbell为true，一次性传入所有wr
            taskInfo.wrInfos = wrInfoVec;
        } else {
            taskInfo.wrInfos.push_back(wrInfoVec[i]);
        }

        u32 dbIndex = static_cast<u32>(opRspVec[i].db.dbIndex);
        HCCL_DEBUG("dbIndex = [%d]", dbIndex);
        u64 dbInfo = static_cast<u64>(opRspVec[i].db.dbInfo);

        ret = dispatcher_->RdmaSend(dbIndex, dbInfo, stream, taskInfo);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TransportDeviceIbverbs][RdmaSendAsync]errNo[0x%016llx] In lbv exp op base mode, "
                "rdma send failed. dbIndex[%u] dbInfo[%llu] wqe type[%llu] addr[%llu]",
                HCCL_ERROR_CODE(ret), dbIndex, dbInfo, wrInfoVec[i].type, wrInfoVec[i].wrDataAddr),
            ret);
    }
    return HCCL_SUCCESS;
}

HcclResult
TransportDeviceIbverbs::RdmaSendAsync(struct SendWr& wr, Stream& stream, WqeType wqeType, u64 notifyAddr, u32 notifyId)
{
    HcclResult ret;
    WrInformation wrInfoTmp;
    struct SendWrRsp opRsp = {};
    struct WrAuxInfo aux = {};
    wrInfoTmp.wrData.memList = wr.bufList[0];
    wrInfoTmp.wrData.dstAddr = wr.dstAddr;
    wrInfoTmp.wrData.op = wr.op;
    wrInfoTmp.wrData.sendFlags = wr.sendFlag;
    wrInfoTmp.wrData.immData = 0;
    wrInfoTmp.wrData.wrId = 0xFF;
    wrInfoTmp.wrData.rkey = wr.rkey;
    wrInfoTmp.wrData.aux = aux;

    CHK_RET(SendWrList(1U, &wrInfoTmp, &opRsp));
    u32 dbIndex = static_cast<u32>(opRsp.db.dbIndex);
    u64 dbInfo = static_cast<u64>(opRsp.db.dbInfo);
    HCCL_DEBUG("dbIndex = [%d]", dbIndex);
    RdmaTaskInfo taskInfo = {};
    taskInfo.remoteRank = machinePara_.remoteWorldRank;
    taskInfo.rdmaType = (wqeType == WqeType::WQE_TYPE_DATA) ? RdmaType::RDMA_SEND_PAYLOAD : RdmaType::RDMA_SEND_NOTIFY;
    wrInfoTmp.type = static_cast<u64>(wqeType);
    wrInfoTmp.wrDataAddr = notifyAddr;
    wrInfoTmp.notifyId = notifyId;
    taskInfo.wrInfos.push_back(wrInfoTmp);

    ret = dispatcher_->RdmaSend(dbIndex, dbInfo, stream, taskInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportDeviceIbverbs][RdmaSendAsync]errNo[0x%016llx] In lbv exp op base mode, "
            "rdma send failed. dbIndex[%u] dbInfo[%llu], addr[%llu]",
            HCCL_ERROR_CODE(ret), dbIndex, dbInfo, notifyAddr),
        ret);
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::GetWrDataAddr(void* dstAddr, WqeType wqeType, u64& wrDataAddr, u32& notifyId)
{
    switch (wqeType) {
        case WqeType::WQE_TYPE_DATA:
        case WqeType::WQE_TYPE_DATA_WITH_NOTIFY:
        case WqeType::WQE_TYPE_DATA_WITH_REDUCE:
        case WqeType::WQE_TYPE_READ_DATA:
            wrDataAddr = reinterpret_cast<u64>(dstAddr);
            notifyId = INVALID_UINT;
            break;
        case WqeType::WQE_TYPE_DATA_NOTIFY:
            wrDataAddr = reinterpret_cast<u64>(remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].addr);
            notifyId = remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].notifyId;
            break;
        case WqeType::WQE_TYPE_ACK_NOTIFY:
            wrDataAddr = reinterpret_cast<u64>(remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].addr);
            notifyId = remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].notifyId;
            break;
        case WqeType::WQE_TYPE_DATA_ACK_NOTIFY:
            wrDataAddr = reinterpret_cast<u64>(remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].addr);
            notifyId = remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].notifyId;
            break;
        default:
            HCCL_ERROR("[Get][WrDataAddr]error wqeType[%d]", wqeType);
            return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG(
        "%s dstAddr:%p, wqeType:%d, wrDataAddr:%llu, notifyId:%u", __func__, dstAddr, wqeType, wrDataAddr, notifyId);
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::TxSendWqe(
    void* dstMemPtr, u32 dstKey, const void* srcMemPtr, u32 srcKey, u64 srcMemSize, Stream& stream, WqeType wqeType)
{
    struct SgList list = {};
    struct SendWr wr = {};
    // 构造wr信息
    list.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(srcMemPtr));
    list.len = srcMemSize;
    list.lkey = srcKey;

    wr.bufList = &list;
    wr.bufNum = 1; /* 此处list只有一个，设置为1 */
    wr.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(dstMemPtr));
    wr.rkey = dstKey;
    wr.op = 0; /* RDMA_WRITE: 0 */
    wr.sendFlag = fence_ ? (RA_SEND_SIGNALED | RA_SEND_FENCE) : RA_SEND_SIGNALED;
    fence_ = false;

    // 获取notify偏移地址，对于发送数据时，偏移地址为0
    u64 wrDataAddr = 0;
    u32 notifyId = INVALID_UINT;
    CHK_RET(GetWrDataAddr(dstMemPtr, wqeType, wrDataAddr, notifyId));

    // RDMA异步发送
    CHK_RET(RdmaSendAsync(wr, stream, wqeType, wrDataAddr, notifyId));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::RxAsync(
    [[maybe_unused]] UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream)
{
    CHK_SMART_PTR_NULL(stream);
    // 等待TS把任务处理完成
    HCCL_DEBUG("RX dst[%p] len[%llu] srcOffset[%llu]", dst, len, srcOffset);
    u32 actualMultiQpNum = 1;
    const u32 KByteToByte = 1024; // 1024 多QP阈值单位是KB
    if (len / qpsPerConnection_ > multiQpThreshold_ * KByteToByte) {
        actualMultiQpNum = qpsPerConnection_;
    } else {
        u32 quotient = len / (multiQpThreshold_ * KByteToByte);
        u32 remainder = len % (multiQpThreshold_ * KByteToByte);
        actualMultiQpNum = quotient + (remainder != 0 ? 1 : 0);
    }
    if (UseMultiQp() && actualMultiQpNum != 1 && actualMultiQpNum <= qpsPerConnection_ && len != 0) {
        for (u32 i = 0; i < actualMultiQpNum; i++) {
            CHK_RET(dispatcher_->SignalWait(
                multiQpDataNotify_[i]->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank,
                INVALID_VALUE_STAGE, false, multiQpDataNotify_[i]->notifyId_));
        }
    } else {
        CHK_RET(dispatcher_->SignalWait(
            dataNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank, INVALID_VALUE_STAGE,
            false, dataNotify_->notifyId_));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream)
{
    CHK_PRT_RET(rxMems.size() == 0, HCCL_ERROR("Invalid rxMem size[%u]", rxMems.size()), HCCL_E_PARA);
    CHK_SMART_PTR_NULL(stream);
    for (auto& mem : rxMems) {
        HCCL_DEBUG("RX dst[%p] len[%llu] dstOffset[%llu]", mem.dst, mem.len, mem.srcOffset);
    }
    u32 maxLength = 0;
    for (u32 i = 0; i < rxMems.size(); i++) {
        if (rxMems[i].len > maxLength) {
            maxLength = rxMems[i].len;
        }
    }

    CHK_RET(RxAsync(rxMems[0].srcMemType, rxMems[0].srcOffset, rxMems[0].dst, maxLength, stream));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::DataReceivedAck(Stream& stream)
{
    CHK_RET(PostFinAck(stream));
    CHK_RET(WaitFinAck(stream));

    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::TxWaitDone([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

/* 发送ack消息(同步模式) */
HcclResult TransportDeviceIbverbs::TxAck(Stream& stream)
{
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].addr,
        remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].lkey,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey, notifySize_, stream, WqeType::WQE_TYPE_ACK_NOTIFY));
    return HCCL_SUCCESS;
}

/* 接收ack消息(同步模式) */
HcclResult TransportDeviceIbverbs::RxAck(Stream& stream)
{
    CHK_RET(dispatcher_->SignalWait(
        ackNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank, INVALID_VALUE_STAGE, false,
        ackNotify_->notifyId_));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::TxDataSignal(Stream& stream)
{
    // 发送data notify同步信息
    void* remoteNotifyaddr = remoteDataNotifyMsg_.addr;
    HcclResult ret = TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].addr,
        remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].lkey,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey, notifySize_, stream, WqeType::WQE_TYPE_DATA_NOTIFY);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportDeviceIbverbs][TxDataSignal]errNo[0x%016llx] In ibv tx data signal, send notify "
            "wqe failed. dstMemPtr[%p], srcMemPtr[%p], srcMemSize[%llu]",
            HCCL_ERROR_CODE(ret), remoteNotifyaddr, notifyValueAddr_, notifySize_),
        ret);
    // 每发送一个data notify wqe, count 自增
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::RxDataSignal(Stream& stream)
{
    /* 等待send_ready_event事件 */
    CHK_RET(dispatcher_->SignalWait(
        dataNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank, INVALID_VALUE_STAGE,
        false, dataNotify_->notifyId_));
    return HCCL_SUCCESS;
}

/* 发送ack消息(同步模式) */
HcclResult TransportDeviceIbverbs::TxPrepare(Stream& stream)
{
    CHK_RET(dispatcher_->SignalWait(
        ackNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank, INVALID_VALUE_STAGE, false,
        ackNotify_->notifyId_));
    return HCCL_SUCCESS;
}

/* 接收ack消息(同步模式) */
HcclResult TransportDeviceIbverbs::RxPrepare(Stream& stream)
{
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].addr,
        remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].lkey,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey, notifySize_, stream, WqeType::WQE_TYPE_ACK_NOTIFY));
    return HCCL_SUCCESS;
}

HcclResult
TransportDeviceIbverbs::TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream)
{
    CHK_SMART_PTR_NULL(stream);
    std::vector<WrInformation> wrInfoVec;
    struct WrAuxInfo aux = {};
    HCCL_DEBUG("TX src[%p] len[%llu] dstOffset[%llu]", src, len, dstOffset);

    if (len > 0) {
        CHK_PTR_NULL(src);
        CHK_RET(TxPayLoad(dstMemType, dstOffset, src, len, WqeType::WQE_TYPE_DATA, aux, wrInfoVec));
    }

    CHK_RET(RdmaSendAsync(wrInfoVec, stream, false));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::RxData(
    [[maybe_unused]] UserMemType srcMemType, [[maybe_unused]] u64 srcOffset, [[maybe_unused]] void* dst,
    [[maybe_unused]] u64 len, [[maybe_unused]] Stream& stream)
{
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::TxDone(Stream& stream)
{
    // 发送数据接收确认notify
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].addr,
        remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].lkey,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey, notifySize_, stream, WqeType::WQE_TYPE_DATA_NOTIFY));
    // 接收数据接收确认notify
    CHK_RET(dispatcher_->SignalWait(
        dataAckNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank, INVALID_VALUE_STAGE,
        false, dataAckNotify_->notifyId_));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::RxDone(Stream& stream)
{
    // 接收数据接收确认notify
    CHK_RET(dispatcher_->SignalWait(
        dataNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank, INVALID_VALUE_STAGE,
        false, dataNotify_->notifyId_));

    // 发送数据接收确认notify
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].addr,
        remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].lkey,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey, notifySize_, stream,
        WqeType::WQE_TYPE_DATA_ACK_NOTIFY));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::PostReady(Stream& stream)
{
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].addr,
        remoteMemMsg_[static_cast<u32>(MemType::ACK_NOTIFY_MEM)].lkey,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey, notifySize_, stream, WqeType::WQE_TYPE_ACK_NOTIFY));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::WaitReady(Stream& stream)
{
    CHK_RET(dispatcher_->SignalWait(
        ackNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank, INVALID_VALUE_STAGE, false,
        ackNotify_->notifyId_));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::PostFin(Stream& stream)
{
    // 发送data notify同步信息
    void* remoteNotifyaddr = remoteDataNotifyMsg_.addr;
    HcclResult ret = TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].addr,
        remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].lkey,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey, notifySize_, stream, WqeType::WQE_TYPE_DATA_NOTIFY);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportDeviceIbverbs][PostFin]errNo[0x%016llx] In ibv tx data signal, send notify "
            "wqe failed. dstMemPtr[%p], srcMemPtr[%p], srcMemSize[%llu]",
            HCCL_ERROR_CODE(ret), remoteNotifyaddr, notifyValueAddr_, notifySize_),
        ret);
    // 每发送一个data notify wqe, count 自增
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::WaitFin(Stream& stream)
{
    CHK_RET(dispatcher_->SignalWait(
        dataNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank, INVALID_VALUE_STAGE,
        false, dataNotify_->notifyId_));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::PostFinAck(Stream& stream)
{
    CHK_RET(TxSendWqe(
        remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].addr,
        remoteMemMsg_[static_cast<u32>(MemType::DATA_ACK_NOTIFY_MEM)].lkey,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr,
        memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey, notifySize_, stream,
        WqeType::WQE_TYPE_DATA_ACK_NOTIFY));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::WaitFinAck(Stream& stream)
{
    CHK_RET(dispatcher_->SignalWait(
        dataNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank, INVALID_VALUE_STAGE,
        false, dataAckNotify_->notifyId_));
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::ResolveRdmaAddrsAndKeys(RdmaAddrKeyResolveParam& param)
{
    param.transLocalAddr = const_cast<void*>(param.localAddr);
    param.transRemoteAddr = const_cast<void*>(param.remoteAddr);
    if (useMemDetailsLookup_) {
        return ResolveRdmaKeysFromMemDetails(param);
    }
    return ResolveRdmaKeysFromIoMemRanges(param);
}

HcclResult TransportDeviceIbverbs::ResolveRdmaKeysFromMemDetails(RdmaAddrKeyResolveParam& param)
{
    auto rf = remoteMemDetailsRmaMgr_->Find(MakeMemLookupKey(param.remoteAddr, param.length));
    if (!rf.first || rf.second == nullptr) {
        HCCL_ERROR(
            "[TransportDeviceIbverbs]Can't find remoteBuffer key by addr and size {%p, %llu}, "
            "registered remote MR count[%zu]",
            param.remoteAddr, param.length, remoteMemDetailsRmaMgr_->size());
        return HCCL_E_INTERNAL;
    }
    param.dstKey = rf.second->key;
    param.transRemoteAddr = LogicalPtrToDevPtr(*rf.second, param.remoteAddr);

    auto lf = localMemDetailsRmaMgr_->Find(MakeMemLookupKey(param.localAddr, param.length));
    CHK_PRT_RET(
        !lf.first || lf.second == nullptr,
        HCCL_ERROR(
            "[TransportDeviceIbverbs]Can't find localBuffer key by addr and size {%p, %llu}, "
            "registered local MR count[%zu]",
            param.localAddr, param.length, localMemDetailsRmaMgr_->size()),
        HCCL_E_INTERNAL);
    param.srcKey = lf.second->key;
    param.transLocalAddr = LogicalPtrToDevPtr(*lf.second, param.localAddr);
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::ResolveRdmaKeysFromIoMemRanges(RdmaAddrKeyResolveParam& param)
{
    u64 dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(param.remoteAddr));
    u64 remoteInputAddr
        = static_cast<u64>(reinterpret_cast<uintptr_t>(remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].addr));
    u64 remoteInputSize = localInputMem_.size;
    u32 remoteInputKey = remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].lkey;
    u64 remoteOutputAddr
        = static_cast<u64>(reinterpret_cast<uintptr_t>(remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].addr));
    u64 remoteOutputSize = localOutputMem_.size;
    u32 remoteOutputKey = remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].lkey;
    if (dstAddr >= remoteInputAddr && dstAddr < remoteInputAddr + remoteInputSize) {
        param.dstKey = remoteInputKey;
    } else if (dstAddr >= remoteOutputAddr && dstAddr <= remoteOutputAddr + remoteOutputSize) {
        param.dstKey = remoteOutputKey;
    } else {
        HCCL_ERROR(
            "[TransportDeviceIbverbs][TxAsync]src_ptr=%p is out of range, inputmem src[%p], size[%llu];"
            " outputmem src[%p] size[%llu]",
            param.remoteAddr, remoteInputAddr, remoteInputSize, remoteOutputAddr, remoteOutputSize);
        return HCCL_E_INTERNAL;
    }

    u64 srcAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(param.localAddr));
    if (srcAddr >= localInputMem_.addr && srcAddr < localInputMem_.addr + localInputMem_.size) {
        param.srcKey = localInputMem_.key;
    } else if (srcAddr >= localOutputMem_.addr && srcAddr <= localOutputMem_.addr + localOutputMem_.size) {
        param.srcKey = localOutputMem_.key;
    } else {
        HCCL_ERROR(
            "[TransportDeviceIbverbs][TxAsync]src_ptr=%p is out of range, inputmem src[%p], size[%llu];"
            " outputmem src[%p] size[%llu]",
            param.localAddr, localInputMem_.addr, localInputMem_.size, localOutputMem_.addr, localOutputMem_.size);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::WriteCommon(
    const void* remoteAddr, const void* localAddr, u64 length, Stream& stream, WqeType wqeType, struct WrAuxInfo& aux)
{
    if (machinePara_.dctxPtr != nullptr) {
        CHK_RET(SetDispatcherCtx(static_cast<DispatcherCtxPtr>(machinePara_.dctxPtr)));
    }

    std::vector<WrInformation> wrInfoVec;
    HCCL_DEBUG("write localAddr[%p] remoteAddr[%p] len[%llu]", localAddr, remoteAddr, length);

    if (localAddr != nullptr) {
        // 为保证单算子下不同数据量下子图的结构相同，zero byte message 时也需要下发task
        u32 txSendDataTimes = (length == 0) ? 1 : (length + RDMA_SEND_MAX_SIZE - 1) / RDMA_SEND_MAX_SIZE;

        RdmaAddrKeyResolveParam resolve{};
        resolve.remoteAddr = remoteAddr;
        resolve.localAddr = localAddr;
        resolve.length = length;
        CHK_RET(ResolveRdmaAddrsAndKeys(resolve));

        CHK_RET(ConstructPayLoadWqe(
            resolve.transRemoteAddr, resolve.dstKey, resolve.transLocalAddr, resolve.srcKey, length, wqeType, aux,
            wrInfoVec, txSendDataTimes));
    }
    u32 maxLength = 0;
    for (u32 i = 0; i < wrInfoVec.size(); i++) {
        if (wrInfoVec[i].wrData.memList.len > maxLength) {
            maxLength = wrInfoVec[i].wrData.memList.len;
        }
    }

    u32 actualMultiQpNum = GetActualQpNum(maxLength);

    HCCL_DEBUG(
        "[TransportDeviceIbverbs][TxSendDataAndNotify] UseMultiQp[%d] MultiQpNum[%u] actualMultiQpNum[%u] "
        "maxLength[%u]",
        UseMultiQp(), qpsPerConnection_, actualMultiQpNum, maxLength);
    if (UseMultiQp() && actualMultiQpNum != 1 && actualMultiQpNum <= qpsPerConnection_ && maxLength != 0) {
        std::vector<std::vector<WrInformation>> multiQpWqeInfoVct(actualMultiQpNum, wrInfoVec);
        for (u32 i = 0; i < wrInfoVec.size(); i++) {
            WrInformation tmpWqeInfo = wrInfoVec[i];
            u32 curLen = tmpWqeInfo.wrData.memList.len;
            std::vector<u32> splittedLen = RdmaLengthSplit(curLen, actualMultiQpNum);
            uint64_t curSrcAddr = tmpWqeInfo.wrData.memList.addr;
            uint64_t curDstAddr = tmpWqeInfo.wrData.dstAddr;
            for (u32 qpIndex = 0; qpIndex < actualMultiQpNum; qpIndex++) {
                multiQpWqeInfoVct[qpIndex][i].wrData.memList.len = splittedLen[qpIndex];
                multiQpWqeInfoVct[qpIndex][i].wrData.memList.addr = curSrcAddr;
                multiQpWqeInfoVct[qpIndex][i].wrData.dstAddr = curDstAddr;
                curSrcAddr += splittedLen[qpIndex];
                curDstAddr += splittedLen[qpIndex];
            }
        }

        // useOneDoorbell 配置成true。最后一个payload去按doorbell
        for (u32 qpIndex = 0; qpIndex < actualMultiQpNum; qpIndex++) {
            CHK_RET(RdmaSendAsync(
                multiQpWqeInfoVct[qpIndex], stream, true, qpIndex)); // 多QP使用同一个stream异步doorbell触发
        }
    } else {
        CHK_RET(RdmaSendAsync(wrInfoVec, stream, GetUseOneDoorbellValue()));
    }
    return HCCL_SUCCESS;

#ifndef CCL_LLT
    CHK_RET(RdmaSendAsync(wrInfoVec, stream, GetUseOneDoorbellValue()));
#endif
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::WriteAsync(
    struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream)
{
    struct WrAuxInfo aux = {};
    return WriteCommon(remoteBuf.addr, localBuf.addr, remoteBuf.size, stream, WqeType::WQE_TYPE_DATA, aux);
}

HcclResult TransportDeviceIbverbs::ReadAsync(
    struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream)
{
    HCCL_DEBUG("[TransportDeviceIbverbs][ReadAsync]");
    struct WrAuxInfo aux = {};
    return WriteCommon(remoteBuf.addr, localBuf.addr, remoteBuf.size, stream, WqeType::WQE_TYPE_READ_DATA, aux);
}

HcclResult TransportDeviceIbverbs::ResolveTransferDesc(
    const HcommBatchTransferDesc& desc, const void*& remoteAddr, const void*& localAddr, u64& length, WqeType& wqeType,
    [[maybe_unused]] struct WrAuxInfo& aux)
{
    if (desc.transType == HCOMM_TRANSFER_TYPE_WRITE) {
        CHK_PTR_NULL(desc.transferInfo.write.dst);
        CHK_PTR_NULL(desc.transferInfo.write.src);
        remoteAddr = desc.transferInfo.write.dst;
        localAddr = desc.transferInfo.write.src;
        length = desc.transferInfo.write.len;
        wqeType = WqeType::WQE_TYPE_DATA;
    } else if (desc.transType == HCOMM_TRANSFER_TYPE_READ) {
        CHK_PTR_NULL(desc.transferInfo.read.dst);
        CHK_PTR_NULL(desc.transferInfo.read.src);
        remoteAddr = desc.transferInfo.read.src;
        localAddr = desc.transferInfo.read.dst;
        length = desc.transferInfo.read.len;
        wqeType = WqeType::WQE_TYPE_READ_DATA;
    } else {
        HCCL_ERROR("[ResolveTransferDesc] Unsupported transType[%d].", desc.transType);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::SubmitWqeBatch(std::vector<WrInformation>& wrInfoVec, Stream& stream)
{
    u32 maxLength = 0;
    for (u32 i = 0; i < wrInfoVec.size(); i++) {
        if (wrInfoVec[i].wrData.memList.len > maxLength) {
            maxLength = wrInfoVec[i].wrData.memList.len;
        }
    }

    u32 actualMultiQpNum = GetActualQpNum(maxLength);
    if (UseMultiQp() && actualMultiQpNum != 1 && actualMultiQpNum <= qpsPerConnection_ && maxLength != 0) {
        CHK_RET(TxSendDataAndNotifyWithMultiQP(wrInfoVec, actualMultiQpNum, stream, true));
    } else {
        CHK_RET(TxSendDataAndNotifyWithSingleQP(wrInfoVec, stream, true));
    }
    return HCCL_SUCCESS;
}

HcclResult
TransportDeviceIbverbs::BatchTransferImpl(const HcommBatchTransferDesc* transferDescs, uint32_t descNum, Stream& stream)
{
    if (machinePara_.dctxPtr != nullptr) {
        CHK_RET(SetDispatcherCtx(static_cast<DispatcherCtxPtr>(machinePara_.dctxPtr)));
    }

    CHK_PTR_NULL(transferDescs);
    std::vector<WrInformation> wrInfoVec;
    for (uint32_t i = 0; i < descNum; i++) {
        const void* localAddr = nullptr;
        const void* remoteAddr = nullptr;
        u64 length = 0;
        WqeType wqeType = WqeType::WQE_TYPE_DATA;
        struct WrAuxInfo aux = {};
        CHK_RET(ResolveTransferDesc(transferDescs[i], remoteAddr, localAddr, length, wqeType, aux));

        HCCL_DEBUG(
            "[BatchTransferImpl] index[%u] localAddr[%p] remoteAddr[%p] len[%llu] wqeType[%d]", i, localAddr,
            remoteAddr, length, static_cast<int>(wqeType));

        if (localAddr != nullptr) {
            bool isLast = (i == descNum - 1);
            u32 txSendDataTimes = (length == 0) ? 1 : (length + RDMA_SEND_MAX_SIZE - 1) / RDMA_SEND_MAX_SIZE;
            RdmaAddrKeyResolveParam resolve{};
            resolve.remoteAddr = remoteAddr;
            resolve.localAddr = localAddr;
            resolve.length = length;
            CHK_RET(ResolveRdmaAddrsAndKeys(resolve));
            CHK_RET(ConstructPayLoadWqe(
                resolve.transRemoteAddr, resolve.dstKey, resolve.transLocalAddr, resolve.srcKey, length, wqeType, aux,
                wrInfoVec, txSendDataTimes, isLast));
        }
    }

    if (!wrInfoVec.empty()) {
        return SubmitWqeBatch(wrInfoVec, stream);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::BatchTransferAsync(
    const HcommBatchTransferDesc* transferDescs, uint32_t descNum, Stream& stream)
{
    return BatchTransferImpl(transferDescs, descNum, stream);
}

HcclResult TransportDeviceIbverbs::WriteReduceAsync(
    struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, const HcclDataType datatype,
    HcclReduceOp redOp, Stream& stream)
{
    struct WrAuxInfo aux = {};
    aux.dataType = RDMA_REDUCE_DATA_TYPE_TABLE[datatype];
    aux.reduceType = RDMA_REDUCE_OP_TYPE_TABLE[redOp];
    if (aux.dataType == static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID)
        || aux.reduceType == static_cast<uint8_t>(RdmaReduceOpType::RDMA_REDUCE_OP_INVALID)) {
        HCCL_ERROR(
            "unsupported data type [%s] or Reduce type [%s]", GetDataTypeEnumStr(datatype).c_str(),
            GetReduceOpEnumStr(redOp).c_str());
        return HCCL_E_INTERNAL;
    }

    return WriteCommon(remoteBuf.addr, localBuf.addr, remoteBuf.size, stream, WqeType::WQE_TYPE_DATA_WITH_REDUCE, aux);
}

HcclResult TransportDeviceIbverbs::Post(u32 notifyIdx, Stream& stream)
{
    // 校验notifyIdx有效性
    bool bRet = (notifyIdx >= notifyNum_);
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[TransportDeviceIbverbs][Post]notifyNum[%u], notifyIdx[%u] out of range[0, %u]", notifyNum_, notifyIdx,
            notifyNum_ - 1),
        HCCL_E_INTERNAL);

    // 每个QP发送一个指定idx的notify
    for (u32 i = 0; i < qpsPerConnection_; i++) {
        CHK_RET(TxSendWqe(
            userMultiQpRemoteNotifyMsg_[i][notifyIdx].addr, userMultiQpRemoteNotifyMsg_[i][notifyIdx].lkey,
            memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr,
            memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey, notifySize_, stream,
            WqeType::WQE_TYPE_DATA_WITH_NOTIFY));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::Wait(u32 notifyIdx, Stream& stream, const u32 timeOut)
{
    // 校验notifyIdx有效性
    bool bRet = (notifyIdx >= notifyNum_);
    CHK_PRT_RET(
        bRet,
        HCCL_ERROR(
            "[TransportDeviceIbverbs][Wait]notifyNum[%u], notifyIdx[%u] out of range[0, %u]", notifyNum_, notifyIdx,
            notifyNum_ - 1),
        HCCL_E_INTERNAL);

    // 单QP接收一个指定idx的notify
    // 每个qp接收一个指定idx的notify
    for (u32 i = 0; i < qpsPerConnection_; i++) {
        CHK_RET(dispatcher_->SignalWait(
            userMultiQpLocalNotify_[i][notifyIdx]->ptr(), stream, machinePara_.localUserrank,
            machinePara_.remoteWorldRank, INVALID_VALUE_STAGE, false, userMultiQpLocalNotify_[i][notifyIdx]->notifyId_,
            timeOut));
    }
    return HCCL_SUCCESS;
}

bool TransportDeviceIbverbs::UseMultiQp() { return qpsPerConnection_ != 1; }

u32 TransportDeviceIbverbs::GetActualQpNum(u32 maxLength)
{
    u32 actualMultiQpNum = 1;
    const u32 KByteToByte = 1024; // 1024 多QP阈值单位是KB
    if (maxLength / qpsPerConnection_ >= multiQpThreshold_ * KByteToByte) {
        actualMultiQpNum = qpsPerConnection_;
    } else {
        u32 quotient = maxLength / (multiQpThreshold_ * KByteToByte);
        u32 remainder = maxLength % (multiQpThreshold_ * KByteToByte);
        actualMultiQpNum = quotient + (remainder != 0 ? 1 : 0);
    }

    return actualMultiQpNum;
}

HcclResult TransportDeviceIbverbs::TxSendDataAndNotifyWithMultiQP(
    std::vector<WrInformation>& wqeInfoVec, u32 actualMultiQpNum, Stream& stream, [[maybe_unused]] bool useOneDoorbell)
{
    // vector<WrInformation> 是一个vector的原因是 单个wqe只能发2GB数据，如果超过2GB，就拆分到多个WqeInfo中了
    // 多QP下，对每个WqeInfo都进行多QP切分，然后在收发每一个QP的数据
    std::vector<std::vector<WrInformation>> multiQpWqeInfoVct(actualMultiQpNum, wqeInfoVec);
    for (u32 i = 0; i < wqeInfoVec.size(); i++) {
        WrInformation tmpWqeInfo = wqeInfoVec[i];
        u32 curLen = tmpWqeInfo.wrData.memList.len;
        std::vector<u32> splittedLen = RdmaLengthSplit(curLen, actualMultiQpNum);
        uint64_t curSrcAddr = tmpWqeInfo.wrData.memList.addr;
        uint64_t curDstAddr = tmpWqeInfo.wrData.dstAddr;
        for (u32 qpIndex = 0; qpIndex < actualMultiQpNum; qpIndex++) {
            multiQpWqeInfoVct[qpIndex][i].wrData.memList.len = splittedLen[qpIndex];
            multiQpWqeInfoVct[qpIndex][i].wrData.memList.addr = curSrcAddr;
            multiQpWqeInfoVct[qpIndex][i].wrData.dstAddr = curDstAddr;
            curSrcAddr += splittedLen[qpIndex];
            curDstAddr += splittedLen[qpIndex];
        }
    }
    // 给每个QP最后增加一个属于该QP的DataNotify
    for (u32 qpIndex = 0; qpIndex < actualMultiQpNum; qpIndex++) {
        struct WrAuxInfo aux = {};
        void* remoteNotifyaddr = multiQpDataNotifyRemoteMemMsg_[qpIndex].addr;
        CHK_RET(AddWrList(
            remoteNotifyaddr, notifyValueAddr_, notifySize_, memMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey,
            remoteMemMsg_[static_cast<u32>(MemType::DATA_NOTIFY_MEM)].lkey, WqeType::WQE_TYPE_DATA_NOTIFY, aux,
            multiQpWqeInfoVct[qpIndex]));
    }
    // useOneDoorbell 配置成true。最后一个payload去按doorbell
    for (u32 qpIndex = 0; qpIndex < actualMultiQpNum; qpIndex++) {
        CHK_RET(
            RdmaSendAsync(multiQpWqeInfoVct[qpIndex], stream, true, qpIndex)); // 多QP使用同一个stream异步doorbell触发
    }
    return HCCL_SUCCESS;
}
HcclResult TransportDeviceIbverbs::GetTransportId(u32& id)
{
    struct ibv_qp* qp = reinterpret_cast<struct ibv_qp*>(combineAiQpInfo_.aiQpInfo.aiQpAddr);
    if (nullptr != qp) {
        id = qp->qp_num;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDeviceIbverbs::HnsPostSend(
    const TransportDeviceNormalData& ibvData, struct MemDetails* localMems, struct MemDetails* remoteMems, u32 memNum,
    HcclWrOpCode opCode, u64& dbInfo, bool fence)
{
    CHK_PTR_NULL(localMems);
    CHK_PTR_NULL(remoteMems);

    const uint32_t SEND_WR_LEN = 8;
    uint32_t last = memNum - 1;
    CHK_PRT_RET(
        memNum > SEND_WR_LEN,
        HCCL_ERROR("[TransportDeviceIbverbs][HnsPostSend] buffer size is:%u over SEND_WR_LEN: %u", memNum, SEND_WR_LEN),
        HCCL_E_PARA);
    struct ibv_send_wr sendWr[SEND_WR_LEN] = {};
    struct ibv_sge sge[SEND_WR_LEN] = {};

    for (uint32_t index = 0; index < memNum; index++) {
        // 设置WR的SGE
        sge[index].addr = reinterpret_cast<u64>(localMems[index].addr);
        sge[index].length = remoteMems[index].size;
        sge[index].lkey = localMems[index].key;

        // 设置WR属性
        sendWr[index].wr_id = wrIdOffset_.fetch_add(1, std::memory_order_relaxed);
        sendWr[index].num_sge = 1; // 只有一个SGE
        sendWr[index].sg_list = &sge[index];
        sendWr[index].wr.rdma.remote_addr = reinterpret_cast<u64>(remoteMems[index].addr);
        sendWr[index].wr.rdma.rkey = remoteMems[index].key;
        sendWr[index].next = (index == last) ? nullptr : &sendWr[index + 1]; // 第一个WR指向第二个WR
        sendWr[index].send_flags = (index == last) ?
                                       (fence ? (IBV_SEND_SIGNALED | IBV_SEND_FENCE) : IBV_SEND_SIGNALED) :
                                       0; // 最后一个WR才需要回复CQE
        sendWr[index].opcode = static_cast<enum ibv_wr_opcode>(opCode);
        HCCL_DEBUG(
            "[TransportDeviceIbverbs][HnsPostSend] Direct ibv_post_send[%llu], opcode=[0x%x], "
            "remote_addr=[0x%llx], size=[%u], fence[%u]",
            wrIdOffset_.load(), sendWr[index].opcode, sendWr[index].wr.rdma.remote_addr, sendWr[index].sg_list->length,
            fence);
    }

    struct ibv_send_wr* badWr = nullptr;
    struct WrExpRsp exp_rsp = {};
    struct ibv_qp* qp = reinterpret_cast<struct ibv_qp*>(ibvData.qpInfo.qpPtr);
    CHK_PTR_NULL(qp);
    HCCL_DEBUG(
        "[TransportDeviceIbverbs][HnsPostSend] qp=%p, handle=%u, qp_num=%u, qp_type=%d, qp_stat=%d", qp, qp->handle,
        qp->qp_num, qp->qp_type, qp->state);
    HcclResult ret = HrtHnsIbvExpPostSend(qp, &sendWr[0], &badWr, &exp_rsp);
    HCOMM_DSB();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN,
        HCCL_ERROR(
            "[TransportDeviceIbverbs][HnsPostSend] failed, qp=%p, handle=%u, qp_num=%u, qp_type=%d, qp_stat=%d", qp,
            qp->handle, qp->qp_num, qp->qp_type, qp->state),
        ret);
    if (ret == HCCL_SUCCESS) {
        dbInfo = exp_rsp.db_info;
    }
    return ret;
}

HcclResult TransportDeviceIbverbs::Drain(Stream& stream)
{
    CHK_PTR_NULL(dataNotify_);
    CHK_PTR_NULL(remoteMemMsg_[MemType::NOTIFY_SRC_MEM].addr);
    CHK_PTR_NULL(memMsg_[MemType::DATA_NOTIFY_MEM].addr);

    struct SgList list = {};
    struct SendWr wr = {};
    list.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(memMsg_[MemType::DATA_NOTIFY_MEM].addr));
    list.len = static_cast<u32>(memMsg_[MemType::DATA_NOTIFY_MEM].len);
    list.lkey = memMsg_[MemType::DATA_NOTIFY_MEM].lkey;

    wr.bufList = &list;
    wr.bufNum = 1; /* 此处list只有一个，设置为1 */
    wr.dstAddr = reinterpret_cast<u64>(remoteMemMsg_[MemType::NOTIFY_SRC_MEM].addr);
    wr.rkey = remoteMemMsg_[MemType::NOTIFY_SRC_MEM].lkey;
    wr.op = RaWrOpcode::RA_WR_RDMA_READ;
    wr.sendFlag = RA_SEND_SIGNALED | RA_SEND_FENCE; // fence

    CHK_RET(RdmaSendAsync(wr, stream, WqeType::WQE_TYPE_DATA_NOTIFY, wr.dstAddr, INVALID_UINT));
    CHK_RET(dispatcher_->SignalWait(
        dataNotify_->ptr(), stream, machinePara_.localUserrank, machinePara_.remoteWorldRank, INVALID_VALUE_STAGE,
        false, dataNotify_->notifyId_));
    return HCCL_SUCCESS;
}
} // namespace hccl

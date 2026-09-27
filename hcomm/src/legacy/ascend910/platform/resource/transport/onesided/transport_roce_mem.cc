/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_roce_mem.h"
#include "log.h"
#include "adapter_hal.h"
#include "adapter_hccp.h"
#include "adapter_rts.h"
#include "network_manager_pub.h"
#include "dispatcher_pub.h"
#include "hccl_network.h"
#include "device_capacity.h"
#include "externalinput.h"

namespace hccl {
using namespace std;
using LocalRdmaRmaBufferMgr = NetDevContext::LocalRdmaRmaBufferMgr;

constexpr s32 REG_VALID = 1;
constexpr u32 WAIT_LINK_BUILD_DELAY_TIME_US = 10;
constexpr s32 QP_FLAG_RC = 0;         // flag: 0 = RC, 1= UD，其它预留
constexpr s32 OPBASE_QP_MODE_EXT = 4; // 单算子模式(910B/910_93)的QP
constexpr u32 WR_NUM = 1;             // 当前只支持一个WR
std::atomic<uint64_t> TransportRoceMem::sendWrHandle{0};
TransportRoceMem::TransportRoceMem(
    const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx, const HcclDispatcher& dispatcher,
    AttrInfo& attrInfo, bool aicpuUnfoldMode)
    : TransportMem(notifyPool, netDevCtx, dispatcher, attrInfo, aicpuUnfoldMode),
      trafficClass_(attrInfo.trafficClass),
      serviceLevel_(attrInfo.serviceLevel)
{}

TransportRoceMem::~TransportRoceMem()
{
    //  de rdmaSignal and Mr
    if (rdmaSignalMrHandle_ != nullptr) {
        HcclResult ret = HCCL_SUCCESS;
        ret = hrtRaDeRegGlobalMr(nicRdmaHandle_, rdmaSignalMrHandle_);
        rdmaSignalMrHandle_ = nullptr;
        if (ret != 0) {
            HCCL_ERROR("deReg rdmaSignal GlobalMr failed, ret[%d]", ret);
        }
    }
    // destroy notify mem and notifyMem Mr
    if (notifyValueMemMrHandle_ != nullptr) {
        HcclResult ret = HCCL_SUCCESS;
        ret = hrtRaDeRegGlobalMr(nicRdmaHandle_, notifyValueMemMrHandle_);
        notifyValueMemMrHandle_ = nullptr;
        if (ret != 0) {
            HCCL_ERROR("deReg notify Mem Mr failed, ret[%d]", ret);
        }
    }
    if (notifyMem_.ptr() != nullptr) {
        notifyMem_.free();
    }
    // destroy QP
    DestroyCqAndQp();
}

HcclResult TransportRoceMem::CheckRaSendNormalWrlistSupport()
{
    if ((LIKELY(isSupportRaSendNormalWrlist_ == SupportStatus::SUPPORT))) {
        // 已判断支持，直接返回成功，避免重复判断
        return HCCL_SUCCESS;
    } else if (isSupportRaSendNormalWrlist_ == SupportStatus::NOT_SUPPORT) {
        HCCL_ERROR("[TransportRoceMem]RDMALite and RaSendNormalWrlist are not supported");
        return HCCL_E_NOT_SUPPORT;
    } else {
        // 判断是否支持
        bool isSupportRDMALite = IsSupportRDMALite(deviceLogicId_);
        if (isSupportRDMALite) {
            // 支持RDMALite场景可直接支持
            isSupportRaSendNormalWrlist_ = SupportStatus::SUPPORT;
        } else {
            // 不支持RDMALite场景，需要根据opcode检查是否支持RaSendNormalWrlist接口
            bool isSupportTmp;
            CHK_RET(IsSupportRaSendNormalWrlist(isSupportTmp));
            isSupportRaSendNormalWrlist_ = isSupportTmp ? SupportStatus::SUPPORT : SupportStatus::NOT_SUPPORT;
        }
        if (isSupportRaSendNormalWrlist_ == SupportStatus::SUPPORT) {
            HCCL_RUN_INFO("[TransportRoceMem]RaSendNormalWrlist is supported");
        } else {
            HCCL_ERROR("[TransportRoceMem]RDMALite and RaSendNormalWrlist are not supported");
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
TransportRoceMem::ExchangeMemDesc(const RmaMemDescs& localMemDescs, RmaMemDescs& remoteMemDescs, u32& actualNumOfRemote)
{
    return DoExchangeMemDesc(localMemDescs, remoteMemDescs, actualNumOfRemote);
}

HcclResult TransportRoceMem::EnableMemAccess(const RmaMemDesc& remoteMemDesc, RmaMem& remoteMem)
{
    std::string tempDesc = RmaMemDescCopyToStr(remoteMemDesc);
    std::shared_ptr<RemoteRdmaRmaBuffer> tempRemoteBufferPtr = make_shared<RemoteRdmaRmaBuffer>();
    HcclResult ret = tempRemoteBufferPtr->Deserialize(tempDesc);
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS), HCCL_ERROR("[TransportRoceMem][EnableMemAccess]RemoteBuffer Deserialize failed."), ret);

    BufferKey<uintptr_t, u64> tempKey(
        reinterpret_cast<uintptr_t>(tempRemoteBufferPtr->GetAddr()), tempRemoteBufferPtr->GetSize());
    auto resultPair = remoteRdmaRmaBufferMgr_.Add(tempKey, tempRemoteBufferPtr);
    if (resultPair.first == remoteRdmaRmaBufferMgr_.End()) {
        // 输入key是表中某一个最相近key的交集、子集。返回空迭代器
        HCCL_ERROR("[TransportRoceMem][EnableMemAccess]The memory that is expected to enable"
                   " overlaps with the memory that has been enabled, please check params");
        return HCCL_E_INTERNAL;
    }

    // 已使能：输入key是表中某一最相近key的全集。 返回添加该key的迭代器，及false
    // 未使能：输入key是表中某一最相近key的空集。 返回添加成功的迭代器，及true
    std::string logInfo = resultPair.second ? "Enable memory access success!" :
                                              "Memory is already enabled, just increase the reference count.";
    HCCL_INFO("[TransportRoceMem][EnableMemAccess]:%s", logInfo.c_str());
    // 填充出参TransportRmaMem信息
    remoteMem.addr = tempRemoteBufferPtr->GetAddr();
    remoteMem.size = tempRemoteBufferPtr->GetSize();
    remoteMem.type = tempRemoteBufferPtr->GetMemType();
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::DisableMemAccess(const RmaMemDesc& remoteMemDesc)
{
    // 内存去使能管理
    std::string tempDesc = RmaMemDescCopyToStr(remoteMemDesc);
    RemoteRdmaRmaBuffer tempRemoteBuffer;
    HcclResult ret = tempRemoteBuffer.Deserialize(tempDesc);
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS), HCCL_ERROR("[TransportRoceMem][DisableMemAccess]RemoteBuffer Deserialize failed."), ret);

    BufferKey<uintptr_t, u64> tempKey(
        reinterpret_cast<uintptr_t>(tempRemoteBuffer.GetAddr()), tempRemoteBuffer.GetSize());
    try {
        if (remoteRdmaRmaBufferMgr_.Del(tempKey)) {
            // 删除成功：输入key是表中某一最相近key的全集，计数-1后为0，返回true
            HCCL_INFO("[TransportRoceMem][DisableMemAccess]Memory reference count is 0, disable memory access.");
        } else {
            // 删除失败：输入key是表中某一最相近key的全集，计数不为0（存在其他remoteRank使用），返回false
            HCCL_INFO("[TransportRoceMem][DisableMemAccess]Memory reference count is larger than 0"
                      "(used by other RemoteRank), do not disable memory.");
        }
        return HCCL_SUCCESS;
    } catch (std::out_of_range& e) {
        HCCL_ERROR("[TransportRoceMem][DisableMemAccess] catch RmaBufferMgr Del exception: %s", e.what());
        return HCCL_E_NOT_FOUND;
    }
}

HcclResult TransportRoceMem::FillRmaBufferSlice(
    const HcclBuf& localMem, const HcclBuf& remoteMem, RmaBufferSlice& localRmaBufferSlice,
    RmaBufferSlice& remoteRmaBufferSlice)
{
    void* remoteAddr = remoteMem.addr;
    void* localAddr = localMem.addr;
    u64 byteSize = std::min(remoteMem.len, localMem.len);
    auto localKey = BufferKey<uintptr_t, u64>(reinterpret_cast<uintptr_t>(localAddr), byteSize);

    NetDevContext* netDevCtx = static_cast<NetDevContext*>(netDevCtx_);
    std::shared_ptr<LocalRdmaRmaBufferMgr> localRmaBufferMgr = netDevCtx->GetlocalRdmaRmaBufferMgr();
    if (localRmaBufferMgr == nullptr) {
        HCCL_ERROR("[TransportRoceMem] can't get LocalRdmaRmaBufferMgr");
        return HCCL_E_INTERNAL;
    }
    auto localBuffer = localRmaBufferMgr->Find(localKey);
    CHK_PRT_RET(
        !localBuffer.first,
        HCCL_ERROR(
            "[TransportRoceMem][FillRmaBufferSlice] Can't find localBuffer by key {%p, %llu}", localAddr, byteSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        !localBuffer.second->GetAddr(),
        HCCL_ERROR("[TransportRoceMem][FillRmaBufferSlice] The addr of local Buffer or remote buffer is nullptr."),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        !localBuffer.second->GetDevAddr(),
        HCCL_ERROR("[TransportRoceMem][FillRmaBufferSlice]The dev addr of local Buffer is nullptr."), HCCL_E_NOT_FOUND);
    CHK_RET(CheckHcclBuffer(localAddr, localBuffer.second.get()));

    RmaBuffer* remoteBuffer = static_cast<RmaBuffer*>(remoteMem.handle);
    CHK_PRT_RET(
        !remoteBuffer->GetDevAddr(),
        HCCL_ERROR("[TransportRoceMem][FillRmaBufferSlice] The dev addr of remote buffer is nullptr."),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        !remoteBuffer->GetAddr(),
        HCCL_ERROR("[TransportRoceMem][FillRmaBufferSlice] The addr of remote buffer is nullptr."), HCCL_E_NOT_FOUND);
    CHK_RET(CheckHcclBuffer(remoteAddr, remoteBuffer));
    u64 localDataOffSet = static_cast<u8*>(localAddr) - static_cast<u8*>(localBuffer.second->GetAddr());
    u64 remoteDataOffSet = static_cast<u8*>(remoteAddr) - static_cast<u8*>(remoteBuffer->GetAddr());
    localRmaBufferSlice.addr = static_cast<void*>(static_cast<u8*>(localBuffer.second->GetDevAddr()) + localDataOffSet);
    localRmaBufferSlice.len = byteSize;
    localRmaBufferSlice.rmaBuffer = localBuffer.second;
    localRmaBufferSlice.memType = localBuffer.second->GetMemType();

    remoteRmaBufferSlice.addr = static_cast<void*>(static_cast<u8*>(remoteBuffer->GetDevAddr()) + remoteDataOffSet);
    remoteRmaBufferSlice.len = byteSize;
    std::shared_ptr<RmaBuffer> temp(
        remoteBuffer, []([[maybe_unused]] RmaBuffer* p) {}); // 在外部进行删除操作，内部不能用智能指针进行生命周期管理
    remoteRmaBufferSlice.rmaBuffer = temp;
    remoteRmaBufferSlice.memType = remoteBuffer->GetMemType();
    HCCL_INFO(
        "[TransportRoceMem][FillRmaBufferSlice] Local address before mapping is [%p], after mapping is [%p]."
        "Remote address before mapping is [%p], after mapping is [%p]. Datasize is [%llu].",
        localAddr, localRmaBufferSlice.addr, remoteAddr, remoteRmaBufferSlice.addr, byteSize);
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::FillRmaBufferSlice(
    const RmaOpMem& localMem, const RmaOpMem& remoteMem, RmaBufferSlice& localRmaBufferSlice,
    RmaBufferSlice& remoteRmaBufferSlice)
{
    void* remoteAddr = remoteMem.addr;
    void* localAddr = localMem.addr;
    u64 byteSize = std::min(remoteMem.size, localMem.size);
    auto localKey = BufferKey<uintptr_t, u64>(reinterpret_cast<uintptr_t>(localAddr), byteSize);
    auto remoteKey = BufferKey<uintptr_t, u64>(reinterpret_cast<uintptr_t>(remoteAddr), byteSize);

    NetDevContext* netDevCtx = static_cast<NetDevContext*>(netDevCtx_);
    std::shared_ptr<LocalRdmaRmaBufferMgr> localRmaBufferMgr = netDevCtx->GetlocalRdmaRmaBufferMgr();
    if (localRmaBufferMgr == nullptr) {
        HCCL_ERROR("[TransportRoceMem] can't get LocalRdmaRmaBufferMgr");
        return HCCL_E_INTERNAL;
    }
    auto localBuffer = localRmaBufferMgr->Find(localKey);
    CHK_PRT_RET(
        !localBuffer.first,
        HCCL_ERROR(
            "[TransportRoceMem][FillRmaBufferSlice] Can't find localBuffer by key {%p, %llu}", localAddr, byteSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        !localBuffer.second->GetAddr(),
        HCCL_ERROR("[TransportRoceMem][FillRmaBufferSlice] The addr of local Buffer or remote buffer is nullptr."),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        !localBuffer.second->GetDevAddr(),
        HCCL_ERROR("[TransportRoceMem][FillRmaBufferSlice]The dev addr of local Buffer is nullptr."), HCCL_E_NOT_FOUND);
    CHK_RET(CheckHcclBuffer(localAddr, localBuffer.second.get()));

    auto remoteBuffer = remoteRdmaRmaBufferMgr_.Find(remoteKey);
    CHK_PRT_RET(
        !remoteBuffer.first,
        HCCL_ERROR(
            "[TransportRoceMem][FillRmaBufferSlice]Can't find remoteBuffer by key {%p, %llu}", remoteAddr, byteSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        !remoteBuffer.second->GetDevAddr(),
        HCCL_ERROR("[TransportRoceMem][FillRmaBufferSlice]The dev addr of remote buffer is nullptr."),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(
        !remoteBuffer.second->GetAddr(),
        HCCL_ERROR("[TransportRoceMem][FillRmaBufferSlice]The addr of remote buffer is nullptr."), HCCL_E_NOT_FOUND);
    CHK_RET(CheckHcclBuffer(remoteAddr, remoteBuffer.second.get()));

    u64 localDataOffSet = static_cast<u8*>(localAddr) - static_cast<u8*>(localBuffer.second->GetAddr());
    u64 remoteDataOffSet = static_cast<u8*>(remoteAddr) - static_cast<u8*>(remoteBuffer.second->GetAddr());
    localRmaBufferSlice.addr = static_cast<void*>(static_cast<u8*>(localBuffer.second->GetDevAddr()) + localDataOffSet);
    localRmaBufferSlice.len = byteSize;
    localRmaBufferSlice.rmaBuffer = localBuffer.second;
    localRmaBufferSlice.memType = localBuffer.second->GetMemType();

    remoteRmaBufferSlice.addr
        = static_cast<void*>(static_cast<u8*>(remoteBuffer.second->GetDevAddr()) + remoteDataOffSet);
    remoteRmaBufferSlice.len = byteSize;
    remoteRmaBufferSlice.rmaBuffer = remoteBuffer.second;
    remoteRmaBufferSlice.memType = remoteBuffer.second->GetMemType();

    HCCL_INFO(
        "[TransportRoceMem][FillRmaBufferSlice] Local address before mapping is [%p], after mapping is [%p]."
        "Remote address before mapping is [%p], after mapping is [%p]. Datasize is [%llu].",
        localAddr, localRmaBufferSlice.addr, remoteAddr, remoteRmaBufferSlice.addr, byteSize);
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::SetSocket(const std::shared_ptr<HcclSocket>& socket)
{
    CHK_SMART_PTR_NULL(socket);
    if (socket->GetStatus() != HcclSocketStatus::SOCKET_OK) {
        HCCL_ERROR("sockets is not connected");
        return HCCL_E_PARA;
    }
    socket_ = socket;
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::GetRdmaHandle()
{
    RaResourceInfo raResourceInfo;
    CHK_RET(NetworkManager::GetInstance(deviceLogicId_).GetRaResourceInfo(raResourceInfo));
    auto it = raResourceInfo.nicSocketMap.find(socket_->GetLocalIp());
    if (it == raResourceInfo.nicSocketMap.end()) {
        HCCL_ERROR("[TransportRoceMem][GetRdmaHandle]nic socket handle did not found");
        return HCCL_E_PARA;
    }
    nicRdmaHandle_ = it->second.nicRdmaHandle;
    CHK_PTR_NULL(nicRdmaHandle_);
    HCCL_INFO(
        "TransportRoceMem GetNetworkResource deviceLogicId_[%d] nicRdmaHandle_[%p]", deviceLogicId_, nicRdmaHandle_);
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::CheckRdmaVal(void)
{
    DevType devType;
    const u32 HCCL_RDMA_TC_MAX = 255;
    const u32 HCCL_RDMA_SL_MAX = 7;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_910B || devType == DevType::DEV_TYPE_910_93) {
        if ((trafficClass_ != HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET) && (trafficClass_ > HCCL_RDMA_TC_MAX)) {
            HCCL_ERROR("[TransportRoceMem][CheckRdmaVal]trafficClass is invalid, trafficClass:%u", trafficClass_);
            return HCCL_E_PARA;
        }

        if ((serviceLevel_ != HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET) && (serviceLevel_ > HCCL_RDMA_SL_MAX)) {
            HCCL_ERROR("[TransportRoceMem][CheckRdmaVal]serviceLevel is invalid, serviceLevel:%u", serviceLevel_);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::ConnectImpl(s32 timeoutSec)
{
    // 增加1s的超时时间防止剩余超时时间不足
    s32 redundantTimeout = timeoutSec == INT_MAX ? timeoutSec : timeoutSec + 1;
    CHK_RET(GetRdmaHandle());
    CHK_RET(CreateCqAndQp());
    CHK_RET(CreatSignalMesg());
    CHK_RET(CreateNotifyValueBuffer());
    CHK_RET(ExchangeNotifyValueBuffer(redundantTimeout));
    CHK_RET(QpConnect(redundantTimeout));
    CHK_RET(WaitQPLinkComplete(redundantTimeout));
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::Connect(s32 timeoutSec)
{
    devicePhyId_ = (static_cast<NetDevContext*>(netDevCtx_))->GetPhyId();
    CHK_PRT_RET(devicePhyId_ == HOST_DEVICE_ID, HCCL_ERROR("[Connect] devicePhyId is invalid"), HCCL_E_INTERNAL);
    deviceLogicId_ = (static_cast<NetDevContext*>(netDevCtx_))->GetLogicId();
    CHK_PRT_RET(
        deviceLogicId_ == HOST_DEVICE_ID, HCCL_ERROR("deviceLogicId is same as host device id"), HCCL_E_INTERNAL);
    CHK_RET(CheckRdmaVal());
    CHK_PTR_NULL(dispatcher_);
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_RET(notifyPool_->RegisterOp(socket_->GetTag()));
    auto ret = ConnectImpl(timeoutSec);
    // 解注册之后再返回ret
    CHK_RET(notifyPool_->UnregisterOp(socket_->GetTag()));
    return ret;
}

HcclResult TransportRoceMem::TransportRdmaWithType(
    const RmaBufferSlice& localRmaBufferSlice, const RmaBufferSlice& remoteRmaBufferSlice, const rtStream_t& stream,
    const RdmaOp& rdmaOp)
{
    CHK_PTR_NULL(localRmaBufferSlice.addr);
    CHK_PTR_NULL(remoteRmaBufferSlice.addr);
    u64 processedOffset = 0;
    u64 remainingBytes = remoteRmaBufferSlice.len;
    u64 byteSizeChunk = 0;
    uint64_t localStartAddr = 0;
    uint64_t remoteStartAddr = 0;
    while (remainingBytes > 0) {
        localStartAddr = reinterpret_cast<uint64_t>(static_cast<u8*>(localRmaBufferSlice.addr) + processedOffset);
        remoteStartAddr = reinterpret_cast<uint64_t>(static_cast<u8*>(remoteRmaBufferSlice.addr) + processedOffset);
        byteSizeChunk = remainingBytes > MAX_RDMA_WQE_SIZE ? MAX_RDMA_WQE_SIZE : remainingBytes;
        std::shared_ptr<RemoteRdmaRmaBuffer> remoteRdmaRmaBuffer
            = dynamic_pointer_cast<RemoteRdmaRmaBuffer>(remoteRmaBufferSlice.rmaBuffer);
        std::shared_ptr<LocalRdmaRmaBuffer> localRdmaRmaBuffer
            = dynamic_pointer_cast<LocalRdmaRmaBuffer>(localRmaBufferSlice.rmaBuffer);
        struct WrInfo wr[WR_NUM];
        wr[0].wrId = sendWrHandle.fetch_add(1, std::memory_order_relaxed);
        wr[0].memList.addr = localStartAddr;
        wr[0].memList.len = byteSizeChunk;
        wr[0].memList.lkey = localRdmaRmaBuffer->GetKey();
        wr[0].dstAddr = remoteStartAddr;
        wr[0].rkey = remoteRdmaRmaBuffer->GetKey();
        wr[0].op = static_cast<u32>(rdmaOp);
        wr[0].sendFlags = 0;

        struct SendWrRsp opRsp[WR_NUM];

        HCCL_DEBUG(
            "Op type[%d], wr.wrId[%llu], src addr[%p], dest addr[%p], len[%u], sendflags[%d]", rdmaOp, wr[0].wrId,
            localRmaBufferSlice.addr, remoteRmaBufferSlice.addr, wr[0].memList.len, wr[0].sendFlags);
        u32 completeNum = 0;
        CHK_RET(HrtRaSendNormalWrlist(dataQpInfo_.qpHandle, wr, opRsp, WR_NUM, &completeNum));
        CHK_RET(DoorBellSend(dataQpInfo_.qpMode, wr[0], opRsp[0], stream));
        remainingBytes -= byteSizeChunk;
        processedOffset += byteSizeChunk;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::TransportIpc(
    const RmaBufferSlice& dstRmaBufferSlice, const RmaBufferSlice& srcRmaBufferSlice, const rtStream_t& stream)
{
    Stream hcclStream(stream);
    DispatcherPub* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcher_);
    CHK_RET(dispatcher->MemcpyAsync(
        dstRmaBufferSlice.addr, dstRmaBufferSlice.len, srcRmaBufferSlice.addr, srcRmaBufferSlice.len,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, hcclStream, remoteRankId_, hccl::LinkType::LINK_HCCS));
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::Write(const HcclBuf& remoteMem, const HcclBuf& localMem, const rtStream_t& stream)
{
    CHK_RET(CheckRaSendNormalWrlistSupport());
    CHK_PRT_RET(
        (localMem.addr == nullptr) || (remoteMem.addr == nullptr),
        HCCL_ERROR("[TransportRoceMem]localMem addr or remoteMem addr is invalid"), HCCL_E_PARA);
    CHK_PRT_RET(
        (localMem.len == 0U) || (remoteMem.len == 0U),
        HCCL_ERROR(
            "[TransportRoceMem]localMem size[%llu] or remoteMem size[%llu] is invalid", localMem.len, remoteMem.len),
        HCCL_E_PARA);
    CHK_PRT_RET(stream == nullptr, HCCL_ERROR("[TransportRoceMem]stream is invalid"), HCCL_E_PARA);

    RmaBufferSlice localRmaBufferSlice{};
    RmaBufferSlice remoteRmaBufferSlice{};
    CHK_RET(FillRmaBufferSlice(localMem, remoteMem, localRmaBufferSlice, remoteRmaBufferSlice));
    CHK_RET(TransportRdmaWithType(localRmaBufferSlice, remoteRmaBufferSlice, stream, RdmaOp::OP_WRITE));
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::Write(const RmaOpMem& remoteMem, const RmaOpMem& localMem, const rtStream_t& stream)
{
    CHK_RET(CheckRaSendNormalWrlistSupport());
    CHK_PRT_RET(
        (localMem.addr == nullptr) || (remoteMem.addr == nullptr),
        HCCL_ERROR("[TransportRoceMem]localMem addr or remoteMem addr is invalid"), HCCL_E_PARA);
    CHK_PRT_RET(
        (localMem.size == 0U) || (remoteMem.size == 0U),
        HCCL_ERROR(
            "[TransportRoceMem]localMem size[%llu] or remoteMem size[%llu] is invalid", localMem.size, remoteMem.size),
        HCCL_E_PARA);
    CHK_PRT_RET(stream == nullptr, HCCL_ERROR("[TransportRoceMem]stream is invalid"), HCCL_E_PARA);

    RmaBufferSlice localRmaBufferSlice{};
    RmaBufferSlice remoteRmaBufferSlice{};
    CHK_RET(FillRmaBufferSlice(localMem, remoteMem, localRmaBufferSlice, remoteRmaBufferSlice));
    CHK_RET(TransportRdmaWithType(localRmaBufferSlice, remoteRmaBufferSlice, stream, RdmaOp::OP_WRITE));
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::Read(const HcclBuf& localMem, const HcclBuf& remoteMem, const rtStream_t& stream)
{
    CHK_RET(CheckRaSendNormalWrlistSupport());
    CHK_PRT_RET(
        (localMem.addr == nullptr) || (remoteMem.addr == nullptr),
        HCCL_ERROR("[TransportRoceMem]localMem addr or remoteMem addr is invalid"), HCCL_E_PARA);
    CHK_PRT_RET(
        (localMem.len == 0U) || (remoteMem.len == 0U),
        HCCL_ERROR(
            "[TransportRoceMem]localMem size[%llu] or remoteMem size[%llu] is invalid", localMem.len, remoteMem.len),
        HCCL_E_PARA);
    CHK_PRT_RET(stream == nullptr, HCCL_ERROR("[TransportRoceMem]stream is invalid"), HCCL_E_PARA);

    RmaBufferSlice localRmaBufferSlice{};
    RmaBufferSlice remoteRmaBufferSlice{};
    CHK_RET(FillRmaBufferSlice(localMem, remoteMem, localRmaBufferSlice, remoteRmaBufferSlice));
    CHK_RET(TransportRdmaWithType(localRmaBufferSlice, remoteRmaBufferSlice, stream, RdmaOp::OP_READ));
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::Read(const RmaOpMem& localMem, const RmaOpMem& remoteMem, const rtStream_t& stream)
{
    CHK_RET(CheckRaSendNormalWrlistSupport());
    CHK_PRT_RET(
        (localMem.addr == nullptr) || (remoteMem.addr == nullptr),
        HCCL_ERROR("[TransportRoceMem]localMem addr or remoteMem addr is invalid"), HCCL_E_PARA);
    CHK_PRT_RET(
        (localMem.size == 0U) || (remoteMem.size == 0U),
        HCCL_ERROR(
            "[TransportRoceMem]localMem size[%llu] or remoteMem size[%llu] is invalid", localMem.size, remoteMem.size),
        HCCL_E_PARA);
    CHK_PRT_RET(stream == nullptr, HCCL_ERROR("[TransportRoceMem]stream is invalid"), HCCL_E_PARA);

    RmaBufferSlice localRmaBufferSlice{};
    RmaBufferSlice remoteRmaBufferSlice{};
    CHK_RET(FillRmaBufferSlice(localMem, remoteMem, localRmaBufferSlice, remoteRmaBufferSlice));
    CHK_RET(TransportRdmaWithType(localRmaBufferSlice, remoteRmaBufferSlice, stream, RdmaOp::OP_READ));
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::AddOpFence(const rtStream_t& stream)
{
    CHK_RET(CheckRaSendNormalWrlistSupport());
    auto opType = static_cast<u32>(MemType::SEND_NOTIFY_MEM);
    struct WrInfo wr[WR_NUM];
    wr[0].wrId = sendWrHandle.fetch_add(1, std::memory_order_relaxed);
    wr[0].memList.addr = reinterpret_cast<uint64_t>(rdmaSignal_[0].addr);
    wr[0].memList.len = notifyMemMsg_[opType].len;
    wr[0].memList.lkey = rdmaSignal_[0].lkey;
    wr[0].dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(notifyMemMsg_[opType].addr));
    wr[0].rkey = notifyMemMsg_[opType].rkey;
    wr[0].op = static_cast<u32>(RdmaOp::OP_READ);
    wr[0].sendFlags = RA_SEND_SIGNALED | RA_SEND_FENCE;
    u32 completeNum = 0;
    struct SendWrRsp opRsp[WR_NUM];
    CHK_RET(HrtRaSendNormalWrlist(dataQpInfo_.qpHandle, wr, opRsp, WR_NUM, &completeNum));
    CHK_RET(DoorBellSend(dataQpInfo_.qpMode, wr[0], opRsp[0], stream));
    CHK_RET(WaitOpFence(stream));
    HCCL_DEBUG(
        "[AddOpFence] wr.wrId[%llu], local addr[%p], remote addr[%p], len[%u], lkey[%u], rkey[%u], sendFlags[%d]",
        wr[0].wrId, rdmaSignal_[0].addr, notifyMemMsg_[opType].addr, wr[0].memList.len, wr[0].memList.lkey, wr[0].rkey,
        wr[0].sendFlags);
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::GetQpInfo(HcclQpInfoV2& qpInfo)
{
    qpInfo.qpPtr = aiQpInfo_.aiQpAddr; // reinterpret_cast<u64>(dataQpInfo_.qp)
    qpInfo.sqIndex = aiQpInfo_.sqIndex;
    qpInfo.dbIndex = aiQpInfo_.dbIndex;
    qpInfo.retryCnt = static_cast<u16>(GetExternalInputRdmaRetryCnt());
    qpInfo.retryTime = static_cast<u16>(GetExternalInputRdmaTimeOut());
    struct ibv_qp* qp = reinterpret_cast<struct ibv_qp*>(qpInfo.qpPtr);
    HCCL_DEBUG("[%s] qp=%p", __func__, qp);
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::GetMemInfo(u32& lkey, u32& rkey, HcclBuf& localMem, HcclBuf& remoteMem)
{
    CHK_PRT_RET(
        (localMem.addr == nullptr) || (remoteMem.addr == nullptr),
        HCCL_ERROR(
            "[TransportRoceMem] localMem addr[%p] or remoteMem addr[%p] is invalid", localMem.addr, remoteMem.addr),
        HCCL_E_PARA);
    CHK_PRT_RET(
        (localMem.len == 0U) || (remoteMem.len == 0U),
        HCCL_ERROR(
            "[TransportRoceMem] localMem size[%llu] or remoteMem size[%llu] is invalid", localMem.len, remoteMem.len),
        HCCL_E_PARA);

    RmaBufferSlice localRmaBufferSlice{};
    RmaBufferSlice remoteRmaBufferSlice{};
    CHK_RET(FillRmaBufferSlice(localMem, remoteMem, localRmaBufferSlice, remoteRmaBufferSlice));

    auto localRdmaRmaBuffer = dynamic_pointer_cast<LocalRdmaRmaBuffer>(localRmaBufferSlice.rmaBuffer);
    lkey = localRdmaRmaBuffer->GetKey();
    localMem.addr = localRmaBufferSlice.addr;
    localMem.len = localRmaBufferSlice.len;

    auto remoteRdmaRmaBuffer = dynamic_pointer_cast<RemoteRdmaRmaBuffer>(remoteRmaBufferSlice.rmaBuffer);
    rkey = remoteRdmaRmaBuffer->GetKey();
    remoteMem.addr = remoteRmaBufferSlice.addr;
    remoteMem.len = remoteRmaBufferSlice.len;

    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::GetOpFence(u32& lkey, u32& rkey, HcclBuf& localMem, HcclBuf& remoteMem)
{
    auto opType = static_cast<u32>(MemType::SEND_NOTIFY_MEM);
    lkey = rdmaSignal_[0].lkey;
    localMem.addr = rdmaSignal_[0].addr;
    localMem.len = rdmaSignal_[0].len;
    rkey = notifyMemMsg_[opType].rkey;
    remoteMem.addr = notifyMemMsg_[opType].addr;
    HCCL_DEBUG(
        "[GetOpFence] local addr[%p], remote addr[%p], len[%u], lkey[%u], rkey[%u]", localMem.addr, remoteMem.addr,
        localMem.len, lkey, rkey);
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::GetTransInfo(
    HcclQpInfoV2& qpInfo, u32* lkey, u32* rkey, HcclBuf* localMem, HcclBuf* remoteMem, u32 num)
{
    CHK_PTR_NULL(lkey);
    CHK_PTR_NULL(rkey);
    CHK_PTR_NULL(localMem);
    CHK_PTR_NULL(remoteMem);
    CHK_PRT_RET(
        num == 0, HCCL_ERROR("[GetTransInfo] mem num should not be zero, at least one for OpFence"), HCCL_E_PARA);
    CHK_RET(GetQpInfo(qpInfo));
    for (u32 i = 0; i < num - 1; ++i) { // last element is signal
        HcclResult ret = GetMemInfo(lkey[i], rkey[i], localMem[i], remoteMem[i]);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[GetTransInfo] failed at index[%u], localAddr[%p/%llu], "
                "remoteAddr[%p/%llu]",
                i, localMem[i].addr, localMem[i].len, remoteMem[i].addr, remoteMem[i].len),
            ret);
    }
    CHK_RET(GetOpFence(lkey[num - 1], rkey[num - 1], localMem[num - 1], remoteMem[num - 1]));
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::WaitOpFence(const rtStream_t& stream)
{
    auto opType = static_cast<u32>(MemType::SEND_NOTIFY_MEM);
    hccl::Stream hcclStream(stream);
    DispatcherPub* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcher_);
    const u32 timeOut = (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET)
                                || dispatcher->GetExecTimeOutSet() ?
                            dispatcher->GetExecTimeOut() :
                            NOTIFY_DEFAULT_WAIT_TIME;
    HcclResult ret = LocalIpcNotify::Wait(
        hcclStream, dispatcher, remoteIsendDoneSignal_, INVALID_VALUE_STAGE, timeOut, localRankId_, remoteRankId_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[WaitOpFence] timeout[%u s], local addr[%p], remote addr[%p], remoteRankId[%u], streamId[%u]", timeOut,
            rdmaSignal_[0].addr, notifyMemMsg_[opType].addr, remoteRankId_, hcclStream.id()),
        ret);
    HCCL_DEBUG(
        "[WaitOpFence] local addr[%p], remote addr[%p], remoteRankId[%u], streamId[%u]", rdmaSignal_[0].addr,
        notifyMemMsg_[opType].addr, remoteRankId_, hcclStream.id());
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::BatchWrite(
    [[maybe_unused]] const std::vector<MemDetails>& remoteMems,
    [[maybe_unused]] const std::vector<MemDetails>& localMems, [[maybe_unused]] Stream& stream)
{
    HCCL_ERROR("TransportRoceMem doesn't support BatchWrite");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportRoceMem::BatchRead(
    [[maybe_unused]] const std::vector<MemDetails>& localMems,
    [[maybe_unused]] const std::vector<MemDetails>& remoteMems, [[maybe_unused]] Stream& stream)
{
    HCCL_ERROR("TransportRoceMem doesn't support BatchRead");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportRoceMem::AddOpFence(
    [[maybe_unused]] const MemDetails& localFenceMem, [[maybe_unused]] const MemDetails& remoteFenceMem,
    [[maybe_unused]] Stream& stream)
{
    HCCL_ERROR("TransportRoceMem doesn't support AICPU AddOpFence");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportRoceMem::CreateCqAndQp()
{
    dataQpInfo_.flag = QP_FLAG_RC;
    dataQpInfo_.qpMode = OPBASE_QP_MODE_EXT;
    dataQpInfo_.trafficClass = trafficClass_;
    dataQpInfo_.serviceLevel = serviceLevel_;
    if (aicpuUnfoldMode_) {
        CHK_RET(CreateAiQp(nicRdmaHandle_, aiQpInfo_, dataQpInfo_, devicePhyId_));
    } else {
        CHK_RET(CreateQpWithCq(nicRdmaHandle_, -1, -1, nullptr, nullptr, dataQpInfo_, true, true));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::QpConnect(s32 timeoutSec)
{
    CHK_RET(HrtRaQpConnectAsync(
        dataQpInfo_.qpHandle, socket_->GetFdHandle(),
        [this]() -> bool {
            return this->socket_->GetStopFlag();
        },
        timeoutSec));

    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::RecoverNotifyMsg(MemMsg* remoteRdmaSignal, u64 signalNum)
{
    if (signalNum <= 0) {
        return HCCL_E_NOT_FOUND;
    }
    MemType tmpMemType = MemType::MEM_TYPE_RESERVED;
    for (u64 i = 0; i < signalNum; i++) {
        HCCL_DEBUG(
            "recv mrRegFlag:[%d] notifyAddr:[%p] len:[%lu] memType:[%d], rkey:[%u]  ",
            (remoteRdmaSignal + i)->mrRegFlag, (remoteRdmaSignal + i)->addr, (remoteRdmaSignal + i)->len,
            static_cast<int>((remoteRdmaSignal + i)->memType), (remoteRdmaSignal + i)->lkey);
        tmpMemType = (remoteRdmaSignal + i)->memType;
        if ((remoteRdmaSignal + i)->memType == MemType::NOTIFY_SRC_MEM) {
            tmpMemType = MemType::SEND_NOTIFY_MEM;
            notifyMemMsg_[tmpMemType].mrRegFlag = (remoteRdmaSignal + i)->mrRegFlag;
            notifyMemMsg_[tmpMemType].addr = (remoteRdmaSignal + i)->addr;
            notifyMemMsg_[tmpMemType].len = (remoteRdmaSignal + i)->len;
            notifyMemMsg_[tmpMemType].memType = MemType::SEND_NOTIFY_MEM;
            notifyMemMsg_[tmpMemType].rkey = (remoteRdmaSignal + i)->lkey;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::CreatSignalMesg()
{
    CHK_RET(GetNotifySize());
    CHK_RET(CreateRdmaSignal(remoteIsendDoneSignal_, rdmaSignal_[0], MemType::RECV_NOTIFY_MEM));
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::GetNotifySize()
{
    DevType devType;
    CHK_RET(hrtHalGetDeviceType(deviceLogicId_, devType));
    if ((devType == DevType::DEV_TYPE_910B) || (devType == DevType::DEV_TYPE_910_93)) {
        notifySize_ = NOTIFY_SIZE_BYTES_910B;
    } else {
        notifySize_ = NOTIFY_SIZE_BYTES_910A;
    }
    HCCL_INFO("devType[%d] notifySize[%d]", devType, notifySize_);
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::ExchangeNotifyValueBuffer(s32 timeoutSec)
{
    CHK_RET(socket_->Send(
        &notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)], sizeof(MemMsg) * REMOTE_RDMA_SIGNAL_SIZE));
    HCCL_DEBUG(
        "send mrRegFlag:[%d] notifyAddr:[%p] len:[%lu] memType:[%d], rkey:[%u]  ",
        notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].mrRegFlag,
        notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr,
        notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].len,
        notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].memType,
        notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey);
    MemMsg remoteNotifyValue[REMOTE_RDMA_SIGNAL_SIZE];
    CHK_RET(socket_->Recv(remoteNotifyValue, sizeof(MemMsg) * REMOTE_RDMA_SIGNAL_SIZE, timeoutSec));
    CHK_RET(RecoverNotifyMsg(remoteNotifyValue, REMOTE_RDMA_SIGNAL_SIZE));
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::CreateRdmaSignal(
    std::shared_ptr<LocalIpcNotify>& localNotify, MemMsg& rdmaSignalInfo, MemType notifyType)
{
    u64 notifyOffset = 0;
    u64 notifyBaseVa = 0; // notify寄存器虚拟地址
    u64 notifyTotalSize = 0;

    RemoteRankInfo info(devicePhyId_, remoteRankId_);
    CHK_RET(SalGetBareTgid(&info.remotePid)); // 当前进程id
    CHK_RET(notifyPool_->Alloc(socket_->GetTag(), info, localNotify));
    // 设置remote id
    s64 recvId = 0xFFFFFFFF00000000 | (static_cast<s64>(info.remotePid) & 0xFFFFFFFF);
    CHK_RET(localNotify->Grant(recvId));

    CHK_RET(HrtRaGetNotifyBaseAddr(nicRdmaHandle_, &notifyBaseVa, &notifyTotalSize));
    CHK_RET(localNotify->GetNotifyOffset(notifyOffset));
    u64 notifyVa = notifyBaseVa + notifyOffset;
    rdmaSignalInfo.mrRegFlag = 0;
    rdmaSignalInfo.addr = reinterpret_cast<void*>(static_cast<uintptr_t>(notifyVa));
    rdmaSignalInfo.len = notifySize_;
    rdmaSignalInfo.memType = notifyType;

    HCCL_INFO(
        "notifyBaseVa=0x%llx, notifyTotalSize=0x%x, notifyOffset=0x%llx, notifyVa=0x%llx", notifyBaseVa,
        notifyTotalSize, notifyOffset, notifyVa);

    struct MrInfoT mrInfo = {};
    CHK_RET(HrtRaGetNotifyMrInfo(devicePhyId_, nicRdmaHandle_, &mrInfo));
    rdmaSignalInfo.lkey = mrInfo.lkey;

    HcclSignalInfo notifyInfo{};
    notifyInfo.resId = INVALID_U64;
    CHK_RET(localNotify->GetNotifyData(notifyInfo));
    HCCL_INFO("CreateRdmaSignal localNotify id[%llu]", notifyInfo.resId);
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::CreateNotifyValueBuffer()
{
    if (notifyMem_.ptr() == nullptr) {
        u64 notifyVaule = 1; // notify值写1表示record
        CHK_RET(DeviceMem::alloc(notifyMem_, notifyValueSize_));

        CHK_RET(hrtMemSyncCopy(
            notifyMem_.ptr(), notifyMem_.size(), &notifyVaule, notifySize_,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    }

    struct MrInfoT mrInfo = {};
    mrInfo.addr = notifyMem_.ptr();
    mrInfo.size = notifySize_;
    mrInfo.access = access_;
    CHK_RET(hrtRaRegGlobalMr(nicRdmaHandle_, mrInfo, notifyValueMemMrHandle_));
    notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].mrRegFlag = REG_VALID;
    notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].addr = notifyMem_.ptr();
    notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].len = notifySize_;
    notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].memType = MemType::NOTIFY_SRC_MEM;
    notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_SRC_MEM)].lkey = mrInfo.lkey;
    HCCL_DEBUG("notifyValueMem_=%p", notifyMem_.ptr());
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::DoorBellSend(
    [[maybe_unused]] const s32 qpMode, WrInfo& sendWrInfo, const SendWrRsp& opRsp, rtStream_t stream)
{
    struct SendWr sendwr = {};
    sendwr.bufList = &sendWrInfo.memList;
    sendwr.bufNum = 1; /* 此处list只有一个，设置为1 */
    sendwr.dstAddr = sendWrInfo.dstAddr;
    sendwr.rkey = sendWrInfo.rkey;
    sendwr.op = sendWrInfo.op;
    sendwr.sendFlag = sendWrInfo.sendFlags;
    u32 dbIndex = static_cast<u32>(opRsp.db.dbIndex);
    u64 dbInfo = static_cast<u64>(opRsp.db.dbInfo);
    CHK_RET(RdmaDbSend(dbIndex, dbInfo, sendwr, stream));
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::RdmaDbSend(u32 dbindex, u64 dbinfo, const struct SendWr& sendWr, rtStream_t stream)
{
    hccl::Stream hcclStream(stream);
    DispatcherPub* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcher_);
    s32 ret = dispatcher->RdmaSend(dbindex, dbinfo, sendWr, hcclStream, remoteRankId_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[RdmaDbSend]errNo[0x%016llx] rdma db send fail, "
            "return[%d]. para: dbindex[%u] dbinfo[%llu].",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), ret, dbindex, dbinfo),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::WaitQPLinkComplete(s32 timeoutSec)
{
    auto startTime = chrono::steady_clock::now();
    maxTimeOut_ = std::chrono::seconds(timeoutSec);
    while ((chrono::steady_clock::now() - startTime) < maxTimeOut_) {
        HcclResult ret = GetQpStatus();
        if (ret == HCCL_E_AGAIN) {
            SaluSleep(WAIT_LINK_BUILD_DELAY_TIME_US);
            continue;
        }
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("TransportRoceMem QP connect success");
        } else {
            HCCL_ERROR("TransportRoceMem QP connect failed, ret[%d]!", ret);
        }
        return ret;
    }
    HCCL_RUN_INFO(
        "WaitBuildLinkComplete timeOut[%d] s, localRank[%u], remoteRank[%u]", timeoutSec, localRankId_, remoteRankId_);
    return HCCL_E_TIMEOUT;
}

HcclResult TransportRoceMem::GetQpStatus()
{
    int qpStatus = 0;
    s32 ret = 0;

    ret = hrtGetRaQpStatus(dataQpInfo_.qpHandle, &qpStatus);
    if (ret != 0) {
        return HCCL_E_INTERNAL;
    } else if (ret == 0 && qpStatus != 1) { // 为1时，qp 建链成功
        return HCCL_E_AGAIN;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportRoceMem::DestroyCqAndQp()
{
    HCCL_INFO("TransportRoceMem DestroyCqAndQp");
    if (aicpuUnfoldMode_) {
        CHK_RET(DestroyAiQp(dataQpInfo_));
    } else {
        CHK_RET(DestroyQpWithCq(dataQpInfo_, true));
    }
    return HCCL_SUCCESS;
}

} // namespace hccl

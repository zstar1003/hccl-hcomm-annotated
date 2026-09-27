/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport.h"
#include "transport_base.h"
#include "transport_ibverbs.h"
#include "transport_direct_npu.h"
#ifdef CCL_KERNEL
#include "transport_device_p2p.h"
#include "transport_device_ibverbs.h"
#endif
#include "transport_p2p.h"
#include "transport_virtural.h"
namespace hccl {
std::mutex Transport::mapMutex_;
std::unordered_map<TransportBase*, Transport*> Transport::transportMap_;
Transport::Transport(
    TransportType type, TransportPara& para, const HcclDispatcher dispatcher,
    const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
    [[maybe_unused]] const TransportDeviceP2pData& transDevP2pData,
    [[maybe_unused]] const TransportDeviceIbverbsData& transDevIbverbsData)
    : type_(type)
{
    DispatcherPub* dispatcherPub = reinterpret_cast<DispatcherPub*>(const_cast<HcclDispatcher>(dispatcher));
    if (type == TransportType::TRANS_TYPE_IBV_EXP) {
        pimpl_ = new (std::nothrow) TransportIbverbs(dispatcherPub, notifyPool, machinePara, para.timeout);
        if (pimpl_ != nullptr) {
            std::lock_guard<std::mutex> maplock(mapMutex_);
            transportMap_.insert({pimpl_, this});
        }
    } else if (type == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
        pimpl_ = new (std::nothrow) TransportDirectNpu(dispatcherPub, notifyPool, machinePara, para.timeout);
        if (pimpl_ != nullptr) {
            std::lock_guard<std::mutex> maplock(mapMutex_);
            transportMap_.insert({pimpl_, this});
        }
    } else if (type == TransportType::TRANS_TYPE_P2P) {
        pimpl_ = new (std::nothrow) TransportP2p(dispatcherPub, notifyPool, machinePara, para.timeout);
    } else if (type == TransportType::TRANS_TYPE_DEVICE_P2P) {
#ifdef CCL_KERNEL
        pimpl_ = new (std::nothrow)
            TransportDeviceP2p(dispatcherPub, notifyPool, machinePara, para.timeout, transDevP2pData);
        // 创建设备间P2P传输
#else
        HCCL_ERROR("TRANS_TYPE_DEVICE_P2P Only running on the AICPU");
#endif
    } else if (type == TransportType::TRANS_TYPE_DEVICE_IBVERBS) {
#ifdef CCL_KERNEL
        pimpl_ = new (std::nothrow)
            TransportDeviceIbverbs(dispatcherPub, notifyPool, machinePara, para.timeout, transDevIbverbsData);
#else
        HCCL_ERROR("TRANS_TYPE_DEVICE_IBVERBS Only running on the AICPU");
#endif
    } else if (para.virtualFlag) {
        pimpl_ = new (std::nothrow) TransportVirtural(dispatcherPub, notifyPool, machinePara, para.timeout, para.index);
    } else {
        pimpl_ = new (std::nothrow) TransportBase(dispatcherPub, notifyPool, machinePara, para.timeout);
    }
    CHK_PRT_CONT(
        pimpl_ == nullptr,
        HCCL_ERROR("[Transport][Transport] create pimpl_ failed, type[%d].", static_cast<int>(type)));
    HCCL_DEBUG("Transport::Transport, type = %d", static_cast<int>(type));
}

Transport::~Transport()
{
    std::unique_lock<std::mutex> maplock(mapMutex_);
    if (transportMap_.find(pimpl_) != transportMap_.end()) {
        transportMap_.erase(pimpl_);
    }
    maplock.unlock();

    delete pimpl_;
    pimpl_ = nullptr;
}

HcclResult Transport::Init()
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->Init();
}

HcclResult Transport::DeInit()
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->DeInit();
}

HcclResult Transport::TxDataSignal(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->TxDataSignal(stream);
}

HcclResult Transport::RxDataSignal(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->RxDataSignal(stream);
}

HcclResult Transport::TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    // src在transport内部校验
    return pimpl_->TxAsync(dstMemType, dstOffset, src, len, stream);
}

HcclResult Transport::TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->TxAsync(txMems, stream);
}

HcclResult Transport::TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    // src在transport内部校验
    return pimpl_->TxData(dstMemType, dstOffset, src, len, stream);
}

HcclResult Transport::RxData(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    // dst在transport内部校验
    return pimpl_->RxData(srcMemType, srcOffset, dst, len, stream);
}

HcclResult Transport::TxPrepare(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->TxPrepare(stream);
}

HcclResult Transport::RxPrepare(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->RxPrepare(stream);
}

HcclResult Transport::TxDone(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->TxDone(stream);
}

HcclResult Transport::RxDone(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->RxDone(stream);
}

HcclResult Transport::Stop()
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->Stop();
}

HcclResult Transport::Resume()
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->Resume();
}

HcclResult Transport::TxWithReduce(
    UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, const HcclDataType datatype, HcclReduceOp redOp,
    Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    // src在transport内部校验
    return pimpl_->TxWithReduce(dstMemType, dstOffset, src, len, datatype, redOp, stream);
}

HcclResult Transport::TxWithReduce(
    const std::vector<TxMemoryInfo>& txWithReduceMems, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->TxWithReduce(txWithReduceMems, datatype, redOp, stream);
}

HcclResult Transport::RxWithReduce(
    UserMemType recvSrcMemType, u64 recvSrcOffset, void* recvDst, u64 recvLen, void* reduceSrc, void* reduceDst,
    u64 reduceDataCount, HcclDataType reduceDatatype, HcclReduceOp reduceOp, Stream& stream, const u64 reduceAttr)
{
    CHK_PTR_NULL(pimpl_);
    CHK_PTR_NULL(recvDst);
    CHK_PTR_NULL(reduceSrc);
    CHK_PTR_NULL(reduceDst);
    return pimpl_->RxWithReduce(
        recvSrcMemType, recvSrcOffset, recvDst, recvLen, reduceSrc, reduceDst, reduceDataCount, reduceDatatype,
        reduceOp, stream, reduceAttr);
}

HcclResult Transport::RxWithReduce(
    const std::vector<RxWithReduceMemoryInfo>& rxWithReduceMems, HcclDataType reduceDatatype, HcclReduceOp reduceOp,
    Stream& stream, const u64 reduceAttr)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->RxWithReduce(rxWithReduceMems, reduceDatatype, reduceOp, stream, reduceAttr);
}

bool Transport::IsSupportTransportWithReduce()
{
    if (pimpl_ == nullptr) {
        return false;
    }
    return pimpl_->IsSupportTransportWithReduce();
}

HcclResult Transport::RxAsync(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    // dst在transport内部校验
    return pimpl_->RxAsync(srcMemType, srcOffset, dst, len, stream);
}

HcclResult Transport::RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->RxAsync(rxMems, stream);
}

HcclResult Transport::DataReceivedAck(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->DataReceivedAck(stream);
}

HcclResult Transport::TxAck(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->TxAck(stream);
}

HcclResult Transport::RxAck(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->RxAck(stream);
}

HcclResult Transport::TxWaitDone(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->TxWaitDone(stream);
}

HcclResult Transport::RxWaitDone(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->RxWaitDone(stream);
}

HcclResult Transport::Post(u32 notifyIdx, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->Post(notifyIdx, stream);
}

HcclResult Transport::Wait(u32 notifyIdx, Stream& stream, const u32 timeOut)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->Wait(notifyIdx, stream, timeOut);
}

u32 Transport::GetNotifyNum()
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetNotifyNum();
}

HcclResult Transport::GetLocalNotify(std::vector<HcclSignalInfo>& localNotify)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetLocalNotify(localNotify);
}

HcclResult Transport::GetRemoteNotify(std::vector<HcclSignalInfo>& localNotify)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetRemoteNotify(localNotify);
}

HcclResult Transport::GetIndOpRemoteMemDetails(MemDetails** remoteMem, uint32_t* memNum, HcclMemType memType)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetIndOpRemoteMemDetails(remoteMem, memNum, memType);
}

HcclResult Transport::GetIndOpRemoteMem(HcclMem** remoteMem, uint32_t* memNum)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetIndOpRemoteMem(remoteMem, memNum);
}

HcclResult Transport::GetRemoteMem(UserMemType memType, void** remotePtr)
{
    CHK_PTR_NULL(pimpl_);
    CHK_PTR_NULL(remotePtr);
    return pimpl_->GetRemoteMem(memType, remotePtr);
}

HcclResult Transport::GetRemoteMem(std::vector<void*>* remotePtrVec)
{
    CHK_PTR_NULL(pimpl_);
    CHK_PTR_NULL(remotePtrVec);
    return pimpl_->GetRemoteMem(remotePtrVec);
}

HcclResult Transport::GetRemoteMemKey(UserMemType memType, uint32_t* remoteMemKey)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetRemoteMemKey(memType, remoteMemKey);
}

HcclResult Transport::GetLocalRdmaNotify(std::vector<HcclSignalInfo>& rdmaNotify)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetLocalRdmaNotify(rdmaNotify);
}

HcclResult Transport::GetDrainLocalDataNotify(void*& localAddr, uint32_t& lkey, HcclSignalInfo& dataNotify)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetDrainLocalDataNotify(localAddr, lkey, dataNotify);
}

HcclResult Transport::GetRemoteRdmaNotifyAddrKey(std::vector<AddrKey>& rdmaNotifyAddr)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetRemoteRdmaNotifyAddrKey(rdmaNotifyAddr);
}

HcclResult Transport::GetLocalNotifyValueAddrKey(std::vector<AddrKey>& notifyValue)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetLocalNotifyValueAddrKey(notifyValue);
}

HcclResult Transport::GetLocalMemDetails(UserMemType memType, MemDetails& memDetails)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetLocalMemDetails(memType, memDetails);
}

HcclResult Transport::GetChipId(s64& chipId)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetChipId(chipId);
}

HcclResult Transport::GetAiQpInfo(std::vector<HcclQpInfoV2>& aiQpInfo)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetAiQpInfo(aiQpInfo);
}
HcclResult Transport::GetTransportId(u32& id)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetTransportId(id);
}

HcclResult Transport::GetAiRMAQueueInfo(std::vector<HcclAiRMAQueueInfo>& aiRMAQueueInfo)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetAiRMAQueueInfo(aiRMAQueueInfo);
}

HcclResult Transport::GetRemoteMemSize(UserMemType memType, u64& size)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetRemoteMemSize(memType, size);
}

HcclResult Transport::GetTxAckDevNotifyInfo(HcclSignalInfo& notifyInfo)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetTxAckDevNotifyInfo(notifyInfo);
}

HcclResult Transport::GetRxAckDevNotifyInfo(HcclSignalInfo& notifyInfo)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetRxAckDevNotifyInfo(notifyInfo);
}

HcclResult Transport::GetTxDataSigleDevNotifyInfo(HcclSignalInfo& notifyInfo)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetTxDataSigleDevNotifyInfo(notifyInfo);
}

HcclResult Transport::GetRxDataSigleDevNotifyInfo(HcclSignalInfo& notifyInfo)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetRxDataSigleDevNotifyInfo(notifyInfo);
}

hccl::LinkType Transport::GetLinkType() const
{
    if (pimpl_ == nullptr) {
        return hccl::LinkType::LINK_RESERVED;
    }
    return pimpl_->GetLinkType();
}

bool Transport::GetSupportDataReceivedAck() const
{
    if (pimpl_ == nullptr) {
        return false;
    }
    return pimpl_->GetSupportDataReceivedAck();
}

void Transport::SetSupportDataReceivedAck(bool supportDataReceivedAck)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->SetSupportDataReceivedAck(supportDataReceivedAck);
}

bool Transport::IsSpInlineReduce() const
{
    if (pimpl_ == nullptr) {
        return false;
    }
    return pimpl_->IsSpInlineReduce();
}

u32 Transport::GetRemoteRank()
{
    if (pimpl_ == nullptr) {
        return INVALID_VALUE_RANKID;
    }
    return pimpl_->GetRemoteRank();
}

HcclResult Transport::ConnectAsync(u32& status)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->ConnectAsync(status);
}

HcclResult Transport::ConnectQuerry(u32& status)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->ConnectQuerry(status);
}

void Transport::Break()
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->Break();
}

void Transport::EnableUseOneDoorbell()
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->EnableUseOneDoorbell();
}

bool Transport::GetUseOneDoorbellValue()
{
    if (pimpl_ == nullptr) {
        return false;
    }
    return pimpl_->GetUseOneDoorbellValue();
}

HcclResult Transport::GetTransportAttr(TransportAttr& attr)
{
    CHK_PTR_NULL(pimpl_);
    attr = pimpl_->GetTransportAttr();
    return HCCL_SUCCESS;
}

HcclResult Transport::TxEnv(const void* ptr, const u64 len, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    CHK_PTR_NULL(ptr);
    return pimpl_->TxEnv(ptr, len, stream);
}

HcclResult Transport::RxEnv(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->RxEnv(stream);
}

bool Transport::IsTransportRoce() { return false; }

HcclResult Transport::WriteAsync(struct Buffer& remoteBuf, struct Buffer& localBuf, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    CHK_PTR_NULL(remoteBuf.addr);
    CHK_PTR_NULL(localBuf.addr);
    // localAddr在transport内部校验
    return pimpl_->WriteAsync(remoteBuf, localBuf, stream);
}

HcclResult Transport::WriteSync(struct Buffer& remoteBuf, struct Buffer& localBuf, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    CHK_PTR_NULL(remoteBuf.addr);
    CHK_PTR_NULL(localBuf.addr);
    // localAddr在transport内部校验
    return pimpl_->WriteSync(remoteBuf, localBuf, stream);
}

HcclResult Transport::WriteReduceAsync(
    struct Buffer& remoteBuf, struct Buffer& localBuf, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    CHK_PTR_NULL(remoteBuf.addr);
    CHK_PTR_NULL(localBuf.addr);
    // localAddr在transport内部校验
    return pimpl_->WriteReduceAsync(remoteBuf, localBuf, datatype, redOp, stream);
}

HcclResult Transport::ReadAsync(struct Buffer& localBuf, struct Buffer& remoteBuf, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    CHK_PTR_NULL(remoteBuf.addr);
    CHK_PTR_NULL(localBuf.addr);
    // localAddr在transport内部校验
    return pimpl_->ReadAsync(localBuf, remoteBuf, stream);
}

HcclResult Transport::ReadSync(struct Buffer& localBuf, struct Buffer& remoteBuf, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    CHK_PTR_NULL(remoteBuf.addr);
    CHK_PTR_NULL(localBuf.addr);
    // localAddr在transport内部校验
    return pimpl_->ReadSync(localBuf, remoteBuf, stream);
}

HcclResult Transport::ReadReduceSync(
    struct Buffer& localBuf, struct Buffer& remoteBuf, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    CHK_PTR_NULL(localBuf.addr);
    CHK_PTR_NULL(remoteBuf.addr);
    return pimpl_->ReadReduceSync(localBuf, remoteBuf, datatype, redOp, stream);
}

HcclResult Transport::BatchTransferAsync(const HcommBatchTransferDesc* transferDescs, uint32_t descNum, Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->BatchTransferAsync(transferDescs, descNum, stream);
}

HcclResult Transport::PostReady(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->PostReady(stream);
}

HcclResult Transport::WaitReady(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->WaitReady(stream);
}

HcclResult Transport::PostFin(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->PostFin(stream);
}

HcclResult Transport::WaitFin(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->WaitFin(stream);
}

HcclResult Transport::PostFinAck(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->PostFinAck(stream);
}

HcclResult Transport::WaitFinAck(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->WaitFinAck(stream);
}

HcclResult Transport::SetStopFlag(bool value)
{
    if (pimpl_ != nullptr) {
        return pimpl_->SetStopFlag(value);
    }
    return HCCL_SUCCESS;
}

HcclResult Transport::UpdateRemoteAddr(void* remoteIn, void* remoteOut)
{
    CHK_PTR_NULL(pimpl_);
    CHK_PTR_NULL(remoteIn);
    CHK_PTR_NULL(remoteOut);
    return pimpl_->UpdateRemoteAddr(remoteIn, remoteOut);
}

std::vector<u8> Transport::GetExchangeInfo()
{
    if (UNLIKELY(pimpl_ == nullptr)) {
        return std::vector<u8>();
    }
    return pimpl_->GetExchangeInfo();
}

HcclResult Transport::GetTransportErrorCqe(
    const HcclNetDevCtx netDevCtx, std::vector<std::pair<Transport*, CqeInfo>>& infos, u32& num)
{
    CHK_PTR_NULL(netDevCtx);
    HcclIpAddress localIp;
    CHK_RET(HcclNetDevGetLocalIp(netDevCtx, localIp));

    std::vector<std::pair<TransportBase*, CqeInfo>> infolist;
    CHK_RET(TransportIbverbs::GetTransportErrorCqe(netDevCtx, infolist, num));

    std::lock_guard<std::mutex> maplock(mapMutex_);
    for (auto info : infolist) {
        auto iter = transportMap_.find(info.first);
        if (iter != transportMap_.end()) {
            infos.push_back(std::make_pair(iter->second, info.second));
        } else {
            HCCL_RUN_WARNING(
                "[GetTransportErrorCqe]get err failed, transport is not found, localIp[%s], remoteIp[%s]",
                localIp.GetReadableAddress(), info.second.remoteIp.GetReadableAddress());
        }
    }
    num = infos.size();

    return HCCL_SUCCESS;
}

HcclResult Transport::Fence()
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->Fence();
}

bool Transport::GetIsUseAtomicWrite()
{
    if (pimpl_ == nullptr) {
        return false;
    }
    return pimpl_->GetIsUseAtomicWrite();
}

HcclResult Transport::GetSpecificNotify(HcclSignalInfo& notifyInfo, bool& isValid, const std::string& notifyName)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetSpecificNotify(notifyInfo, isValid, notifyName);
}

HcclResult Transport::HcclBatchRead(
    [[maybe_unused]] const TransportDeviceNormalData& ibvData, [[maybe_unused]] struct MemDetails* localMems,
    [[maybe_unused]] struct MemDetails* remoteMems, [[maybe_unused]] u32 memNum, [[maybe_unused]] u64& dbInfo)
{
#ifdef CCL_KERNEL
    return TransportDeviceIbverbs::HnsPostSend(
        ibvData, localMems, remoteMems, memNum, HcclWrOpCode::HCCL_WR_RDMA_READ, dbInfo);
#else
    HCCL_ERROR("[Transport][HcclBatchRead]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult Transport::SetDeviceUnavailable(u32 deviceId)
{
    return MemNameRepository::GetInstance(deviceId)->SetDeviceUnavailable(true);
}

HcclResult Transport::HcclBatchWrite(
    [[maybe_unused]] const TransportDeviceNormalData& ibvData, [[maybe_unused]] struct MemDetails* localMems,
    [[maybe_unused]] struct MemDetails* remoteMems, [[maybe_unused]] u32 memNum, [[maybe_unused]] u64& dbInfo)
{
#ifdef CCL_KERNEL
    return TransportDeviceIbverbs::HnsPostSend(
        ibvData, localMems, remoteMems, memNum, HcclWrOpCode::HCCL_WR_RDMA_WRITE, dbInfo);
#else
    HCCL_ERROR("[Transport][HcclBatchWrite]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult Transport::Drain(Stream& stream)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->Drain(stream);
}

HcclResult Transport::InitDrainNotifyInfo()
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->InitDrainNotifyInfo();
}

HcclResult Transport::GetDrainRemSrcMem(void*& remoteAddr, uint32_t& remoteKey, uint32_t& size)
{
    CHK_PTR_NULL(pimpl_);
    return pimpl_->GetDrainRemSrcMem(remoteAddr, remoteKey, size);
}
} // namespace hccl

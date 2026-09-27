/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_P2P_PUB_H
#define TRANSPORT_P2P_PUB_H

#include <sstream>

#include "transport_base_pub.h"
#include "rma_buffer_mgr.h"

namespace hccl {
using HcclMemExMgr = hcomm::RmaBufferMgr<hccl::BufferKey<uintptr_t, u64>, std::shared_ptr<HcclMemEx>>;

typedef enum {
    EX_IPCMEN_SIZE = 0, /**< ipcMenSize */
    EX_NOTIFY_SIZE = 1, /**< notifySize */
    EX_EXDATA_SIZE = 2  /**< exDataSize */
} ExInfoType;
struct ExchangeInfoSize {
    u32 ipcMenSize;
    u32 notifySize;
    u32 exDataSize;
    u32 indOpMemSize; // 独立算子内存两端大小不同，不参与比较

    bool compare(ExchangeInfoSize& that)
    {
        if ((this->ipcMenSize != that.ipcMenSize) || (this->notifySize != that.notifySize)
            || (this->exDataSize != that.exDataSize)) {
            return false;
        }
        return true;
    }
};

class TransportP2p : public TransportBase {
public:
    explicit TransportP2p(
        DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
        std::chrono::milliseconds timeout);
    ~TransportP2p() override;

    HcclResult Init() override;

    HcclResult TxDataSignal(Stream& stream) override;
    HcclResult RxDataSignal(Stream& stream) override;

    HcclResult TxAck(Stream& stream) override;
    HcclResult RxAck(Stream& stream) override;

    HcclResult TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream) override;
    HcclResult TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream) override;

    HcclResult RxAsync(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream) override;
    HcclResult RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream) override;

    HcclResult DataReceivedAck(Stream& stream) override;

    HcclResult TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream) override;

    HcclResult RxData(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream) override;

    HcclResult TxPrepare(Stream& stream) override;
    HcclResult RxPrepare(Stream& stream) override;

    HcclResult TxDone(Stream& stream) override;
    HcclResult RxDone(Stream& stream) override;

    HcclResult GetIndOpRemoteMem(HcclMem** remoteMem, uint32_t* memNum) override;
    HcclResult GetRemoteMem(UserMemType memType, void** remotePtr) override;
    HcclResult GetRemoteMem(std::vector<void*>* remotePtrVec) override;

    HcclResult GetRemoteMemSize(UserMemType memType, u64& size) override;

    HcclResult WriteAsync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream);
    HcclResult WriteAsyncEx(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream);
    HcclResult WriteSync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream);
    HcclResult WriteReduceAsync(
        struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, const HcclDataType datatype,
        HcclReduceOp redOp, Stream& stream) override;

    HcclResult ReadAsync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream);
    HcclResult ReadAsyncEx(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream);
    HcclResult ReadSync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream);
    HcclResult ReadReduceSync(
        struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, const HcclDataType datatype,
        HcclReduceOp redOp, Stream& stream) override;

    HcclResult PostReady(Stream& stream);
    HcclResult WaitReady(Stream& stream);

    HcclResult PostFin(Stream& stream);
    HcclResult WaitFin(Stream& stream);

    HcclResult Post(u32 notifyIdx, Stream& stream) override;
    HcclResult Wait(u32 notifyIdx, Stream& stream, const u32 timeOut = NOTIFY_INVALID_WAIT_TIME) override;

protected:
    HcclResult FillExchangeDataTotalSize() override;

    HcclResult ConstructExchangeForSend() override;
    HcclResult ConstructIpcMemInfoForSend(void* ptr, u64 size, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ConstructIntraProcMemInfoForSend(void* ptr, u64 size, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ConstructNumInfoForSend(u64 num, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);

    HcclResult ConstructNotifyInfoForSend(u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ConstructNotifyVectorInfoForSend(u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ConstructDataLenForSend(u8*& exchangeDataPtr, u64& exchangeDataBlankSize);

    HcclResult ParseReceivedExchangeData() override;
    HcclResult ParseIpcMemInfo(
        void** memPtr, u64& size, u8* memName, u64& offset, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ParseIntraProcMemInfo(u64* addr, u64* size, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ParseNotifyInfo(u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ParseNotifyInfoEx(u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ParseNotifyVectorInfo(u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ParseCheckDataLen(ExchangeInfoSize& remoteInfoSize, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ParseMemNumInfo(u64& memNum, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);

    HcclResult GetLocalNotify(std::vector<HcclSignalInfo>& localNotify) override;
    HcclResult GetRemoteNotify(std::vector<HcclSignalInfo>& localNotify) override;

    HcclResult WaitPeerMemConfig(void** memPtr, const u8* memName, uint64_t size, u64 offset);
    HcclResult ExchangeMemAndNotifyMesg();
    HcclResult SendIpcMemMesg(void* ptr, u64 size) const;
    HcclResult RecvIpcMemMesg(void** memPtr, u8* memName, u64& offset);
    HcclResult SendMemMesgWithoutIpc(void* ptr, u64 size) const;
    HcclResult RecvMemMesgWithoutIpc(u64& addr, u8* memName, u64& offset);
    HcclResult ExchangeMemAndNotifyWithIpc();
    HcclResult ExchangeMemAndNotifyWithoutIpc();
    virtual HcclResult SignalRecord(
        std::shared_ptr<RemoteNotify>& remoteSignal, u64 remoteSignalAddr, u64 remoteSignalOffset, Stream& stream);
    void SetTransportRelationship();
    HcclResult SetLinkType();
    HcclResult CreateNotifyValueBuffer();
    HcclResult SumCheckSizeAndConsistent(
        ExInfoType exInfoType, u32 rightInfoSize, u64& blankSizeRecord, u64 exchangeDataBlankSize);
    void SetUseSdmaToSignalRecord();

    void* remoteInputPtr_;
    void* remoteOutputPtr_;
    std::vector<void*> remoteIpcMemPtrVector_;
    std::vector<void*> remoteIndOpHostMemPtrVector_;
    std::vector<void*> remoteIndOpDeviceMemPtrVector_;
    u64 remoteInputSize_{0};
    u64 remoteOutputSize_{0};
    std::vector<u64> remoteIpcMemSizeVector_;
    std::vector<u64> remoteIndOpHostMemSizeVector_;
    std::vector<u64> remoteIndOpDeviceMemSizeVector_;
    u64 remoteOutputOffsetValue_;
    u64 remoteInputOffsetValue_;
    std::vector<u64> remoteIpcMemOffsetValueVector_;
    std::vector<u64> remoteIndOpHostMemOffsetValueVector_;
    std::vector<u64> remoteIndOpDeviceMemOffsetValueVector_;

    bool useSdmaToSignalRecord_{false};

    HcclResult ReplaceMemAddr(
        Transport::Buffer& localMem, Transport::Buffer& remoteMem, Transport::Buffer& newLocalMem,
        Transport::Buffer& newRemoteMem, bool& isLocalHostAddr, bool& isRemoteHostAddr);
    HcclResult InitHcclMemExMgrWithMem(HcclMemEx* bufMem, u32 bufSize, HcclMemExMgr& hcommMemExMgr);
    HcclResult InitHcclMemExMgr(MachinePara& machinePara);
    HcclMemExMgr localHcclMemExMgr_;
    HcclMemExMgr remoteHcclMemExMgr_;

private:
    HcclResult ParseSpecifyLink(LinkTypeInServer& linkType);
    void SetMemIncludeFlag();
    HcclResult ConstructMemIncludeInfoForSend(u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ParseMemIncludeInfo(void** memPtr, u64& size, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    SecIpcName_t remoteOutputMemName_;
    SecIpcName_t remoteInputMemName_;
    std::vector<SecIpcName_t> remoteIpcMemNameVector_;
    std::vector<SecIpcName_t> remoteIndOpHostMemNameVector_;
    std::vector<SecIpcName_t> remoteIndOpDeviceMemNameVector_;
    static std::array<DeviceMem, MAX_MODULE_DEVICE_NUM> notifyValueMem_;
    static std::array<std::mutex, MAX_MODULE_DEVICE_NUM> notifyValueMutex_;
    const u64 notifyValueSize_{LARGE_PAGE_MEMORY_MIN_SIZE};            // 避免申请小页内存。最小2*1024*1024
    static std::array<Referenced, MAX_MODULE_DEVICE_NUM> instanceRef_; // 实例计数，用于释放静态资源
    ExchangeInfoSize exchangeInfoSize_{};
    bool isSioToHccs_{false};  // 是否是sio->hccs的链路
    bool isMemInclude_{false}; // input output是否在machinePara_.mem[0]范围内
};
} // namespace hccl

#endif /* TRANSPORT_P2P_PUB_H */

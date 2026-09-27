/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_TRANSPORT_BASE_PUB_H
#define HCOMM_TRANSPORT_BASE_PUB_H

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include "stream_pub.h"
#include "sal.h"
#include "dispatcher_pub.h"
#include "mem_name_repository_pub.h"
#include "task_logic_info_pub.h"
#include "transport_pub.h"

#include "hccl_socket.h"
#include "notify_pool.h"
#include "local_ipc_notify.h"
#include "remote_notify.h"
#include "hccl_mem_defs.h"

namespace hccl {

const std::map<LinkType, std::string> LINK_TYPE_STR_MAP{
    {LinkType::LINK_ONCHIP, "ONCHIP"},
    {LinkType::LINK_HCCS, "HCCS"},
    {LinkType::LINK_PCIE, "PCIE"},
    {LinkType::LINK_ROCE, "ROCE"},
    {LinkType::LINK_SIO, "SIO"},
    {LinkType::LINK_HCCS_SW, "HCCS_SW"},
    {LinkType::LINK_STANDARD_ROCE, "STANDARD_ROCE"},
    {LinkType::LINK_RESERVED, "RESERVED"}};

inline std::string GetLinkTypeEnumStr(LinkType linkType)
{
    auto iter = LINK_TYPE_STR_MAP.find(linkType);
    if (iter == LINK_TYPE_STR_MAP.end()) {
        return "Invalid LinkType";
    } else {
        return iter->second;
    }
}

class TransportBase {
public:
    explicit TransportBase(
        DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
        std::chrono::milliseconds timeout);
    virtual ~TransportBase();

    virtual HcclResult Init();
    virtual HcclResult DeInit();

    virtual HcclResult TxDataSignal(Stream& stream);
    virtual HcclResult RxDataSignal(Stream& stream);

    virtual HcclResult Stop();
    virtual HcclResult Resume();
    virtual HcclResult TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream);
    virtual HcclResult TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream);

    virtual HcclResult TxWithReduce(
        UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, const HcclDataType datatype,
        HcclReduceOp redOp, Stream& stream);
    virtual HcclResult TxWithReduce(
        const std::vector<TxMemoryInfo>& txWithReduceMems, const HcclDataType datatype, HcclReduceOp redOp,
        Stream& stream);

    virtual HcclResult RxWithReduce(
        UserMemType recvSrcMemType, u64 recvSrcOffset, void* recvDst, u64 recvLen, void* reduceSrc, void* reduceDst,
        u64 reduceDataCount, HcclDataType reduceDatatype, HcclReduceOp reduceOp, Stream& stream, const u64 reduceAttr);
    virtual HcclResult RxWithReduce(
        const std::vector<RxWithReduceMemoryInfo>& rxWithReduceMems, HcclDataType reduceDatatype, HcclReduceOp reduceOp,
        Stream& stream, const u64 reduceAttr);

    virtual bool IsSupportTransportWithReduce();

    virtual HcclResult RxAsync(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream);
    virtual HcclResult RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream);

    virtual HcclResult DataReceivedAck(Stream& stream);

    virtual HcclResult TxAck(Stream& stream);
    virtual HcclResult RxAck(Stream& stream);

    virtual HcclResult TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream);

    virtual HcclResult RxData(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream);

    virtual HcclResult TxPrepare(Stream& stream);
    virtual HcclResult RxPrepare(Stream& stream);

    virtual HcclResult TxDone(Stream& stream);
    virtual HcclResult RxDone(Stream& stream);

    // 保证send语义完成
    virtual HcclResult TxWaitDone(Stream& stream);
    // 保证recv语义完成
    virtual HcclResult RxWaitDone(Stream& stream);
    // TxWaitDone、RxWaitDone共同出现保证sendrecv语义完成

    virtual HcclResult Post(u32 notifyIdx, Stream& stream);
    virtual HcclResult Wait(u32 notifyIdx, Stream& stream, const u32 timeOut = NOTIFY_INVALID_WAIT_TIME);

    virtual HcclResult GetIndOpRemoteMemDetails(MemDetails** remoteMem, uint32_t* memNum, HcclMemType memType);
    virtual HcclResult GetIndOpRemoteMem(HcclMem** remoteMem, uint32_t* memNum);
    virtual HcclResult GetRemoteMem(UserMemType memType, void** remotePtr);
    virtual HcclResult GetRemoteMem(std::vector<void*>* remotePtrVec);
    virtual HcclResult GetRemoteMemKey(UserMemType memType, uint32_t* remoteMemKey);
    virtual HcclResult GetRemoteMemSize(UserMemType memType, u64& size);
    virtual HcclResult GetLocalRdmaNotify(std::vector<HcclSignalInfo>& rdmaNotify);
    virtual HcclResult GetDrainLocalDataNotify(void*& localAddr, uint32_t& lkey, HcclSignalInfo& dataNotify);
    virtual HcclResult GetRemoteRdmaNotifyAddrKey(std::vector<AddrKey>& rdmaNotifyAddr);
    virtual HcclResult GetLocalNotifyValueAddrKey(std::vector<AddrKey>& notifyValue);
    virtual HcclResult GetLocalMemDetails(UserMemType memType, MemDetails& memDetails);
    virtual HcclResult GetLocalNotify(std::vector<HcclSignalInfo>& localNotify);
    virtual HcclResult GetRemoteNotify(std::vector<HcclSignalInfo>& localNotify);

    virtual HcclResult GetAiQpInfo(std::vector<HcclQpInfoV2>& aiQpInfo);
    virtual HcclResult GetAiRMAQueueInfo(std::vector<HcclAiRMAQueueInfo>& aiRMAQueueInfo);
    virtual HcclResult GetTransportId(u32& id);
    HcclResult GetChipId(s64& chipId);
    HcclResult GetTxAckDevNotifyInfo(HcclSignalInfo& notifyInfo);
    HcclResult GetRxAckDevNotifyInfo(HcclSignalInfo& notifyInfo);
    HcclResult GetTxDataSigleDevNotifyInfo(HcclSignalInfo& notifyInfo);
    HcclResult GetRxDataSigleDevNotifyInfo(HcclSignalInfo& notifyInfo);
    inline hccl::LinkType GetLinkType() const { return transportAttr_.linkType; }

    inline bool GetSupportDataReceivedAck() const { return machinePara_.supportDataReceivedAck; }

    inline void SetSupportDataReceivedAck(bool supportDataReceivedAck)
    {
        machinePara_.supportDataReceivedAck = supportDataReceivedAck;
    }

    inline bool IsSpInlineReduce() const
    {
        bool isSpInlineReduce
            = transportAttr_.linkType == LinkType::LINK_HCCS || transportAttr_.linkType == LinkType::LINK_PCIE
              || transportAttr_.linkType == LinkType::LINK_SIO || transportAttr_.linkType == LinkType::LINK_HCCS_SW;
        return isSpInlineReduce;
    }

    inline u32 GetRemoteRank() const { return machinePara_.remoteWorldRank; }
    virtual HcclResult ConnectAsync([[maybe_unused]] u32& status) { return HCCL_SUCCESS; };
    virtual HcclResult ConnectQuerry([[maybe_unused]] u32& status) { return HCCL_SUCCESS; };

    virtual void Break() { return; }

    inline void EnableUseOneDoorbell() { useOneDoorbell_ = true; }

    inline bool GetUseOneDoorbellValue() { return useOneDoorbell_; }

    inline u32 GetNotifyNum() { return notifyNum_; }
    HcclResult OpenRemoteNotify(const std::vector<u8>& byteVector, std::shared_ptr<RemoteNotify>& remoteNotify);

    virtual HcclResult TxEnv(const void* ptr, const u64 len, Stream& stream);
    virtual HcclResult RxEnv(Stream& stream);

    virtual HcclResult
    WriteAsync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream);
    virtual HcclResult
    WriteSync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream);

    virtual HcclResult WriteReduceAsync(
        struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, const HcclDataType datatype,
        HcclReduceOp redOp, Stream& stream);

    virtual HcclResult
    ReadAsync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream);
    virtual HcclResult
    ReadSync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream);
    virtual HcclResult ReadReduceSync(
        struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, const HcclDataType datatype,
        HcclReduceOp redOp, Stream& stream);

    virtual HcclResult
    BatchTransferAsync(const HcommBatchTransferDesc* transferDescs, uint32_t descNum, Stream& stream);

    virtual HcclResult PostReady(Stream& stream);
    virtual HcclResult WaitReady(Stream& stream);

    virtual HcclResult PostFin(Stream& stream);
    virtual HcclResult WaitFin(Stream& stream);

    virtual HcclResult PostFinAck(Stream& stream);
    virtual HcclResult WaitFinAck(Stream& stream);
    virtual HcclResult Drain(Stream& stream);

    virtual HcclResult GetDrainRemSrcMem(void*& remoteAddr, uint32_t& remoteKey, uint32_t& size);
    TransportAttr GetTransportAttr();

    HcclResult SetStopFlag(bool value);
    bool GetStopFlag();
    virtual HcclResult Fence();
    virtual HcclResult UpdateRemoteAddr(void* remoteIn, void* remoteOut);

    std::vector<u8>& GetExchangeInfo() { return exchangeMsg_; }

    virtual bool GetIsUseAtomicWrite() { return useAtomicWrite_; }

    inline HcclResult GetSpecificNotify(HcclSignalInfo& notifyInfo, bool& isValid, const std::string& notifyName)
    {
        // 针对alltoallv算子aicpu cache, 提供Tx/RxAck和Tx/RxDataSignal的相关notify信息
        if (notifyName == "localSendReady") { // For RxDataSignal
            if (!localSendReadyNotify_) {
                isValid = false;
            } else {
                CHK_RET(localSendReadyNotify_->GetNotifyData(notifyInfo));
                isValid = true;
            }
        } else if (notifyName == "localSendDone") { // For RxAck
            if (!localSendDoneNotify_) {
                isValid = false;
            } else {
                CHK_RET(localSendDoneNotify_->GetNotifyData(notifyInfo));
                isValid = true;
            }
        } else if (notifyName == "remoteSendReady") { // For TxDataSignal
            if (!remoteSendReadyNotify_) {
                isValid = false;
            } else {
                CHK_RET(remoteSendReadyNotify_->GetNotifyData(notifyInfo));
                isValid = true;
            }
        } else if (notifyName == "remoteSendDone") { // For TxAck
            if (!remoteSendDoneNotify_) {
                isValid = false;
            } else {
                CHK_RET(remoteSendDoneNotify_->GetNotifyData(notifyInfo));
                isValid = true;
            }
        } else {
            HCCL_ERROR("[TransportBase][GetSpecificNotify] unsupported notifyName[%s]", notifyName.c_str());
            return HCCL_E_NOT_SUPPORT;
        }

        return HCCL_SUCCESS;
    }

    virtual HcclResult InitDrainNotifyInfo();

protected:
    virtual HcclResult FillExchangeDataTotalSize();
    virtual HcclResult ConstructExchangeForSend();
    virtual HcclResult ParseReceivedExchangeData();
    HcclResult ConstructExchangeDataForSend(u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ParseExchangeData(u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult ExchangeTgidMesg();
    // 以下两个接口用于ibv、tcp进行信息交换、校验
    HcclResult RecvAndCheckExchangeData(void);
    HcclResult SendExchangeData(void);

    // 以下接口仅用于P2P和host shm中转的子类
    HcclResult SendNotifyReadyMesg();
    HcclResult SendNotifyDoneMesg();
    HcclResult SendDeviceIpcNotifyReadyMesg();
    HcclResult SendDeviceIpcNotifyDoneMesg();
    HcclResult RecvNotifyReadyMesg();
    HcclResult RecvNotifyDoneMesg();
    HcclResult RecvDeviceIpcNotifyReadyMesg();
    HcclResult RecvDeviceIpcNotifyDoneMesg();
    HcclResult CheckLinkStatus();
    HcclResult CheckLinkMode();
    HcclResult LinkSendNotifyMesg();
    HcclResult LinkRecvNotifyMesg();

    // 以下接口用于aicpu侧的transport子类
    HcclResult SetNotify();
    HcclResult SetNotifyPtr(const TransportDeviceP2pData& transDevP2pData);
    HcclResult SignalInit(const std::shared_ptr<LocalNotify>& notify, std::shared_ptr<LocalIpcNotify>& ipcNotify);

    void SignalDestroy(); // TransportP2P & TranshportShm 公有信号销毁函数
    void DestroyDeviceSignal();
    void DestroyHostSignal();
    HcclResult CheckDeviceId();

    inline HcclResult CheckExchangeData()
    {
        CHK_PRT_RET(
            machinePara_.exchangeInfo.size() > MAX_EXCHANGE_DATA_LEN,
            HCCL_ERROR(
                "[[Check][ExchangeData]errNo[0x%016llx]custom exchange data size[%zu]is too large, "
                "Expected to less than[%llu]",
                HCCL_ERROR_CODE(HCCL_E_PARA), machinePara_.exchangeInfo.size(), MAX_EXCHANGE_DATA_LEN),
            HCCL_E_PARA);
        return HCCL_SUCCESS;
    }
    u64 exchangeDataTotalSize_;
    std::vector<u8> exchangeDataForSend_;
    std::vector<u8> exchangeDataForRecv_;
    DispatcherPub* dispatcher_;
    const std::unique_ptr<NotifyPool>& notifyPool_;
    std::shared_ptr<HcclSocket> defaultSocket_;
    MachinePara machinePara_;
    const std::chrono::milliseconds timeout_;
    std::shared_ptr<LocalIpcNotify> localSendReadyNotify_ = nullptr;
    std::shared_ptr<LocalIpcNotify> localSendDoneNotify_ = nullptr;
    std::shared_ptr<LocalIpcNotify> localSendReadyDeviceNotify_ = nullptr;
    std::shared_ptr<LocalIpcNotify> localSendDoneDeviceNotify_ = nullptr;
    std::vector<std::shared_ptr<LocalIpcNotify>> userLocalNotify_;

    std::shared_ptr<RemoteNotify> remoteSendReadyNotify_ = nullptr;
    std::shared_ptr<RemoteNotify> remoteSendDoneNotify_ = nullptr;

    std::shared_ptr<RemoteNotify> remoteSendReadyDeviceNotify_ = nullptr;
    std::shared_ptr<RemoteNotify> remoteSendDoneDeviceNotify_ = nullptr;
    std::vector<std::shared_ptr<RemoteNotify>> userRemoteNotify_;

    u64 remoteSendReadyAddress_{0};
    u64 remoteSendReadyOffset_{0};
    u64 remoteSendDoneOffset_{0};
    u64 remoteSendDoneAddress_{0};
    std::vector<u64> userRemoteNotifyAddr_;
    std::vector<u64> userRemoteNotifyOffset_;

    s32 recvPid_;
    s32 recvSdid_; // 超节点上device唯一标识, super pod device id
    NICDeployment nicDeploy_;

    bool useOneDoorbell_;
    TransportAttr transportAttr_;
    u32 notifyNum_;

    std::atomic<bool> stopFlag_{false};
    std::vector<u8> exchangeMsg_;
    bool useAtomicWrite_{false}; // 本端和对端同时使能atomic write时，才会使用atomic write，否则退化回普通模式
};

} // namespace hccl

#endif /* TRANSPORT_BASE_PUB_H */

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_DEVICE_IBVERBS_PUB_H
#define TRANSPORT_DEVICE_IBVERBS_PUB_H

#include <functional>
#include <atomic>
#include <cstdint>
#include <memory>
#include "buffer_key.h"
#include "rma_buffer_mgr.h"
#include "transport_ibverbs_pub.h"

namespace hccl {
constexpr u32 IBV_SGLIST_LEN_MAX = 2147483648;

enum class HcclWrOpCode {
    HCCL_WR_RDMA_WRITE = 0,
    HCCL_WR_RDMA_READ = 4,
};

class TransportDeviceIbverbs : public TransportIbverbs {
public:
    TransportDeviceIbverbs(
        DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
        std::chrono::milliseconds timeout, const TransportDeviceIbverbsData& transDevIbverbsData);
    ~TransportDeviceIbverbs() override;

    HcclResult Init() override;

    HcclResult TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream) override;
    HcclResult TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream) override;

    HcclResult RxAsync(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream) override;
    HcclResult RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream) override;

    HcclResult DataReceivedAck(Stream& stream) override;

    HcclResult TxAck(Stream& stream) override;
    HcclResult RxAck(Stream& stream) override;

    HcclResult TxDataSignal(Stream& stream) override;
    HcclResult RxDataSignal(Stream& stream) override;

    HcclResult TxWaitDone(Stream& stream) override;
    HcclResult TxWithReduce(
        UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, const HcclDataType datatype,
        HcclReduceOp redOp, Stream& stream) override;
    HcclResult TxWithReduce(
        const std::vector<TxMemoryInfo>& txWithReduceMems, const HcclDataType datatype, HcclReduceOp redOp,
        Stream& stream) override;

    HcclResult TxPrepare(Stream& stream) override;
    HcclResult RxPrepare(Stream& stream) override;

    HcclResult TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream) override;
    HcclResult RxData(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream) override;

    HcclResult TxDone(Stream& stream) override;
    HcclResult RxDone(Stream& stream) override;

    HcclResult WriteAsync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream);

    HcclResult WriteReduceAsync(
        struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, const HcclDataType datatype,
        HcclReduceOp redOp, Stream& stream) override;

    HcclResult
    ReadAsync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream) override;

    HcclResult
    BatchTransferAsync(const HcommBatchTransferDesc* transferDescs, uint32_t descNum, Stream& stream) override;

    HcclResult PostReady(Stream& stream);
    HcclResult WaitReady(Stream& stream);

    HcclResult PostFin(Stream& stream);
    HcclResult WaitFin(Stream& stream);

    HcclResult PostFinAck(Stream& stream);
    HcclResult WaitFinAck(Stream& stream);

    HcclResult Post(u32 notifyIdx, Stream& stream) override;
    HcclResult Wait(u32 notifyIdx, Stream& stream, const u32 timeOut = NOTIFY_INVALID_WAIT_TIME) override;

    HcclResult AddWrList(
        void* dstMemPtr, const void* srcMemPtr, u64 srcMemSize, u32 srcKey, u32 dstKey, WqeType wqeType, WrAuxInfo& aux,
        std::vector<WrInformation>& wrInfoVec, bool isLast = true);
    HcclResult GetMemInfo(UserMemType memType, void** dstMemPtr, unsigned int* dstKey, u64& dstMemSize);
    HcclResult TxPayLoad(
        UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, WqeType wqeType, WrAuxInfo& aux,
        std::vector<WrInformation>& wrInfoVec);
    HcclResult
    TxSendDataAndNotifyWithSingleQP(std::vector<WrInformation>& wrInfoVec, Stream& stream, bool useOneDoorbell = false);
    HcclResult TxSendDataAndNotify(std::vector<WrInformation>& wrInfoVec, Stream& stream, bool useOneDoorbell = false);
    HcclResult TxWrList(
        std::vector<WrInformation>& wrInfoVec, Stream& stream, std::vector<struct SendWrRsp>& opRspVec,
        u32 multiQpIndex = RDMA_INVALID_QP_INDEX);
    HcclResult
    SendWrList(u32 wrNum, WrInformation* wrlist, struct SendWrRsp* opRsp, u32 multiQpIndex = RDMA_INVALID_QP_INDEX);
    HcclResult SendWrlistExt(
        WrInformation wr[], struct SendWrRsp opRsp[], unsigned int sendNum, unsigned int* completeNum,
        u32 multiQpIndex = RDMA_INVALID_QP_INDEX);
    HcclResult TxSendWrlistExt(
        WrInformation wrList[], u32 sendNum, struct SendWrRsp opRsp[], unsigned int* completeNum,
        u32 multiQpIndex = RDMA_INVALID_QP_INDEX);
    HcclResult RdmaSendAsync(struct SendWr& wr, Stream& stream, WqeType wqeType, u64 notifyAddr, u32 notifyId) override;
    HcclResult RdmaSendAsync(
        std::vector<WrInformation>& wqeInfoVec, Stream& stream, bool useOneDoorbell = false,
        u32 multiQpIndex = RDMA_INVALID_QP_INDEX);
    HcclResult GetWrDataAddr(void* dstAddr, WqeType wqeType, u64& wrDataAddr, u32& notifyId);
    HcclResult TxSendWqe(
        void* dstMemPtr, u32 dstKey, const void* srcMemPtr, u32 srcKey, u64 srcMemSize, Stream& stream,
        WqeType wqeType);

    HcclResult ConstructPayLoadWqe(
        void* dstMemPtr, u32 dstKey, const void* src, u32 srcKey, u64 len, WqeType wqeType, WrAuxInfo& aux,
        std::vector<WrInformation>& wrInfoVec, u32 txSendDataTimes, bool isLast = true);
    HcclResult WriteCommon(
        const void* remoteAddr, const void* localAddr, u64 length, Stream& stream, WqeType wqeType,
        struct WrAuxInfo& aux);
    bool UseMultiQp();
    HcclResult TxSendDataAndNotifyWithMultiQP(
        std::vector<WrInformation>& wqeInfoVec, u32 actualMultiQpNum, Stream& stream, bool useOneDoorbell);
    u32 GetActualQpNum(u32 maxLength);
    HcclResult GetTransportId(u32& id) override;
    static HcclResult HnsPostSend(
        const TransportDeviceNormalData& ibvData, struct MemDetails* localMems, struct MemDetails* remoteMems,
        u32 memNum, HcclWrOpCode opCode, u64& dbInfo, bool fence = false);
    HcclResult Drain(Stream& stream) override;
    HcclResult InitDrainNotifyInfo() override;

private:
    bool IsModifyToAtomicWrite();
    using DeviceMemDetailsRmaMgr = hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, std::shared_ptr<RoceMemDetails>>;
    HcclResult InitMemDetails();
    HcclResult BuildMemDetailsRmaMgrs();
    HcclResult BatchTransferImpl(const HcommBatchTransferDesc* transferDescs, uint32_t descNum, Stream& stream);
    HcclResult ResolveTransferDesc(
        const HcommBatchTransferDesc& desc, const void*& remoteAddr, const void*& localAddr, u64& length,
        WqeType& wqeType, struct WrAuxInfo& aux);
    HcclResult SubmitWqeBatch(std::vector<WrInformation>& wrInfoVec, Stream& stream);

    struct RdmaAddrKeyResolveParam {
        const void* remoteAddr{nullptr};
        const void* localAddr{nullptr};
        u64 length{0};
        u32 dstKey{0};
        u32 srcKey{0};
        void* transLocalAddr{nullptr};
        void* transRemoteAddr{nullptr};
    };
    HcclResult ResolveRdmaAddrsAndKeys(RdmaAddrKeyResolveParam& param);
    HcclResult ResolveRdmaKeysFromMemDetails(RdmaAddrKeyResolveParam& param);
    HcclResult ResolveRdmaKeysFromIoMemRanges(RdmaAddrKeyResolveParam& param);

    TransportDeviceIbverbsData transDevIbverbsData_;
    void* notifyValueAddr_ = nullptr;
    MemDetails localInputMem_;
    MemDetails localOutputMem_;
    static std::atomic<u64> wrIdOffset_;
    u32 multiQpThreshold_{HCCL_MULTI_QP_THRESHOLD_DEFAULT};
    std::unique_ptr<DeviceMemDetailsRmaMgr> localMemDetailsRmaMgr_;
    std::unique_ptr<DeviceMemDetailsRmaMgr> remoteMemDetailsRmaMgr_;
    bool useMemDetailsLookup_{false};
};
} // namespace hccl

#endif /* TRANSPORT_DEVICE_IBVERBS_PUB_H */

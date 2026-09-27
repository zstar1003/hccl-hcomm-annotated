/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LINK_IBV_EXP_PUB_H
#define LINK_IBV_EXP_PUB_H

#include <functional>
#include <mutex>

#include "../host/transport_net_pub.h"
#include "network/hccp_common.h"
#include "workflow_pub.h"
#include "private_types.h"
#include "adapter_hccp.h"
#include "universal_concurrent_map.h"

inline void CopyAiWQInfo(struct HcclAiRMAWQ& dest, const struct AiDataPlaneWq& source, DBMode dbMode, u32 sl)
{
    dest.wqn = source.wqn;
    dest.bufAddr = source.bufAddr;
    dest.wqeSize = source.wqebbSize;
    dest.depth = source.depth;
    dest.headAddr = source.headAddr;
    dest.tailAddr = source.tailAddr;
    dest.dbMode = dbMode;
    if (dbMode == DBMode::SW_DB) {
        dest.dbAddr = source.swdbAddr;
    } else if (dbMode == DBMode::HW_DB) {
        dest.dbAddr = source.dbReg;
    }
    dest.sl = sl;
    HCCL_INFO(
        "CopyAiWQInfo: wqn[%u] bufAddr[%p] wqeSize[%u] depth[%u] headAddr[%p] tailAddr[%p] dbMode[%u]", dest.wqn,
        dest.bufAddr, dest.wqeSize, dest.depth, dest.headAddr, dest.tailAddr, dest.dbMode);
    return;
}

inline void CopyAiCQInfo(struct HcclAiRMACQ& dest, const AiDataPlaneCq& source, DBMode dbMode)
{
    dest.cqn = source.cqn;
    dest.bufAddr = source.bufAddr;
    dest.cqeSize = source.cqeSize;
    dest.depth = source.depth;
    dest.headAddr = source.headAddr;
    dest.tailAddr = source.tailAddr;
    dest.dbMode = dbMode;
    if (dbMode == DBMode::SW_DB) {
        dest.dbAddr = source.swdbAddr;
    } else if (dbMode == DBMode::HW_DB) {
        dest.dbAddr = source.dbReg;
    }

    HCCL_INFO(
        "CopyAiCQInfo: cqn[%u] bufAddr[%p] cqeSize[%u] depth[%u] headAddr[%p] tailAddr[%p] dbMode[%u]", dest.cqn,
        dest.bufAddr, dest.cqeSize, dest.depth, dest.headAddr, dest.tailAddr, dest.dbMode);
    return;
}
namespace hccl {
// WQE payload类型， 区分 notify 模式与数据类型模式
enum class WqeType {
    WQE_TYPE_DATA,             // 发送数据类型的WQE
    WQE_TYPE_DATA_NOTIFY,      // 发送数据同步 Notify 的WQE
    WQE_TYPE_ACK_NOTIFY,       // 发送信息同步 Notify 的WQE
    WQE_TYPE_DATA_ACK_NOTIFY,  // 发送数据接收确认 Notify 的WQE
    WQE_TYPE_DATA_WITH_NOTIFY, // 带有Notify信息的数据WQE
    WQE_TYPE_DATA_WITH_REDUCE, // 带有Reduce信息的数据WQE
    WQE_TYPE_READ_DATA,        // 读数据类型的WQE
    WQE_TYPE_RESEERVED
};

using WqeInfo = struct TagWqeInfo {
    struct SendWrlistDataExt wqeData {};
    u64 wqeType;
    u64 wqeDataOffset;
    u32 notifyId;
    TagWqeInfo() : wqeType(static_cast<u64>(WqeType::WQE_TYPE_DATA)), wqeDataOffset(0), notifyId(INVALID_UINT)
    {
        wqeData = {};
    }
};

struct CombineQpHandle {
    QpHandle qpHandle = nullptr;
    u32 preWrOpcode = INVALID_UINT; // 记录这个qp内最后一次下发的wr的opcode
    CombineQpHandle() {};
    CombineQpHandle(QpHandle& qpHandle) : qpHandle(qpHandle) {};
};

struct CombineQpInfo {
    struct AiQpInfo aiQpInfo {};
    u32 preWrOpcode = INVALID_UINT; // 记录这个qp内最后一次下发的wr的opcode
    CombineQpInfo() {};
    CombineQpInfo(struct AiQpInfo& aiQpInfo) : aiQpInfo(aiQpInfo) {};
};

namespace {
    constexpr u32 WAIT_US_COUNT = 1000;
    constexpr s32 REG_VALID = 1;
    // QP flag
    constexpr s32 QP_FLAG_RC = 0; // flag: 0 = RC, 1= UD，其它预留
    // QP mode
    constexpr s32 NORMAL_QP_MODE = 0;      // 普通的QP模式，但时不用
    constexpr s32 OFFLINE_QP_MODE = 1;     // 下沉模式的QP
    constexpr s32 OPBASE_QP_MODE = 2;      // 单算子模式的QP
    constexpr s32 OFFLINE_QP_MODE_EXT = 3; // 下沉模式(910B/910_93)QP
    constexpr s32 OPBASE_QP_MODE_EXT = 4;  // 单算子模式(910B/910_93)的QP

    // RDMA op type
    constexpr s32 RDMA_OP_WRITE = 0;
    constexpr s32 RDMA_OP_READ = 4;
    constexpr u32 RDMA_SEND_WRLIST_MAX_COUNT = 10;
    // RDMA multi QP
    constexpr u32 RDMA_ADDR_ALIGNMENT = 128;
    constexpr u32 RDMA_INVALID_QP_INDEX = 0xFFFF;

    // RDMA Write With Reduce DataType and OpType
    // reduceType: 0x0:int8 0x1:int16 0x2:int32 0x6:fp16 0x7:fp32 0x8:bf16
    enum class RdmaReduceDataType : uint8_t {
        RDMA_REDUCE_DATA_INT8 = 0,
        RDMA_REDUCE_DATA_INT16 = 1,
        RDMA_REDUCE_DATA_INT32 = 2,
        RDMA_REDUCE_DATA_FP16 = 6,
        RDMA_REDUCE_DATA_FP32 = 7,
        RDMA_REDUCE_DATA_BF16 = 8,
        RDMA_REDUCE_DATA_INVALID = 255
    };
    // reduce_op: 0x0:max 0x1:min 0x2:sum
    enum class RdmaReduceOpType : uint8_t {
        RDMA_REDUCE_OP_MAX = 0,
        RDMA_REDUCE_OP_MIN = 1,
        RDMA_REDUCE_OP_SUM = 2,
        RDMA_REDUCE_OP_INVALID = 255
    };

    constexpr uint8_t RDMA_REDUCE_DATA_TYPE_TABLE[HCCL_DATA_TYPE_RESERVED]
        = {static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INT8),
           static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INT16),
           static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INT32),
           static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_FP16),
           static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_FP32),
           static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID),
           static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID),
           static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID),
           static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID),
           static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID),
           static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_INVALID),
           static_cast<uint8_t>(RdmaReduceDataType::RDMA_REDUCE_DATA_BF16)};
    constexpr uint8_t RDMA_REDUCE_OP_TYPE_TABLE[HCCL_REDUCE_RESERVED]
        = {static_cast<uint8_t>(RdmaReduceOpType::RDMA_REDUCE_OP_SUM),
           static_cast<uint8_t>(RdmaReduceOpType::RDMA_REDUCE_OP_INVALID),
           static_cast<uint8_t>(RdmaReduceOpType::RDMA_REDUCE_OP_MAX),
           static_cast<uint8_t>(RdmaReduceOpType::RDMA_REDUCE_OP_MIN)};
} // namespace

class TransportIbverbs : public TransportNet {
public:
    explicit TransportIbverbs(
        DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
        std::chrono::milliseconds timeout);
    ~TransportIbverbs() override;

    HcclResult Init() override;

    HcclResult DeInit() override;

    HcclResult Stop() override;

    HcclResult Resume() override;

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
    bool IsSupportTransportWithReduce() override;
    HcclResult GetIndOpRemoteMemDetails(MemDetails** remoteMem, uint32_t* memNum, HcclMemType memType) override;
    HcclResult GetIndOpRemoteMem(HcclMem** remoteMem, uint32_t* memNum) override;
    HcclResult GetRemoteMem(UserMemType memType, void** remotePtr) override;
    HcclResult GetRemoteMemSize(UserMemType memType, u64& size) override;

    HcclResult TxPrepare(Stream& stream) override;
    HcclResult RxPrepare(Stream& stream) override;

    HcclResult TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream) override;
    HcclResult RxData(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream) override;

    HcclResult TxDone(Stream& stream) override;
    HcclResult RxDone(Stream& stream) override;

    HcclResult WriteAsync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream);
    HcclResult WriteSync(struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, Stream& stream);

    HcclResult WriteReduceAsync(
        struct Transport::Buffer& remoteBuf, struct Transport::Buffer& localBuf, const HcclDataType datatype,
        HcclReduceOp redOp, Stream& stream) override;

    HcclResult ReadAsync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream);
    HcclResult ReadSync(struct Transport::Buffer& localBuf, struct Transport::Buffer& remoteBuf, Stream& stream);

    HcclResult PostReady(Stream& stream);
    HcclResult WaitReady(Stream& stream);

    HcclResult PostFin(Stream& stream);
    HcclResult WaitFin(Stream& stream);

    HcclResult PostFinAck(Stream& stream);
    HcclResult WaitFinAck(Stream& stream);

    HcclResult Post(u32 notifyIdx, Stream& stream) override;
    HcclResult Wait(u32 notifyIdx, Stream& stream, const u32 timeOut = NOTIFY_INVALID_WAIT_TIME) override;

    static HcclResult GetTransportErrorCqe(
        const HcclNetDevCtx netDevCtx, std::vector<std::pair<TransportBase*, CqeInfo>>& infos, u32& num);
    HcclIpAddress& GetRemoteIp();
    HcclResult GetTransportId(u32& id) override;

    HcclResult Fence() override;
    HcclResult ExchangeCapabilityHybrid();
    HcclResult GetDrainRemSrcMem(void*& remoteAddr, uint32_t& remoteKey, uint32_t& size) override;
    HcclResult Drain(Stream& stream) override;

protected:
    HcclResult GetRemoteAddr(MemType memType, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult GetIndOpRemoteAddr(u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult GetRemoteNotifyAddr(u8*& exchangeDataPtr, u64& exchangeDataBlankSize, MemMsg& memMsg);
    HcclResult FillExchangeDataTotalSize() override;
    HcclResult ConstructExchangeForSend() override;
    HcclResult ParseReceivedExchangeData() override;

    HcclResult DestroyQP(QpHandle& qpHandle);
    HcclResult DestroyQP();
    HcclResult DeRegMR();
    void DeRegMRForQPhandles(MemMsg& memMsg);
    HcclResult DeRegOneMR(QpHandle& qpHandle, MemMsg& memMsg);
    // 初始化
    HcclResult GetNotifySize();
    s32 GetQpMode();
    bool UseMultiQp();
    HcclResult CreateQp();                                      // 创建QP
    HcclResult CreateSingleQp(s32 qpMode);                      // 两个rank之间仅创建一个QP
    HcclResult CreateMultiQp(s32 qpMode, u32 qpsPerConnection); // 两个rank之间创建多QP
    HcclResult ConnectSingleQp(std::function<bool()> needStop = []() {
        return false;
    });
    HcclResult ConnectMultiQp(
        u32 qpsPerConnection, std::function<bool()> needStop = []() {
            return false;
        });
    HcclResult ConnectQp();
    HcclResult InitQpConnect();
    HcclResult GetQpAttr();
    std::vector<u32> RdmaLengthSplit(u32 length, u32 splitNum);

    HcclResult TxSendDataAndNotifyWithMultiQP(
        std::vector<WqeInfo>& wqeInfoVec, u32 actualMultiQpNum, Stream& stream, bool useOneDoorbell = false);

    HcclResult GetWqeDataOffsetAndNotifyId(WqeType wqeType, u64& wqeDataOffset, u32& notifyId);

    HcclResult SendWqeList(QpHandle qpHandle, u32 wqeNum, struct SendWrlistDataExt* wqelist, struct SendWrRsp* opRsp);
    bool IsSupportRdmaNotify();
    bool IsTemplateMode();
    void DestroySignal();
    HcclResult IsUseQpCreateWithAttrs(bool& isUseQpCreateWithAttrs, s32 qpMode);
    HcclResult GetNicHandle();
    u32 GetQpsPerConnection();
    virtual HcclResult
    TxSendWqe(void* dstMemPtr, const void* srcMemPtr, u64 srcMemSize, Stream& stream, WqeType wqeType);
    virtual HcclResult TxSendNotifyWqe(MemMsg& memMsg, const void* srcMemPtr, u64 srcMemSize, Stream& stream);
    virtual HcclResult RegUserMem(MemType memType, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    virtual HcclResult RegCustomUserMem(u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    virtual HcclResult
    RegCustomUserMemWithMsg(void* addr, u64 size, MemMsg& memMsg, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult CreateNotifyBuffer(
        std::shared_ptr<LocalIpcNotify>& localNotify, MemType notifyType, u8*& exchangeDataPtr,
        u64& exchangeDataBlankSize, NotifyLoadType notifyLoadType = NotifyLoadType::HOST_NOTIFY);
    HcclResult CreateNotifyVectorBuffer(
        std::vector<std::shared_ptr<LocalIpcNotify>>& notifyVector, u8*& exchangeDataPtr, u64& exchangeDataBlankSize);
    HcclResult CreateNotifyValueBuffer();
    HcclResult CreateOneQp(
        s32 qpMode, u32 qpsPerConnection, QpHandle& qpHandle, AiQpInfo& aiQpInfo, bool useAicpu = false,
        u32 udpSport = 0);
    HcclResult AddWqeList(
        void* dstMemPtr, const void* srcMemPtr, u64 srcMemSize, WqeType wqeType, WrAuxInfo& aux,
        std::vector<WqeInfo>& wqeInfoVec);
    virtual HcclResult GetMemInfo(UserMemType memType, void** dstMemPtr, u64* dstMemSize);
    virtual HcclResult TxPayLoad(
        UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, WqeType wqeType, WrAuxInfo& aux,
        std::vector<WqeInfo>& wqeInfoVec);
    virtual HcclResult
    TxSendDataAndNotifyWithSingleQP(std::vector<WqeInfo>& wqeInfoVec, Stream& stream, bool useOneDoorbell = false);
    virtual HcclResult
    TxSendDataAndNotify(std::vector<WqeInfo>& wqeInfoVec, Stream& stream, bool useOneDoorbell = false);
    virtual HcclResult TxWqeList(
        std::vector<WqeInfo>& wqeInfoVec, Stream& stream, std::vector<struct SendWrRsp>& opRspVec,
        u32 multiQpIndex = RDMA_INVALID_QP_INDEX);
    virtual HcclResult RdmaSendAsync(
        std::vector<WqeInfo>& wqeInfoVec, Stream& stream, bool useOneDoorbell = false,
        u32 multiQpIndex = RDMA_INVALID_QP_INDEX);
    HcclResult RdmaSendAsyncHostNIC(std::vector<WqeInfo>& wqeInfoVec, Stream& stream);
    virtual HcclResult
    RdmaSendAsync(struct SendWr& wr, Stream& stream, WqeType wqeType, u64 notifyOffset, u32 notifyId);
    HcclResult RdmaSendAsyncHostNIC(struct SendWrlistDataExt& wr, Stream& stream, WqeType wqeType, u64 notifyOffset);

    HcclResult GetLocalNotify(std::vector<HcclSignalInfo>& localNotify) override;
    HcclResult GetLocalRdmaNotify(std::vector<HcclSignalInfo>& rdmaNotify) override;
    HcclResult
    GetDrainLocalDataNotify(void*& localAddr, uint32_t& lkey, HcclSignalInfo& dataNotify) override; // 仅获取dataNotify
    HcclResult GetRemoteRdmaNotifyAddrKey(std::vector<AddrKey>& rdmaNotifyAddr) override;
    HcclResult GetLocalNotifyValueAddrKey(std::vector<AddrKey>& notifyValue) override;
    HcclResult GetRemoteMemKey(UserMemType memType, uint32_t* remoteMemKey) override;
    HcclResult GetLocalMemDetails(UserMemType memType, MemDetails& memDetails) override;
    HcclResult GetAiQpInfo(std::vector<HcclQpInfoV2>& aiQpInfo) override;
    HcclResult GetAiRMAQueueInfo(std::vector<HcclAiRMAQueueInfo>& aiRMAQueueInfo) override;
    HcclResult ConstructPayLoadWqe(
        void* dstMemPtr, const void* src, u64 len, WqeType wqeType, WrAuxInfo& aux, std::vector<WqeInfo>& wqeInfoVec,
        u32 txSendDataTimes);
    u32 GetActualQpNum(u32 maxLength);
    HcclResult WriteCommon(
        const void* remoteAddr, const void* localAddr, u64 length, Stream& stream, WqeType wqeType,
        struct WrAuxInfo& aux);

    virtual void ModifyAtomicWriteAfterReduce(u32& preWrOpcode, u64 wqeType, u32& opcode, u32& immData);

    static std::array<DeviceMem, MAX_MODULE_DEVICE_NUM> notifyValueMem_;
    static std::array<std::mutex, MAX_MODULE_DEVICE_NUM> notifyValueMutex_;
    const u64 notifyValueSize_{LARGE_PAGE_MEMORY_MIN_SIZE};            // 避免申请小页内存。最小2*1024*1024
    static std::array<Referenced, MAX_MODULE_DEVICE_NUM> instanceRef_; // 实例计数，用于释放静态资源

    bool isHybridMode_ = false; // 是否为混合模式:hostNic -- DeviceNic
    std::vector<CombineQpHandle> combineQpHandles_;
    std::vector<CombineQpHandle> multiCombineQpHandles_;

    CombineQpInfo combineAiQpInfo_{};
    std::vector<CombineQpInfo> combineAiQpInfos_;
    u32 qpsPerConnection_;
    u32 notifySize_;

    std::shared_ptr<LocalIpcNotify> ackNotify_;
    std::shared_ptr<LocalIpcNotify> dataAckNotify_;

    s32 access_;

    std::array<MemMsg, static_cast<u32>(MemType::MEM_TYPE_RESERVED)> memMsg_;
    std::array<MemMsg, static_cast<u32>(MemType::MEM_TYPE_RESERVED)> remoteMemMsg_;

    std::vector<MemMsg> userDeviceMemMsg_;
    std::vector<MemMsg> userHostMemMsg_;

    std::vector<MemMsg> remoteUserDeviceMemMsg_;
    std::vector<MemMsg> remoteUserHostMemMsg_;
    std::unique_ptr<HcclMem[]> remoteMemsPtr_;
    u32 remoteMemsNum_{0};
    std::mutex remoteMemsMutex_;

    MemMsg remoteDataNotifyMsg_;
    std::shared_ptr<LocalIpcNotify> dataNotify_;

    std::vector<std::shared_ptr<LocalIpcNotify>> multiQpDataNotify_;
    std::vector<MemMsg> multiQpDataNotifyMemMsg_;
    std::vector<MemMsg> multiQpDataNotifyRemoteMemMsg_;

    std::vector<MemMsg> userRemoteNotifyMsg_;

    std::vector<std::vector<MemMsg>> userMultiQpRemoteNotifyMsg_;
    std::vector<std::vector<std::shared_ptr<LocalIpcNotify>>> userMultiQpLocalNotify_;

    HcclWorkflowMode workFlowMode_; // 工作模式
    u32 sqeCounter_;
    u32 currentQP_;

    std::vector<MrHandle> mrHandles_;
    RdmaHandle nicRdmaHandle_{nullptr};
    bool fence_{false};

    DevType localDeviceType{DevType::DEV_TYPE_COUNT};

    QPMode qpMode_{QPMode::INVALID}; // 是否为普通QP模式

private:
    static void ProcessCqeInfo(
        const s32 deviceId, const struct CqeErrInfo* infolist, const u32 cqeNum,
        std::vector<std::pair<TransportBase*, CqeInfo>>& infos);
    // bit[63:32] devicePhyId, bit[31:0] qpn
    static UniversalConcurrentMap<u64, TransportIbverbs*> g_qpn2IbversLinkMap_;
    static bool g_flag;
    static bool g_isSupCqeErrInfoListConfig;
    static u32 cqeErrQpn_;
    bool isCapture_{false};
};
} // namespace hccl

#endif /* LINK_IBV_EXP_PUB_H */

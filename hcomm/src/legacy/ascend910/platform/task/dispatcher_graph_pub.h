/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DISPATCHER_FFTS_PUB_H
#define HCCL_DISPATCHER_FFTS_PUB_H

#include <vector>
#include "dispatcher_pub.h"

namespace hccl {
constexpr u16 CONTEXT_MAX_NUM = 128;
constexpr uint32_t SDMA_FP32_ATOMIC_ADD_SQE = 0x1E71;
constexpr uint32_t SDMA_FP32_ATOMIC_MOVE_SQE = 0x1E70;
constexpr uint16_t FFTS_TIMEOUT_MAX = 65535;

using FftsSdmaSqeHeader = union {
    uint32_t word;
    struct {
        uint32_t opcode : 4;   // 0:非atomic搬运 1:atomic add 2:atomic max 3:atomic min 4:atomic equal 5:memory set
                               // 6:L2Cache Preload 7:L2Cache Prewriteback 8:L2Cache invalid 9:L2Cache Flush
        uint32_t datatype : 4; // 0:int8 1:int16 2:int32 6:fp16 normal 7:fp32 8:bf16 9:fp16 sat
        uint32_t ie2 : 1;      // 完成后是否上报中断; HCCL设为0
        uint32_t sssv : 1;     // source address对应的substream id是否有效; HCCL设为1
        uint32_t dssv : 1;     // destination address对应的substream id是否有效; HCCL设为1
        uint32_t sns : 1;      // source address对应的安全属性, 0:secure 1:non-secure; HCCL设为1
        uint32_t dns : 1;      // destination address对应的安全属性, 0:secure 1:non-secure; HCCL设为1
        uint32_t qos : 4;      // 0
        uint32_t sro : 1;      // 0
        uint32_t dro : 1;      // 0
        uint32_t partid : 8;   // 0
        uint32_t mpam : 1;     // 0
        uint32_t pmg : 2;      // 0
        uint32_t format : 1;   // 0
        uint32_t res6 : 1;     // 0
    } bit;
};

class DispatcherGraph : public DispatcherPub {
public:
    explicit DispatcherGraph(const s32 deviceLogicId);
    ~DispatcherGraph() override;

    HcclResult SignalRecord(
        HcclRtNotify signal, hccl::Stream& stream, u32 userRank, u64 offset = INVALID_U64,
        s32 stage = INVALID_VALUE_STAGE, bool inchip = false, u64 signalAddr = INVALID_U64,
        u32 notifyId = INVALID_UINT) override;
    HcclResult SignalWait(
        HcclRtNotify signal, hccl::Stream& stream, u32 userRank, u32 remoteUserRank, s32 stage = INVALID_VALUE_STAGE,
        bool inchip = false, u32 notifyId = INVALID_UINT, u32 timeOut = NOTIFY_INVALID_WAIT_TIME) override;
    HcclResult MemcpyAsync(
        hccl::DeviceMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream,
        u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP) override;
    HcclResult ReduceAsync(
        const void* src, void* dst, u64 dataCount, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream,
        HcclReduceType reduceType = HcclReduceType::HCCL_TBE_REDUCE) override;
    HcclResult InlineReduceAsync(
        const void* src, u64 dataCount, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream, void* dst,
        u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP) override;
    HcclResult RdmaSend(
        u32 dbindex, u64 dbinfo, const struct SendWr& wr, hccl::Stream& stream,
        u32 remoteUserRank = INVALID_VALUE_RANKID, bool isCapture = false) override;
    HcclResult RdmaSend(
        u32 dbindex, u64 dbinfo, const struct SendWr& wr, hccl::Stream& stream, u32 userRank, u64 offset,
        bool isCapture = false) override;

    HcclResult ResetGraphCtx(bool enableCache, const std::string& key, bool useGraphConstructorV2 = false) override;
    HcclResult LaunchTasksEx(Stream& stream, std::vector<Stream>& subStreams) override;
    void SetNormalMode() override;
    HcclResult SetMultiQpMode(bool multiQpMode) override;
    virtual HcclResult SignalRecord(Stream& stream, u64 notifyId) override;
    virtual HcclResult SignalWait(Stream& stream, u32 notifyId, u32 timeOut) override;

private:
    HcclResult TbeReduceAsync(
        const void* src1, const void* src2, u64 count, const HcclDataType dataType, HcclReduceOp redOp, Stream& stream,
        const void* dst);
    HcclResult VectorReduce(
        const void* src1, const void* src2, u64 count, const HcclDataType dataType, HcclReduceOp redOp, Stream& stream,
        const void* dst);
    HcclResult VectorReduceLoop(
        const void* src1, const void* src2, u64 count, const HcclDataType dataType, HcclReduceOp redOp, Stream& stream,
        const void* dst);
    HcclResult TailVectorReduce(
        const void* tailSrc1, const void* tailSrc2, u64 tailCount, const HcclDataType dataType, HcclReduceOp redOp,
        Stream& stream, void* tailDst);
    __inline__ __attribute__((always_inline)) HcclResult SignalTaskParaSave(
        HcclRtNotify signal, Stream& stream, u32 userRank, u32 remoteUserRank, u64 offset, s32 stage, TaskType taskType,
        uint64_t beginTime, u32 ctxIdx);
    HcclResult SetGraphTailVectorReduceDescSdma(
        void* devMem, const void* tailSrc, u64 count, const HcclDataType dataType, HcclReduceOp redOp, Stream& stream);
    HcclResult SetGraphDescVectorReduce(
        const void* src, const void* dst, int count, void* addrListDevMemPtr, void* funcAddr, uint32_t numBlocks,
        const HcclDataType dataType, HcclReduceOp redOp, Stream& stream);
    HcclResult GetNotifyDfxInfo(HcclRtNotify signal, u32 userRank, u64& offset, u32& remoteUserRank, u64& notifyID);
    void* fftsCtxsPtr;
    bool disableFfts_;
    bool multiQpMode_;
};
} // namespace hccl
#endif // HCCL_DISPATCHER_FFTS_PUB_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include <chrono>
#include "ub_conn_lite.h"
#include "log.h"
#include "exception_util.h"
#include "udma_data_struct.h"
#include "internal_exception.h"
#include "string_util.h"
#include "binary_stream.h"
#include "data_type.h"

constexpr u32 MAX_LOG_TIMEOUT_MS = 500;
namespace Hccl {
constexpr u32 ADDR_BIT_OFFSET = 32;
constexpr u32 SQE_SIZE_128 = 128;
constexpr u32 SQE_SIZE_64 = 64;
constexpr u32 SQE_INLINE_DATA_SIZE = 16;
constexpr u32 RAW_SIZE = 16;
constexpr u32 RMT_EID_BYTE_SIZE = 16;
constexpr u32 PI_NUM_TWO = 2;
constexpr u32 WRITE_WITH_NOTIFY_OPCODE = 0x5; // 注意: 与aicpu_task_cache_entry.cc保持一致
constexpr u32 ADDR_BIT_LOW = 0xffffffff;
constexpr u32 UB_DMA_MAX_READ_WEITE_SIZE = 256 * 1024 * 1024; // Byte, UB协议一次传输的最大size
constexpr u32 UB_RELAX_ORDER = 0x1; // Relax Order表示当前SQE与后续Strong Order SQE有保序要求
constexpr u32 UB_STRONG_ORDER = 0x2; // Strong Order表示当前SQE有保序要求，该SQE不能超越前面的Relax Order SQE

static std::map<DataType, u32> g_ubmaDataTypeMap
    = {{DataType::INT8, 0x0},   {DataType::INT16, 0x1},   {DataType::INT32, 0x2}, {DataType::UINT8, 0x3},
       {DataType::UINT16, 0x4}, {DataType::UINT32, 0x5},  {DataType::FP16, 0x6},  {DataType::FP32, 0x7},
       {DataType::BFP16, 0x8},  {DataType::BF16_SAT, 0x9}};

static std::map<ReduceOp, u32> g_ubmaDataOpMap = {{ReduceOp::SUM, 0xA}, {ReduceOp::MAX, 0x8}, {ReduceOp::MIN, 0x9}};

void UbConnLite::FillCommSqe(
    UdmaSqeCommon* sqe, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, u32 opCode, SlicePosition slicePos)
{
    u32 cqeEn = (cfg.cqeEn && (slicePos == SlicePosition::LAST || slicePos == SlicePosition::ONLY)) ? 1 : 0;
    sqe->cqe = cqeEn;
    sqe->owner = (pi == (sqDepth_ - 1)) ? 1 : 0;
    sqe->opcode = opCode;
    sqe->tpn = tpn_;

    if (cfg.userConfig) {
        sqe->placeOdr = cfg.placeOdr;
        sqe->compOrder = cfg.compOrder;
        sqe->fence = cfg.fence;
    } else {
        // 当前片是ONLY片(只有一片的情况)和最后一片的情况，全严格保序
        if (slicePos == SlicePosition::ONLY || slicePos == SlicePosition::LAST) {
            sqe->placeOdr = UB_STRONG_ORDER;
            sqe->compOrder = 1;
            sqe->fence = 1;
        } else {
            // 中间片写死配置，第一片由全局cfg配置
            sqe->placeOdr = (slicePos == SlicePosition::MIDDLE) ? UB_RELAX_ORDER : cfg.placeOdr;
            sqe->compOrder = (slicePos == SlicePosition::MIDDLE) ? 0 : cfg.compOrder;
            sqe->fence = (slicePos == SlicePosition::MIDDLE) ? 0 : cfg.fence;
        }
    }

    sqe->se = 1;           // 表示是否使能solicited event
    sqe->rmtJettyType = 1; // 00 JFR  01:JETTY  10:jettyGroup 11:reserved
    s32 ret = memcpy_sp(sqe->rmtEid, RMT_EID_BYTE_SIZE, rmtReverseEid_.raw, RAW_SIZE);
    if (UNLIKELY(ret != 0)) {
        HCCL_ERROR("UbConnLite::FillCommSqe FillCommSqe memcpy failed, ret=%d", ret);
        THROW<InternalException>(StringFormat("UbConnLite::FillCommSqe memcpy_sp failed, ret = %d", ret));
    }

    sqe->sgeNum = 1;
    sqe->targetHint = 0;
    sqe->rmtObjId = rmt.GetTokenId();
    sqe->tokenEn = 1;
    sqe->rmtTokenValue = rmt.GetTokenValue();
    sqe->rmtAddrLow = rmt.GetAddr() & ADDR_BIT_LOW;
    sqe->rmtAddrHigh = rmt.GetAddr() >> ADDR_BIT_OFFSET;
    HCCL_INFO(
        "UbConnLite FillCommSqe UdmaSqeCommon slicePos[%d] sqe->cqe = %u, sqe->owner = %u sqe->opcode = %u, "
        "sqe->tpn = %u, sqe->rmtObjId = %u, sqe->rmtAddrLow = %u, sqe->rmtAddrHigh = %u, sqe->placeOdr = %u, "
        "sqe->compOrder = %u, sqe->fence = %u",
        slicePos, sqe->cqe, sqe->owner, sqe->opcode, sqe->tpn, sqe->rmtObjId, sqe->rmtAddrLow, sqe->rmtAddrHigh,
        sqe->placeOdr, sqe->compOrder, sqe->fence);
}

void UbConnLite::FillCommSqeReduceInfo(UdmaSqeCommon& sqeComm, ReduceOp reduceOp, DataType dataType, u32 udfType) const
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    sqeComm.inlinedata.udfData.udfType = udfType; // 0代表inline reduce

    if ((g_ubmaDataOpMap.find(reduceOp) != g_ubmaDataOpMap.end())
        && (g_ubmaDataTypeMap.find(dataType) != g_ubmaDataTypeMap.end())) {
        sqeComm.inlinedata.udfData.reduceOp = g_ubmaDataOpMap.at(reduceOp);
        sqeComm.inlinedata.udfData.reduceType = g_ubmaDataTypeMap.at(dataType);
    } else {
        THROW<InvalidParamsException>(StringFormat(
            "%s reduceOp[%s] or type[%s] is not supported.", __func__, reduceOp.Describe().c_str(),
            dataType.Describe().c_str()));
    }

    // udf字段是否有效
    sqeComm.udfFlag = 1;

    HCCL_INFO(
        "[UbConnLite::%s] end, reduceOp[%s], reduceType[%s]", __func__, reduceOp.Describe().c_str(),
        dataType.Describe().c_str());
}

void UbConnLite::ProcessSlices(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, u32 maxSliceSize,
    std::function<void(const RmaBufSliceLite&, const RmtRmaBufSliceLite&, SlicePosition)> processOneSlice,
    DataType dataType) const
{
    (void)dataType;
    // reduce操作需要保证切片大小是数据类型大小的整数倍
    u64 sliceSize = static_cast<u64>(maxSliceSize);

    u64 locBufSize = loc.GetSize();
    u64 sliceNum = locBufSize / sliceSize;
    u64 lastSliceSize = locBufSize % sliceSize;

    u64 totalSize = sliceNum * sliceSize;

    if (UNLIKELY(loc.GetAddr() > UINT64_MAX - totalSize || rmt.GetAddr() > UINT64_MAX - totalSize)) {
        THROW<InternalException>("integer overflow occurs");
    }
    for (u64 sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++) {
        u64 offset = sliceIdx * sliceSize;
        u64 locAddr = loc.GetAddr() + offset;
        u64 rmtAddr = rmt.GetAddr() + offset;

        HCCL_INFO(
            "[UbConnLite::%s] Slice[%llu]: offset=0x%llx, locAddr=0x%llx, rmtAddr=0x%llx, size=0x%llx", __func__,
            sliceIdx, offset, locAddr, rmtAddr, sliceSize);

        RmaBufSliceLite locSlice(locAddr, sliceSize, 0, loc.GetTokenId());

        RmtRmaBufSliceLite rmtSlice(rmtAddr, sliceSize, 0, rmt.GetTokenId(), rmt.GetTokenValue(), UINT32_MAX);
        SlicePosition slicePos = (sliceIdx == 0) ? SlicePosition::FIRST : SlicePosition::MIDDLE;
        if ((sliceIdx == sliceNum - 1) && lastSliceSize == 0) {
            // SlicePosition::ONLY表示既是首片又是尾片的情况，只有一片的情况
            slicePos = (sliceIdx == 0) ? SlicePosition::ONLY : SlicePosition::LAST;
        }
        processOneSlice(locSlice, rmtSlice, slicePos);
    }

    if (lastSliceSize > 0) {
        RmaBufSliceLite lastLocSlice(loc.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, loc.GetTokenId());

        RmtRmaBufSliceLite lastRmtSlice(
            rmt.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, rmt.GetTokenId(), rmt.GetTokenValue(), UINT32_MAX);
        SlicePosition slicePos = (sliceNum == 0) ? SlicePosition::ONLY : SlicePosition::LAST;
        processOneSlice(lastLocSlice, lastRmtSlice, slicePos);
        sliceNum++;
    }

    HCCL_INFO(
        "[UbConnLite::%s] end, locBufSize[%llu], sliceNum[%llu], sliceSize[%llu], lastSliceSize[%llu]", __func__,
        locBufSize, sliceNum, sliceSize, lastSliceSize);
}

void UbConnLite::ProcessSlicesWithNotify(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, u32 maxSliceSize,
    std::function<void(const RmaBufSliceLite&, const RmtRmaBufSliceLite&, SlicePosition)> processOneSlice,
    std::function<void(const RmaBufSliceLite&, const RmtRmaBufSliceLite&, SlicePosition)> processOneSliceWithNotify,
    DataType dataType) const
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    // reduce操作需要保证切片大小是数据类型大小的整数倍
    u32 sliceSize = maxSliceSize;
    if (dataType != DataType::INVALID) {
        u32 dataTypeSize = DATA_TYPE_SIZE_MAP.at(dataType);
        sliceSize = maxSliceSize / dataTypeSize * dataTypeSize;
    }

    u64 locBufSize = loc.GetSize();
    u64 sliceNum = locBufSize / sliceSize;
    u64 lastSliceSize = locBufSize % sliceSize;
    if (sliceNum > 0 && lastSliceSize == 0) {
        sliceNum--;
        lastSliceSize = sliceSize;
    }
    u64 totalSize = sliceNum * sliceSize;
    if (UNLIKELY(loc.GetAddr() > UINT64_MAX - totalSize || rmt.GetAddr() > UINT64_MAX - totalSize)) {
        THROW<InternalException>("integer overflow occurs");
    }
    for (u64 sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++) {
        RmaBufSliceLite locSlice(loc.GetAddr() + sliceIdx * sliceSize, sliceSize, 0, loc.GetTokenId());

        RmtRmaBufSliceLite rmtSlice(
            rmt.GetAddr() + sliceIdx * sliceSize, sliceSize, 0, rmt.GetTokenId(), rmt.GetTokenValue(), UINT32_MAX);
        SlicePosition slicePos = (sliceIdx == 0) ? SlicePosition::FIRST : SlicePosition::MIDDLE;
        processOneSlice(locSlice, rmtSlice, slicePos);
    }

    if (lastSliceSize > 0) {
        RmaBufSliceLite lastLocSlice(loc.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, loc.GetTokenId());

        RmtRmaBufSliceLite lastRmtSlice(
            rmt.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, rmt.GetTokenId(), rmt.GetTokenValue(), UINT32_MAX);
        SlicePosition slicePos = (sliceNum == 0) ? SlicePosition::ONLY : SlicePosition::LAST;
        processOneSliceWithNotify(lastLocSlice, lastRmtSlice, slicePos);
    }

    HCCL_INFO(
        "[UbConnLite::%s] end, locBufSize[%llu], sliceNum[%llu], sliceSize[%u], lastSliceSize[%llu]", __func__,
        locBufSize, sliceNum, sliceSize, lastSliceSize);
}

void UbConnLite::FillOneSqeWrite(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, UdmaSqeWrite* sqe,
    UdmaSqOpcode opCode, SlicePosition slicePos)
{
    HCCL_INFO("[UbConnLite::%s] start, loc size[%llu]", __func__, loc.GetSize());

    sqe->comm.inlineEn = 0;
    FillCommSqe(&(sqe->comm), rmt, cfg, opCode, slicePos);
    FillLocalSgeSqe(&(sqe->u.sge), loc);
    if (sqe->u.sge.length == 0) {
        sqe->comm.sgeNum = 0;
    }

    HCCL_INFO("[UbConnLite::%s] end", __func__);
}

void UbConnLite::LaunchOneWqe(UdmaSqeWrite* sqe, UdmaSqOpcode opCode)
{
    HCCL_INFO("[UbConnLite::%s] start, opCode[%s]", __func__, opCode.Describe().c_str());

    // sqOffset是用于计算Ubjetty中下wqe位置的偏移，小于sqDepth
    u32 sqOffset = pi % sqDepth_;
    if (sqOffset < sqDepth_ && (sqOffset + 1) >= sqDepth_) {
        piDetourCount++;
    }
    // pi维护用于传入DB Send用于Rtsq 敲door bell，要求u16数据结构并且自然增长
    pi = pi + 1;

    // 写wqe到va
    u8* va = ReinterpretAs<u8*>(sqVa_ + sqOffset * SQE_SIZE_64);
    if (!dwqeCacheLocked_) {
        auto ret = memcpy_sp(va, SQE_SIZE_64, sqe, SQE_SIZE_64);
        if (UNLIKELY(ret != 0)) {
            THROW<InternalException>(StringFormat("[UbConnLite::%s] memcpy_sp failed, ret = %d", __func__, ret));
        }
    }

    HCCL_INFO(
        "[UbConnLite::%s] end, dieId_[%u], funcId_[%u], jettyId_[%u], pi[%u], ci[%u]", __func__, dieId_, funcId_,
        jettyId_, pi, ci);
}

void UbConnLite::FillOneWqeWithNotify(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, UdmaSqeWriteWithNotify* sqe,
    const RmtRmaBufSliceLite& notify, u64 notifyData, u32 opCode, SlicePosition slicePos)
{
    HCCL_INFO("[UbConnLite::%s] start, locSize[%u], opCode[%u]", __func__, loc.GetSize(), opCode);

    // 填充sqe
    sqe->comm.inlineEn = 0;
    FillCommSqe(&(sqe->comm), rmt, cfg, WRITE_WITH_NOTIFY_OPCODE, slicePos);
    FillNotifySqe(&(sqe->notify), notify, notifyData);
    FillLocalSgeSqe(&(sqe->localU.sge), loc);
    if (sqe->localU.sge.length == 0) {
        sqe->comm.sgeNum = 0;
    }
    sqe->rsv1 = 0;
    sqe->rsv2 = 0;

    HCCL_INFO("[UbConnLite::%s] end", __func__);
}

void UbConnLite::LaunchOneWqeWithNotify(UdmaSqeWriteWithNotify* sqe, u32 opCode)
{
    HCCL_INFO("[UbConnLite::%s] start, opCode[%u]", __func__, opCode);

    // sqOffset是用于计算Ubjetty中下wqe位置的偏移，小于sqDepth
    u32 sqOffset = pi % sqDepth_;
    if (sqOffset < sqDepth_ && (sqOffset + PI_NUM_TWO) >= sqDepth_) {
        piDetourCount++;
    }
    // pi维护用于传入DB Send用于Rtsq 敲door bell，要求u16数据结构并且自然增长
    pi = pi + PI_NUM_TWO;

    u8* va = ReinterpretAs<u8*>(sqVa_ + sqOffset * SQE_SIZE_64);
    if (!dwqeCacheLocked_) {
        // 带notify的wqe是96字节, 需要占用两个wqebb, 实际占用128字节
        if (sqOffset == sqDepth_ - 1) {
            MemorySetAndCopy(va, SQE_SIZE_64, sqe);
            va = ReinterpretAs<u8*>(sqVa_);
            MemorySetAndCopy(va, SQE_SIZE_64, ReinterpretAs<u8*>(sqe) + SQE_SIZE_64);
        } else {
            MemorySetAndCopy(va, SQE_SIZE_128, sqe);
        }
    }

    HCCL_INFO(
        "[UbConnLite::%s] end, dieId_[%u], funcId_[%u], jettyId_[%u], pi[%u], ci[%u]", __func__, dieId_, funcId_,
        jettyId_, pi, ci);
}

void UbConnLite::MemorySetAndCopy(u8* va, u32 sqeSize, void* sqe)
{
    auto ret = memset_s(va, sqeSize, 0, sqeSize);
    if (UNLIKELY(ret != 0)) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] memset fail, ret = %d", __func__, ret));
    }
    ret = memcpy_sp(va, sqeSize, sqe, sqeSize);
    if (UNLIKELY(ret != 0)) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] not support this op type yet.", __func__));
    }
}

void UbConnLite::Read(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, const StreamLite& stream,
    ConnLiteOperationOut& out)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    ProcessSlices(
        loc, rmt, maxReadSize,
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_READ, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_READ, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        });

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::ReadReduce(
    ReduceIn reduceIn, const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const StreamLite& stream,
    const SqeConfigLite& cfg, ConnLiteOperationOut& out)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    ProcessSlices(
        loc, rmt, maxReadSize,
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_READ, slicePos);
            FillCommSqeReduceInfo(sqe.comm, reduceIn.reduceOp, reduceIn.dataType);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_READ, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        },
        reduceIn.dataType);

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::Write(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, const StreamLite& stream,
    ConnLiteOperationOut& out)
{
    HCCL_INFO("[UbConnLite::%s] start, loc size = %llu", __func__, loc.GetSize());

    ProcessSlices(
        loc, rmt, maxWriteSize,
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        });

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::InlineWrite(
    const u8* data, u16 size, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, const StreamLite& stream,
    ConnLiteOperationOut& out)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    // 构造sqe
    UdmaSqeWrite sqe{};
    sqe.comm.inlineEn = 1;
    sqe.comm.inlineMsgLen = size;
    FillCommSqe(&(sqe.comm), rmt, cfg, UdmaSqOpcode::UDMA_OPC_WRITE);
    auto ret = memcpy_sp(sqe.u.inlineData.data, SQE_INLINE_DATA_SIZE, data, size);
    if (UNLIKELY(ret != 0)) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] not support this op type yet.", __func__));
    }

    // 写wqe到va
    ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

    // 按需更新wqe tasks
    UpdateWqeTasks(sqe);

    out.pi = pi;
    HCCL_INFO(
        "[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, ConnLiteOperationOut.datasize = %u, conn[%s]", __func__,
        out.pi, out.dataSize, Describe().c_str());
}

void UbConnLite::FillNotifySqe(struct UdmaSqeNotify* sqe, const RmtRmaBufSliceLite& notify, u64 notifyData) const
{
    sqe->notifyTokenId = notify.GetTokenId();
    sqe->notifyTokenValue = notify.GetTokenValue();
    sqe->notifyAddrLow = notify.GetAddr() & ADDR_BIT_LOW;
    sqe->notifyAddrHigh = notify.GetAddr() >> ADDR_BIT_OFFSET;
    sqe->notifyDataLow = notifyData & ADDR_BIT_LOW;
    sqe->notifyDataHigh = notifyData >> ADDR_BIT_OFFSET;
    HCCL_INFO(
        "UbConnLite FillNotifySqe sqe->notifyAddrLow = %u "
        "sqe->notifyAddrHigh = %u, sqe->notifyDataLow = %u, sqe->notifyDataHigh = %u",
        sqe->notifyAddrLow, sqe->notifyAddrHigh, sqe->notifyDataLow, sqe->notifyDataHigh);
}

void UbConnLite::FillLocalSgeSqe(UdmaNormalSge* sqe, const RmaBufSliceLite& loc) const
{
    sqe->length = loc.GetSize();
    sqe->tokenId = loc.GetTokenId();
    sqe->dataAddrLow = loc.GetAddr() & ADDR_BIT_LOW;
    sqe->dataAddrHigh = loc.GetAddr() >> ADDR_BIT_OFFSET;
    HCCL_INFO(
        "UbConnLite FillLocalSgeSqe sqe->length = %u, sqe->dataAddrLow = %u "
        "sqe->dataAddrHigh = %u",
        sqe->length, sqe->dataAddrLow, sqe->dataAddrHigh);
}

void UbConnLite::WriteReduce(
    DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite& loc, const StreamLite& stream,
    const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, ConnLiteOperationOut& out)
{
    HCCL_INFO(
        "[UbConnLite::%s] start, dataType = %u, reduceOp %u, loc.addr = %llu, "
        "rmt.addr = %llu, cfg.cqeEn = %u, out.pi = %u",
        __func__, dataType, reduceOp, loc.GetAddr(), rmt.GetAddr(), cfg.cqeEn, out.pi);

    ProcessSlices(
        loc, rmt, maxWriteSize,
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillCommSqeReduceInfo(sqe.comm, reduceOp, dataType);
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        },
        dataType);

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::WriteWithNotify(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, ConnLiteOperationOut& out,
    const RmtRmaBufSliceLite& notify, const StreamLite& stream, u64 notifyData)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    ProcessSlicesWithNotify(
        loc, rmt, maxWriteSize,
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        },
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWriteWithNotify sqe{};
            FillOneWqeWithNotify(locSlice, rmtSlice, cfg, &sqe, notify, notifyData, WRITE_WITH_NOTIFY_OPCODE, slicePos);
            ProcessOneWqeWithNotify(&sqe, WRITE_WITH_NOTIFY_OPCODE, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        });

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::WriteReduceWithNotify(
    DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt,
    const SqeConfigLite& cfg, const StreamLite& stream, ConnLiteOperationOut& out, const RmtRmaBufSliceLite& notify,
    u64 notifyData)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    ProcessSlicesWithNotify(
        loc, rmt, maxWriteSize,
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillCommSqeReduceInfo(sqe.comm, reduceOp, dataType);
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        },
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWriteWithNotify sqe{};
            FillCommSqeReduceInfo(sqe.comm, reduceOp, dataType);
            FillOneWqeWithNotify(locSlice, rmtSlice, cfg, &sqe, notify, notifyData, WRITE_WITH_NOTIFY_OPCODE, slicePos);
            ProcessOneWqeWithNotify(&sqe, WRITE_WITH_NOTIFY_OPCODE, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        },
        dataType);

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::CustomizeSqeByOneSidedComm(UdmaSqeCommon* sqe, bool isLastWqe) const
{
    /* 表示SQE是否需要上报CQE:为1表示此SQE需要上报CQE，为0表示不需要 */
    sqe->cqe = isLastWqe;

    /* 2’b00:No order，表示当前报文与其他报文无保序要求
       2’b01:Relax Order，表示当前报文与后续的Strong Order报文有保序要求，strong order报文不能超越relax order报文执行。
       2’b10：Strong Order，表示当前报文有保序要求，该报文与前面的Relax Order报文有保序要求。
       2’b11：Reserved。
    */
    sqe->placeOdr = (isLastWqe == true ? 0x02 : 0x01);

    /* ODR[2]表示请求报文在目的端的completion order属性，表示当前报文和前面报文是否存在completion序：
       1’b0 :no order，表示当前报文和前面报文没有completion序要求，报文对应的CQE可以乱序上报。
       1’b1 :表示当前报文和前面报文有completion序要求，报文对应的CQE需要保序上报
    */
    sqe->compOrder = 1;

    /* 表示是否使能fence保序。为1时表示使能，为0时表示不使能。对于send/write/atomic SQE
       当fence为1时需要等待前面所有read和Atomic完成才开始执行，即等待前面发出的read或Atomic接收到所有response
    */
    sqe->fence = (isLastWqe == true ? 0x01 : 0x00);

    HCCL_INFO(
        "UbConnLite CustomizeSqeByOneSidedComm sqe->cqe =%u, sqe->placeOdr = %u sqe->compOrder =%u, sqe->fence = %u",
        sqe->cqe, sqe->placeOdr, sqe->compOrder, sqe->fence);
}

void UbConnLite::FillBatchOneWqe(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, bool isLastWqe, u32 opCode,
    const StreamLite& stream)
{
    (void)stream;
    HCCL_INFO("UbConnLite FillBatchOneWqe start, loc[%s], rmt[%s]", loc.Describe().c_str(), rmt.Describe().c_str());

    u32 sqOffset = pi % sqDepth_;
    pi = pi + 1;
    if (UNLIKELY(pi > sqDepth_)) {
        pi = pi % sqDepth_;
    }

    // 写入wqe数据到out.data
    UdmaSqeWrite sqe{};
    sqe.comm.inlineEn = 0;
    FillCommSqe(&(sqe.comm), rmt, cfg, opCode);
    FillLocalSgeSqe(&(sqe.u.sge), loc);

    if (UNLIKELY(sqe.u.sge.length == 0)) {
        sqe.comm.sgeNum = 0;
    }

    CustomizeSqeByOneSidedComm(&(sqe.comm), isLastWqe);

    HCCL_INFO("UbConnLite BatchWrite cp data to va %llu, pi %u", sqVa_, pi);
    u8* va = ReinterpretAs<u8*>(sqVa_ + sqOffset * SQE_SIZE_64);
    if (dwqeCacheLocked_ == false) {
        auto ret = memcpy_sp(va, SQE_SIZE_64, &sqe, sizeof(UdmaSqeWrite));
        if (UNLIKELY(ret != 0)) {
            HCCL_ERROR("UbConnLite::BatchWrite FillCommSqe memcpy failed, ret=%d", ret);
            THROW<InternalException>(StringFormat("UbConnLite::BatchWrite memcpy_sp failed, ret = %d", ret));
        }
    }
    HCCL_INFO("UbConnLite BatchWrite cp data to va end va(%p)", va);

    // 按需更新wqe tasks
    UpdateWqeTasks(sqe);
}

void UbConnLite::BatchProcessOneSlice(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, u32 maxSliceSize,
    bool isLastSlice, u32 opCode, const StreamLite& stream)
{
    u64 dataSize = loc.GetSize();
    // 按照UDMA能力切分数据
    bool isLastWqe;
    u64 offset = 0;

    // 使用整数除法和取余运算优化循环
    u64 numIterations = dataSize / maxSliceSize;
    u64 remainingSize = dataSize % maxSliceSize;

    for (u64 i = 0; i < numIterations; ++i) {
        isLastWqe = false;
        if ((remainingSize == 0) && (i == numIterations - 1) && isLastSlice) {
            isLastWqe = true;
        }

        // 构造本次wqe的log和rmt RmaBufSilce
        RmaBufSliceLite locTmp(loc.GetAddr() + offset, UB_DMA_MAX_READ_WEITE_SIZE, loc.GetLkey(), loc.GetTokenId());
        RmtRmaBufSliceLite rmtTmp(
            rmt.GetAddr() + offset, UB_DMA_MAX_READ_WEITE_SIZE, rmt.GetRkey(), rmt.GetTokenId(), rmt.GetTokenValue(),
            UINT32_MAX);

        FillBatchOneWqe(locTmp, rmtTmp, cfg, isLastWqe, opCode, stream);

        offset += UB_DMA_MAX_READ_WEITE_SIZE;
    }

    // 处理剩余的数据
    if (remainingSize > 0 && isLastSlice) {
        isLastWqe = true;

        RmaBufSliceLite locTmp(loc.GetAddr() + offset, remainingSize, loc.GetLkey(), loc.GetTokenId());
        RmtRmaBufSliceLite rmtTmp(
            rmt.GetAddr() + offset, remainingSize, rmt.GetRkey(), rmt.GetTokenId(), rmt.GetTokenValue(), UINT32_MAX);
        FillBatchOneWqe(locTmp, rmtTmp, cfg, isLastWqe, opCode, stream);
    }
}

void UbConnLite::BatchCommDataProcess(
    const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const SqeConfigLite& cfg,
    u32 maxSliceSize, u32 opCode, const StreamLite& stream)
{
    u64 siliceSize = loc.size();
    // 按照UDMA能力切分数据, 组装wqe
    for (u64 i = 0; i < siliceSize; i++) {
        BatchProcessOneSlice(loc[i], rmt[i], cfg, maxSliceSize, (i == (siliceSize - 1)), opCode, stream);
    }

    return;
}

void UbConnLite::BatchOneSidedRead(
    const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const SqeConfigLite& cfg,
    const StreamLite& stream, ConnLiteOperationOut& out)
{
    // 按照UDMA能力切分数据, 组装wqe
    BatchCommDataProcess(loc, rmt, cfg, maxReadSize, UdmaSqOpcode::UDMA_OPC_READ, stream);

    // 更新connlite的输出信息
    out.pi = pi;
    HCCL_INFO("UbConnLite BatchRead end, out.pi = %u", out.pi);
}

void UbConnLite::BatchOneSidedWrite(
    const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const SqeConfigLite& cfg,
    const StreamLite& stream, ConnLiteOperationOut& out)
{
    // 按照UDMA能力切分数据, 组装wqe
    BatchCommDataProcess(loc, rmt, cfg, maxWriteSize, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

    // 更新connlite的输出信息
    out.pi = pi;
    HCCL_INFO("UbConnLite BatchWrite end, out.pi = %u", out.pi);
}

std::string UbConnLite::Describe()
{
    return StringFormat(
        "UbConnLite[dieId=%u, funcId=%u, jettyId=%u, dbAddr=0x%llx, sqVa=0x%llx, sqDepth=%u, "
        "jfcPollMode=%u, tpn=%u, dwqeCacheLocked=%d, locEid=%s, rmtEid=%s,jettyPi=%u, jettyCi=%u]",
        dieId_, funcId_, jettyId_, dbAddr_, sqVa_, sqDepth_, jfcPollMode_, tpn_, dwqeCacheLocked_,
        Bytes2hex(locEid_.raw, sizeof(locEid_.raw)).c_str(), Bytes2hex(rmtEid_.raw, sizeof(rmtEid_.raw)).c_str(), pi,
        ci);
}

constexpr uint32_t UB_WQE_NUM_PER_SQE = 4; // URMA约束每个SQE包含4个WQEBB
UbConnLite::UbConnLite(const UbConnLiteParam& liteParam)
{
    HCCL_INFO("[UbConnLite::%s] liteParam[%s]", __func__, liteParam.Describe().c_str());
    dieId_ = liteParam.dieId;
    funcId_ = liteParam.funcId;
    jettyId_ = liteParam.jettyId;
    dbAddr_ = liteParam.dbAddr;
    sqVa_ = liteParam.sqVa;
    // host侧创建jetty指定的sqDepth为sqeBBNum,device侧需要感知wqebbnum,URMA约束每个SQE包含4个WQEBB
    sqDepth_ = liteParam.sqDepth * UB_WQE_NUM_PER_SQE;
    dwqeCacheLocked_ = liteParam.dwqeCacheLocked;
    jfcPollMode_ = liteParam.jfcPollMode;
    tpn_ = liteParam.tpn;

    maxReadSize = liteParam.maxReadSize;
    maxWriteSize = liteParam.maxWriteSize;

    (void)memcpy_sp(rmtEid_.raw, URMA_EID_LEN, liteParam.rmtEid.raw, URMA_EID_LEN);
    (void)memcpy_sp(locEid_.raw, URMA_EID_LEN, liteParam.locEid.raw, URMA_EID_LEN);
    rmtReverseEid_ = rmtEid_.Reversed();
    jettyHandle_ = liteParam.jettyHandle;
    HCCL_INFO("%s", Describe().c_str());
}

UbConnLite::UbConnLite(const UbJettyLiteId& id, const UbJettyLiteAttr& attr, const Eid& rmtInfo)
    : RmaConnLite(id, attr, rmtInfo),
      maxReadSize(UB_DMA_MAX_READ_WEITE_SIZE),
      maxWriteSize(UB_DMA_MAX_READ_WEITE_SIZE)
{
    rmtReverseEid_ = rmtEid_.Reversed();
}

std::string UbConnLiteParam::Describe() const
{
    return StringFormat(
        "UbConnLiteParam[dieId=%u, funcId=%u, jettyId=%u, dbAddr=0x%llx, sqVa=0x%llx, sqDepth=%u, "
        "jfcPollMode=%u, tpn=%u, dwqeCacheLocked=%d, sqCiAddr=0x%llx, rmtEid=%s, localEid=%s, "
        "maxReadSize=%u, maxWriteSize=%u, jettyHandle=%llu]",
        dieId, funcId, jettyId, dbAddr, sqVa, sqDepth, jfcPollMode, tpn, dwqeCacheLocked, sqCiAddr,
        Bytes2hex(rmtEid.raw, sizeof(rmtEid.raw)).c_str(), Bytes2hex(locEid.raw, sizeof(locEid.raw)).c_str(),
        maxReadSize, maxWriteSize, jettyHandle);
}

UbConnLiteParam::UbConnLiteParam(std::vector<char>& uniqueId)
{
    BinaryStream binaryStream(uniqueId);
    binaryStream >> dieId;
    binaryStream >> funcId;
    binaryStream >> jettyId;

    binaryStream >> jfcPollMode;
    binaryStream >> dwqeCacheLocked;
    binaryStream >> dbAddr;
    binaryStream >> sqCiAddr;
    binaryStream >> sqVa;
    binaryStream >> sqDepth;
    binaryStream >> tpn;
    binaryStream >> rmtEid.raw;
    binaryStream >> locEid.raw;
    binaryStream >> maxReadSize;
    binaryStream >> maxWriteSize;
    binaryStream >> jettyHandle;

    static auto lastPrintTime = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPrintTime).count();
    if (UNLIKELY(duration >= MAX_LOG_TIMEOUT_MS)) {
        HCCL_INFO("%s", Describe().c_str());
        lastPrintTime = now;
    }
    HCCL_INFO(
        "[UbConnLiteParam::%s] locEid[%s], rmtEid[%s]", __func__, locEid.Describe().c_str(), rmtEid.Describe().c_str());
}

} // namespace Hccl

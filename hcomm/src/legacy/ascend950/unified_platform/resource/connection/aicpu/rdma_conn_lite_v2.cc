/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rdma_conn_lite_v2.h"
#include "rdma_vendor_1825_ops.h"

namespace Hccl {
constexpr u64 RDMA_DMA_MAX_SIZE = 0x80000000; // Byte, RDMA一次传输的最大size

void RdmaConnLiteV2::ParseSqContext(std::vector<char>& data)
{
    BinaryStream binaryStream(data);
    binaryStream >> sqContext.qpn;
    binaryStream >> sqContext.sqVa;
    binaryStream >> sqContext.wqeSize;
    binaryStream >> sqContext.depth;
    binaryStream >> sqContext.headAddr;
    binaryStream >> sqContext.tailAddr;
    binaryStream >> sqContext.dbHwVa;
    binaryStream >> sqContext.dbSwVa;
    binaryStream >> sqContext.sl;
    binaryStream >> sqContext.dbVendorSpecified;
}

void RdmaConnLiteV2::ParseCqContext(std::vector<char>& data)
{
    BinaryStream binaryStream(data);
    binaryStream >> cqContext.cqn;
    binaryStream >> cqContext.cqVa;
    binaryStream >> cqContext.cqeSize;
    binaryStream >> cqContext.cqDepth;
    binaryStream >> cqContext.headAddr;
    binaryStream >> cqContext.tailAddr;
    binaryStream >> cqContext.dbSwVa;
    binaryStream >> cqContext.dbVendorSpecified;
}

RdmaConnLiteV2::RdmaConnLiteV2(std::vector<char>& uniqueId) : RmaConnLite()
{
    BinaryStream binaryStream(uniqueId);
    binaryStream >> dmaMode_;

    std::vector<char> sqUniqueId;
    binaryStream >> sqUniqueId;
    ParseSqContext(sqUniqueId);

    std::vector<char> cqUniqueId;
    binaryStream >> cqUniqueId;
    ParseCqContext(cqUniqueId);

    qpVa_ = sqContext.sqVa;
    sqVa_ = sqContext.sqVa;
    sqDepth_ = sqContext.depth;

    // 确定厂商
    GetVendorOps();
}

RdmaConnLiteV2::~RdmaConnLiteV2() {}

std::string RdmaConnLiteV2::Describe()
{
    uint8_t mtuShift
        = static_cast<uint8_t>((sqContext.dbVendorSpecified >> UB_DB_VENDOR_MTUSHIFT_SHIFT) & UB_DB_VENDOR_FIELD_MASK);
    uint8_t cos
        = static_cast<uint8_t>((sqContext.dbVendorSpecified >> UB_DB_VENDOR_COS_SHIFT) & UB_DB_VENDOR_FIELD_MASK);

    return StringFormat(
        "RdmaConnLiteV2[QPN=%u, SQ_VA=0x%llx, WQE_SIZE=%u, SQ_DEPTH=%u, SQ_HEAD_ADDR=0x%llx, SQ_TAIL_ADDR=0x%llx, "
        "SL=%u, DB_HW_VA=0x%llx, DB_SW_VA=0x%llx, MTU_SHIFT=%u, COS=%u, "
        "DB_VENDOR_SPECIFIED=0x%llx, CQN=%u, CQ_VA=0x%llx, CQE_SIZE=%u, CQ_DEPTH=%u, "
        "CQ_HEAD_ADDR=0x%llx, CQ_TAIL_ADDR=0x%llx, DB_HW_VA=0x%llx, DB_SW_VA=0x%llx, "
        "CQ_DB_VENDOR_SPECIFIED=0x%llx]",
        sqContext.qpn, sqContext.sqVa, sqContext.wqeSize, sqContext.depth, sqContext.headAddr, sqContext.tailAddr,
        sqContext.sl, sqContext.dbHwVa, sqContext.dbSwVa, mtuShift, cos, sqContext.dbVendorSpecified, cqContext.cqn,
        cqContext.cqVa, cqContext.cqeSize, cqContext.cqDepth, cqContext.headAddr, cqContext.tailAddr, cqContext.dbHwVa,
        cqContext.dbSwVa, cqContext.dbVendorSpecified);
}

void RdmaConnLiteV2::GetVendorOps()
{
    if (rdmaOps_ != nullptr) {
        return;
    }
    switch (dmaMode_) {
        case 0: { // [PCIe] QBUF_DMA_MODE_DEFAULT
            HCCL_INFO("[RdmaConnLiteV2::%s] Now Aicpu NDA doesn't support PCIE !", __func__);
            rdmaOps_ = nullptr;
            break;
        }
        case 1: { // [UB] QBUF_DMA_MODE_INDEP_UB
            rdmaOps_ = std::make_unique<Rdma1825Ops>(&sqContext, &cqContext);
            break;
        }
        default: {
            HCCL_INFO("[RdmaConnLiteV2::%s] Now dmaMode is invalid !", __func__);
            rdmaOps_ = nullptr;
            break;
        }
    }
}

// 检查Ops不能为空
void RdmaConnLiteV2::CheckVendorOp()
{
    if (UNLIKELY(rdmaOps_ == nullptr)) {
        THROW<InternalException>(StringFormat("NDA Op is null. Now dmaMode_ is %d.", dmaMode_));
    }
}

void RdmaConnLiteV2::Read(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, u64& dbAddr, u64& dbValue)
{
    HCCL_INFO("[RdmaConnLiteV2::%s] Read start, loc size = %u", __func__, loc.GetSize());
    CheckVendorOp();

    // 分片操作
    DoSlice(loc, rmt, [this, &cfg](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice) {
        rdmaOps_->Read(locSlice, rmtSlice, cfg);
    });

    // 构造Doorbell并返回
    rdmaOps_->BuildDoorbell(dbAddr, dbValue);

    HCCL_INFO(
        "[RdmaConnLiteV2::%s] Read end, dbAddr = %llu, dbValue = %llu, conn[%s]", __func__, dbAddr, dbValue,
        Describe().c_str());
}

void RdmaConnLiteV2::Write(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, u64& dbAddr, u64& dbValue)
{
    HCCL_INFO("[RdmaConnLiteV2::%s] Write start, loc size = %u", __func__, loc.GetSize());
    CheckVendorOp();

    // 分片操作
    DoSlice(loc, rmt, [this, &cfg](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice) {
        rdmaOps_->Write(locSlice, rmtSlice, cfg);
    });

    // 构造Doorbell并返回
    rdmaOps_->BuildDoorbell(dbAddr, dbValue);

    HCCL_INFO(
        "[RdmaConnLiteV2::%s] Write end, dbAddr = 0x%llx, dbValue = 0x%llx, conn[%s]", __func__, dbAddr, dbValue,
        Describe().c_str());
}

void RdmaConnLiteV2::WriteReduce(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, DataType dataType,
    ReduceOp reduceOp, u64& dbAddr, u64& dbValue)
{
    HCCL_INFO("[RdmaConnLiteV2::%s] WriteReduce start, loc size = %u", __func__, loc.GetSize());
    CheckVendorOp();

    // 分片操作
    DoSlice(
        loc, rmt,
        [this, &cfg, dataType, reduceOp](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice) {
            rdmaOps_->WriteReduce(locSlice, rmtSlice, cfg, dataType, reduceOp);
        });

    // 构造Doorbell并返回
    rdmaOps_->BuildDoorbell(dbAddr, dbValue);

    HCCL_INFO(
        "[RdmaConnLiteV2::%s] WriteReduce end, dbAddr = %llu, dbValue = %llu, conn[%s]", __func__, dbAddr, dbValue,
        Describe().c_str());
}

void RdmaConnLiteV2::WriteWithNotify(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const RmaBufSliceLite& locNotify,
    const RmtRmaBufSliceLite& notify, const SqeConfigLite& cfg, u64& dbAddr, u64& dbValue)
{
    HCCL_INFO("[RdmaConnLiteV2::%s] WriteWithNotify start, loc size = %u", __func__, loc.GetSize());
    CheckVendorOp();

    // 分片操作
    DoSlice(loc, rmt, [this, &cfg](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice) {
        rdmaOps_->Write(locSlice, rmtSlice, cfg);
    });

    // 补充一个notify操作
    rdmaOps_->Write(locNotify, notify, cfg);

    // 构造Doorbell并返回
    rdmaOps_->BuildDoorbell(dbAddr, dbValue);

    HCCL_INFO(
        "[RdmaConnLiteV2::%s] WriteWithNotify end, dbAddr = %llu, dbValue = %llu, conn[%s]", __func__, dbAddr, dbValue,
        Describe().c_str());
}

void RdmaConnLiteV2::WriteReduceWithNotify(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const RmaBufSliceLite& locNotify,
    const RmtRmaBufSliceLite& notify, const SqeConfigLite& cfg, DataType dataType, ReduceOp reduceOp, u64& dbAddr,
    u64& dbValue)
{
    HCCL_INFO("[RdmaConnLiteV2::%s] WriteReduceWithNotify start, loc size = %u", __func__, loc.GetSize());
    CheckVendorOp();

    // 分片操作
    DoSlice(
        loc, rmt,
        [this, &cfg, dataType, reduceOp](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice) {
            rdmaOps_->WriteReduce(locSlice, rmtSlice, cfg, dataType, reduceOp);
        });

    // 补充一个notify操作
    rdmaOps_->Write(locNotify, notify, cfg);

    // 构造Doorbell并返回
    rdmaOps_->BuildDoorbell(dbAddr, dbValue);

    HCCL_INFO(
        "[RdmaConnLiteV2::%s] WriteReduceWithNotify end, dbAddr = %llu, dbValue = %llu, conn[%s]", __func__, dbAddr,
        dbValue, Describe().c_str());
}

void RdmaConnLiteV2::DoSlice(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt,
    const std::function<void(const RmaBufSliceLite&, const RmtRmaBufSliceLite&)>& op) const
{
    const u64 len = loc.GetSize();
    const u32 fullSlices = static_cast<u32>(len / RDMA_DMA_MAX_SIZE);
    const u32 remain = static_cast<u32>(len % RDMA_DMA_MAX_SIZE);
    const u32 totalSlices = fullSlices + (remain > 0 ? 1 : 0);

    for (u32 sliceIdx = 0; sliceIdx < totalSlices; sliceIdx++) {
        const u64 offset = static_cast<u64>(sliceIdx) * RDMA_DMA_MAX_SIZE;
        const u64 localAddr = loc.GetAddr() + offset;
        const u64 remoteAddr = rmt.GetAddr() + offset;
        const u32 sliceSize = (sliceIdx == totalSlices - 1 && remain > 0) ? remain : RDMA_DMA_MAX_SIZE;

        RmaBufSliceLite locSlice(localAddr, sliceSize, loc.GetLkey(), 0);
        RmtRmaBufSliceLite rmtSlice(remoteAddr, sliceSize, rmt.GetRkey(), 0, 0, UINT32_MAX);

        HCCL_INFO(
            "[RdmaConnLiteV2::%s] Slice[%u]: offset=0x%llx, localAddr=0x%llx, "
            "remoteAddr=0x%llx, size=0x%x",
            __func__, sliceIdx, offset, localAddr, remoteAddr, sliceSize);

        op(locSlice, rmtSlice);
    }
}

HcclResult
RdmaConnLiteV2::PollCq(int32_t numEntries, int32_t timeOut, std::vector<int32_t>& errList, u64& dbAddr, u64& dbValue)
{
    HcclResult ret = HCCL_SUCCESS;

    // Poll numEntries个Cqe, 只返回异常的status
    ret = rdmaOps_->PollCq(numEntries, timeOut, errList);

    // Build And Ring Cq Soft DB
    rdmaOps_->BuildCqDoorbell(dbAddr, dbValue);

    // 返回前构造cq DB
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[RdmaConnLiteV2::%s] PollCq error, error code: %u", __func__, ret);
        return ret;
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl

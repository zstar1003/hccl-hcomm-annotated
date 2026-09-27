/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_UB_CONN_LITE_H_
#define HCCLV2_UB_CONN_LITE_H_

#include <queue>
#include "data_type.h"
#include "reduce_op.h"
#include "rma_buf_slice_lite.h"
#include "rmt_rma_buf_slice_lite.h"
#include "rma_conn_lite.h"
#include "udma_data_struct.h"
#include "kernel_param_lite.h"
#include "stream_lite.h"

namespace Hccl {

enum class SlicePosition { ONLY = 0, FIRST = 1, MIDDLE = 2, LAST = 3 };
struct UbConnLiteParam {
    u32 dieId;
    u32 funcId;
    u32 jettyId;

    u64 dbAddr;
    u64 sqVa;
    u32 sqDepth;
    u32 tpn;
    bool dwqeCacheLocked;
    u32 jfcPollMode; // 0代表STARS POLL， 1代表软件Poll
    u64 sqCiAddr;    // 预留给 软件poll CQ 的Jetty使用

    Eid rmtEid;
    Eid locEid;
    uint64_t jettyHandle{0};

    u32 maxReadSize;
    u32 maxWriteSize;

    UbConnLiteParam(std::vector<char>& uniqueId);

    std::string Describe() const;
};

class UbConnLite : public RmaConnLite {
public:
    UbConnLite(const UbJettyLiteId& id, const UbJettyLiteAttr& attr, const Eid& rmtInfo);

    explicit UbConnLite(const UbConnLiteParam& liteParam);

    std::string Describe() final;

    void FillCommSqe(
        UdmaSqeCommon* sqe, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, u32 opCode,
        SlicePosition slicePos = SlicePosition::ONLY);

    void Read(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, const StreamLite& stream,
        ConnLiteOperationOut& out) override;

    void ReadReduce(
        ReduceIn reduceIn, const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const StreamLite& stream,
        const SqeConfigLite& cfg, ConnLiteOperationOut& out) override;

    void Write(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, const StreamLite& stream,
        ConnLiteOperationOut& out) override;

    void InlineWrite(
        const u8* data, u16 size, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, const StreamLite& stream,
        ConnLiteOperationOut& out) override;

    void WriteReduce(
        DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite& loc, const StreamLite& stream,
        const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, ConnLiteOperationOut& out) override;

    void FillNotifySqe(struct UdmaSqeNotify* sqe, const RmtRmaBufSliceLite& notify, u64 notifyData) const;
    void FillLocalSgeSqe(UdmaNormalSge* sqe, const RmaBufSliceLite& loc) const;

    void WriteWithNotify(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, ConnLiteOperationOut& out,
        const RmtRmaBufSliceLite& notify, const StreamLite& stream, u64 notifyData) override;

    void WriteReduceWithNotify(
        DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt,
        const SqeConfigLite& cfg, const StreamLite& stream, ConnLiteOperationOut& out, const RmtRmaBufSliceLite& notify,
        u64 notifyData) override;

    void CustomizeSqeByOneSidedComm(UdmaSqeCommon* sqe, bool isLastWqe) const;

    void FillBatchOneWqe(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, bool isLastWqe, u32 opCode,
        const StreamLite& stream);

    void BatchProcessOneSlice(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, u32 maxSliceSize,
        bool isLastSlice, u32 opCode, const StreamLite& stream);

    void BatchCommDataProcess(
        const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const SqeConfigLite& cfg,
        u32 maxSliceSize, u32 opCode, const StreamLite& stream);

    void BatchOneSidedRead(
        const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const SqeConfigLite& cfg,
        const StreamLite& stream, ConnLiteOperationOut& out) override;
    void BatchOneSidedWrite(
        const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const SqeConfigLite& cfg,
        const StreamLite& stream, ConnLiteOperationOut& out) override;

    // 用于aicpu task cache或者WQE打印
    inline HcclResult EnableWqeTasks()
    {
        isTrackWqeTasks_ = true;
        wqeTasks_.clear();
        return HCCL_SUCCESS;
    }
    inline HcclResult DisableWqeTasks()
    {
        isTrackWqeTasks_ = false;
        wqeTasks_.clear();
        return HCCL_SUCCESS;
    }
    inline const std::vector<WqeTask>& GetWqeTasks() const { return wqeTasks_; }

    // 用于aicpu task cache下发刷新后的WQE
    void LaunchOneWqe(UdmaSqeWrite* sqe, UdmaSqOpcode opCode);
    void LaunchOneWqeWithNotify(UdmaSqeWriteWithNotify* sqe, u32 opCode);

    // 用于aicpu task cache更新DbSqe
    uint16_t GetPi() const { return pi; }

private:
    u16 pi{0};
    u16 ci{0};
    u32 piDetourCount{0};
    u32 ciDetourCount{0};
    u32 maxReadSize{0};
    u32 maxWriteSize{0};
    void ProcessSlices(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, u32 maxSliceSize,
        std::function<void(const RmaBufSliceLite&, const RmtRmaBufSliceLite&, SlicePosition)> processOneSlice,
        DataType dataType = DataType::INVALID) const;
    void ProcessSlicesWithNotify(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, u32 maxSliceSize,
        std::function<void(const RmaBufSliceLite&, const RmtRmaBufSliceLite&, SlicePosition)> processOneSlice,
        std::function<void(const RmaBufSliceLite&, const RmtRmaBufSliceLite&, SlicePosition)> processOneSliceWithNotify,
        DataType dataType = DataType::INVALID) const;
    inline void ProcessOneWqe(UdmaSqeWrite* sqe, UdmaSqOpcode opCode, const StreamLite& stream)
    {
        (void)stream;
        LaunchOneWqe(sqe, opCode);
    }
    void FillOneWqeWithNotify(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg,
        UdmaSqeWriteWithNotify* sqe, const RmtRmaBufSliceLite& notify, u64 notifyData, u32 opCode,
        SlicePosition slicePos);
    inline void ProcessOneWqeWithNotify(UdmaSqeWriteWithNotify* sqe, u32 opCode, const StreamLite& stream)
    {
        (void)stream;
        LaunchOneWqeWithNotify(sqe, opCode);
    }
    void FillCommSqeReduceInfo(UdmaSqeCommon& sqeComm, ReduceOp reduceOp, DataType dataType, u32 udfType = 0) const;
    void FillOneSqeWrite(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, UdmaSqeWrite* sqe,
        UdmaSqOpcode opCode, SlicePosition slicePos);
    void MemorySetAndCopy(u8* va, u32 sqeSize, void* sqe);

    // 用于aicpu task cache或者WQE打印
    bool isTrackWqeTasks_{false};
    std::vector<WqeTask> wqeTasks_;
    inline void UpdateWqeTasks(UdmaSqeWrite& sqe)
    {
        if (isTrackWqeTasks_) {
            wqeTasks_.emplace_back(sqe);
        }
    }
    inline void UpdateWqeTasks(UdmaSqeWriteWithNotify& sqe)
    {
        if (isTrackWqeTasks_) {
            wqeTasks_.emplace_back(sqe);
        }
    }
};
} // namespace Hccl

#endif // HCCLV2_UB_CONN_LITE_H_

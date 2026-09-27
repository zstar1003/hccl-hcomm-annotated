/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RMA_CONN_LITE_H
#define HCCLV2_RMA_CONN_LITE_H
#include <memory>
#include "rma_buf_slice_lite.h"
#include "rmt_rma_buf_slice_lite.h"
#include "log.h"
#include "reduce_in.h"
#include "data_type.h"
#include "reduce_op.h"
#include "ip_address.h"
#include "ub_jetty_lite.h"
#include "kernel_param_lite.h"
#include "stream_lite.h"
namespace Hccl {
MAKE_ENUM(RmaConnLiteType, P2P, RDMA, UB, CCU) // 需要和RmaConnType一一对应

struct SqeConfigLite {
    SqeConfigLite() : placeOdr(1), compOrder(1), fence(1) {}
    bool cqeEn{true};
    bool userConfig{false};
    u8 placeOdr : 2;
    u8 compOrder : 1;
    u8 fence : 1;
};

struct ConnLiteOperationOut {
    u16 pi{0};
    u8* data{};
    u8 dataSize{0};
};

class RmaConnLite {
public:
    RmaConnLite() = default;

    RmaConnLite(const UbJettyLiteId& id, const UbJettyLiteAttr& attr, const Eid& rmtEid);

    explicit RmaConnLite(const u64 qpVa);

    virtual ~RmaConnLite() = default;

    static std::unique_ptr<RmaConnLite> Create(std::vector<char>& uniqueId);

    UbJettyLiteId GetUbJettyLiteId() const;

    UbJettyLiteAttr GetUbJettyLiteAttr() const;

    Eid GetRmtEid() const;
    Eid GetLocEid() const;

    u32 GetQpVa() const;
    u64 GetJettyHandle() const;
    u32 GetJettyId() const;
    u32 GetTpn() const;

    virtual std::string Describe();

    virtual void Read(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg,
        [[maybe_unused]] const StreamLite& stream, ConnLiteOperationOut& out)
    {
        HCCL_INFO(
            "RmaConnLite read start. loc.addr = %llx, rmt.addr = %llx, cfg.cqeEn = %u, out.pi = %u", loc.GetAddr(),
            rmt.GetAddr(), cfg.cqeEn, out.pi);
    }

    virtual void ReadReduce(
        ReduceIn reduceIn, const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt,
        [[maybe_unused]] const StreamLite& stream, const SqeConfigLite& cfg, ConnLiteOperationOut& out)
    {
        HCCL_INFO(
            "RmaConnLite ReadReduce start. dataType = %u, reduceOp %u, loc.addr = %llx, "
            "rmt.addr = %llx, cfg.cqeEn = %u, out.pi = %u",
            reduceIn.dataType, reduceIn.reduceOp, loc.GetAddr(), rmt.GetAddr(), cfg.cqeEn, out.pi);
    }

    virtual void Write(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg,
        [[maybe_unused]] const StreamLite& stream, ConnLiteOperationOut& out)
    {
        HCCL_INFO(
            "RmaConnLite write start. loc.addr = %llx, rmt.addr = %llx, cfg.cqeEn = %u, out.pi = %u", loc.GetAddr(),
            rmt.GetAddr(), cfg.cqeEn, out.pi);
    }

    virtual void InlineWrite(
        const u8* data, u16 size, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg,
        [[maybe_unused]] const StreamLite& stream, ConnLiteOperationOut& out)
    {
        HCCL_INFO(
            "RmaConnLite InlineWrite start. data = %p, size = %hu, rmt.addr = %llx, cfg.cqeEn = %u, out.pi = %u", data,
            size, rmt.GetAddr(), cfg.cqeEn, out.pi);
    }

    virtual void WriteReduce(
        DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite& loc, [[maybe_unused]] const StreamLite& stream,
        const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, ConnLiteOperationOut& out)
    {
        HCCL_INFO(
            "RmaConnLite WriteReduce start. dataType = %u, reduceOp %u, loc.addr = %llx, "
            "rmt.addr = %llx, cfg.cqeEn = %u, out.pi = %u",
            dataType, reduceOp, loc.GetAddr(), rmt.GetAddr(), cfg.cqeEn, out.pi);
    }

    virtual void WriteWithNotify(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, ConnLiteOperationOut& out,
        const RmtRmaBufSliceLite& notify, [[maybe_unused]] const StreamLite& stream, u64 notifyData)
    {
        HCCL_INFO(
            "RmaConnLite WriteWithNotify start. loc.addr = %llx, rmt.addr = %llx, cfg.cqeEn = %u, "
            "out.pi = %u, notify.addr = %llx, notifyData = %u",
            loc.GetAddr(), rmt.GetAddr(), cfg.cqeEn, out.pi, notify.GetAddr(), notifyData);
    }

    virtual void WriteReduceWithNotify(
        DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt,
        const SqeConfigLite& cfg, [[maybe_unused]] const StreamLite& stream, ConnLiteOperationOut& out,
        const RmtRmaBufSliceLite& notify, u64 notifyData)
    {
        HCCL_INFO(
            "RmaConnLite WriteReduceWithNotify start. dataType = %d, reduceOp %d, loc.addr = %llx, "
            "rmt.addr = %llx, cfg.cqeEn = %u, out.pi = %u, notify.addr = %llx, notifyData = %llu",
            dataType, reduceOp, loc.GetAddr(), rmt.GetAddr(), cfg.cqeEn, out.pi, notify.GetAddr(), notifyData);
    }

    virtual void BatchOneSidedRead(
        const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const SqeConfigLite& cfg,
        [[maybe_unused]] const StreamLite& stream, ConnLiteOperationOut& out)
    {
        HCCL_INFO(
            "RmaConnLite BatchOneSidedRead start. loc.size = %llu, rmt.size = %llu, cfg.cqeEn = %u, out.pi = %u",
            loc.size(), rmt.size(), cfg.cqeEn, out.pi);
    }

    virtual void BatchOneSidedWrite(
        const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const SqeConfigLite& cfg,
        [[maybe_unused]] const StreamLite& stream, ConnLiteOperationOut& out)
    {
        HCCL_INFO(
            "RmaConnLite BatchOneSidedWrite start. loc.size = %llu, rmt.size = %llu, cfg.cqeEn = %u, out.pi = %u",
            loc.size(), rmt.size(), cfg.cqeEn, out.pi);
    }

protected:
    u32 qpVa_{0};

    u32 dieId_{0};
    u32 funcId_{0};
    u32 jettyId_{0};
    u64 dbAddr_{0};
    u64 sqVa_{0};
    u32 sqDepth_{0};
    bool dwqeCacheLocked_{false}; // direct WQE cache Lock
    u32 jfcPollMode_{0};          // 0代表STARS POLL， 1代表软件Poll
    u32 tpn_{0};

    Eid rmtEid_;
    Eid locEid_;
    Eid rmtReverseEid_{}; // 反序Eid，仅用于传递给硬件
    uint64_t jettyHandle_{0};
};

} // namespace Hccl
#endif

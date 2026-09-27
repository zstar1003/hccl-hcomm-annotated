/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_V2_AICPU_SQE_V82_H
#define HCCL_V2_AICPU_SQE_V82_H

#include <cstdint>
#include <memory>
#include "types.h"
#include "log.h"
#include "sqe_v82.h"
#include "hccl_sqe.h"

namespace Hccl {

class HcclUBDmaDBSqe : public HcclSqe {
public:
    HcclUBDmaDBSqe();

    void Config(u16 streamId, u16 taskId, u16 jettyid, u8 funcId, u16 piValue, u16 dieId);

    u64 GetSqe() override;

private:
    std::unique_ptr<Rt91095StarsUbdmaDBmodeSqe> sqe;
};

class HcclUBNotifyWaitSqe : public HcclSqe {
public:
    HcclUBNotifyWaitSqe();

    void Config(u16 streamId, u16 taskId, u64 notifyId);

    u64 GetSqe() override;

private:
    std::unique_ptr<Rt91095StarsNotifySqe> sqe;
};

class HcclUBNotifyRecordSqe : public HcclSqe {
public:
    HcclUBNotifyRecordSqe();
    void Config(u16 streamId, u16 taskId, u64 notifyId);
    u64 GetSqe() override;

private:
    std::unique_ptr<Rt91095StarsNotifySqe> sqe;
};

class HcclUBCntNotifyNto1RecordSqe : public HcclSqe {
public:
    HcclUBCntNotifyNto1RecordSqe();

    void Config(u16 streamId, u16 taskId, u64 notifyId, u32 cntValue);

    u64 GetSqe() override;

private:
    std::unique_ptr<Rt91095StarsNotifySqe> sqe;
};

class HcclUBCntNotify1toNWaitSqe : public HcclSqe {
public:
    HcclUBCntNotify1toNWaitSqe();

    void Config(u16 streamId, u16 taskId, u64 notifyId, u32 cntValue);

    u64 GetSqe() override;

private:
    std::unique_ptr<Rt91095StarsNotifySqe> sqe;
};

class HcclUBCntNotifyNto1WaitSqe : public HcclSqe {
public:
    HcclUBCntNotifyNto1WaitSqe();

    void Config(u16 streamId, u16 taskId, u64 notifyId, u32 cntValue);

    u64 GetSqe() override;

private:
    std::unique_ptr<Rt91095StarsNotifySqe> sqe;
};

class HcclUBCntNotify1toNRecordSqe : public HcclSqe {
public:
    HcclUBCntNotify1toNRecordSqe();

    void Config(u16 streamId, u16 taskId, u64 notifyId, u32 cntValue);

    u64 GetSqe() override;

private:
    std::unique_ptr<Rt91095StarsNotifySqe> sqe;
};

class HcclUBMemcpySqe : public HcclSqe {
public:
    HcclUBMemcpySqe();

    void Config(
        u16 streamId, u16 taskId, RtDataType rtDataType, RtReduceKind rtReduceOp, u64 count, const u64* src,
        const u64* dst, u32 partId);

    u64 GetSqe() override;

private:
    u8 GetUBOpCode(u32 copyKind, u8 copyDataType) const;

    u8 ConvertToMemcpyDataType(u8 copyDataType) const;

    u8 ConvertToMemcpyOpType(u32 copyKind) const;

    std::unique_ptr<Rt91095StarsMemcpySqe> sqe;
};

} // namespace Hccl

#endif

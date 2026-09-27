/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_AICPU_SQEMGR_SQE_H
#define HCCLV2_AICPU_SQEMGR_SQE_H

#include <cstdint>
#include <memory>
#include "types.h"
#include "log.h"
#include "sqe.h"

namespace Hccl {

u32 GetAddrLow(u64 addr);
u32 GetAddrHigh(u64 addr);

class HcclSqe {
public:
    virtual u64 GetSqe() = 0;
    virtual ~HcclSqe() = default;
};

class HcclNotifyWaitSqe : public HcclSqe {
public:
    HcclNotifyWaitSqe();

    void Config(u16 streamId, u16 taskId, u64 notifyId);

    u64 GetSqe() override;

private:
    std::unique_ptr<RtStarsNotifySqe> sqe;
};

class HcclNotifyRecordSqe : public HcclSqe {
public:
    HcclNotifyRecordSqe();
    void Config(u16 streamId, u16 taskId, u64 notifyId);
    u64 GetSqe() override;

private:
    std::unique_ptr<RtStarsNotifySqe> sqe;
};

class HcclWriteValueSqe : public HcclSqe {
public:
    HcclWriteValueSqe();

    void Config(u16 streamId, u16 taskId, u64 notifyWRAddr);

    u64 GetSqe() override;

private:
    std::unique_ptr<RtStarsWriteValueSqe> sqe;
};

class HcclSdmaSqe : public HcclSqe {
public:
    HcclSdmaSqe();

    void Config(
        u16 streamId, u16 taskId, const u64 src, u32 length, RtDataType rtDataType, RtReduceKind rtReduceOp,
        const u64 dst, u32 partId);

    u64 GetSqe() override;

private:
    u8 GetSdmaOpCode(u32 copyKind, u8 copyDataType) const;

    u8 ConvertToMemcpyDataType(u8 copyDataType) const;

    u8 ConvertToMemcpyOpType(u32 copyKind) const;

    std::unique_ptr<RtStarsMemcpyAsyncSqe> sqe;
};
} // namespace Hccl

#endif

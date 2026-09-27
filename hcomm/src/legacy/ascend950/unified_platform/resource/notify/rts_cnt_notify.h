/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RTS_CNT_NOTIFY_H
#define HCCLV2_RTS_CNT_NOTIFY_H

#include <memory>
#include <string>
#include "stream.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "cnt_nto1_notify_lite.h"

namespace Hccl {

class BaseTask;
class RtsCntNotify {
public:
    RtsCntNotify();
    ~RtsCntNotify();
    std::unique_ptr<BaseTask> PostBits(u32 bitValue);
    std::unique_ptr<BaseTask> WaitValue(u32 value);
    void PostBits(u32 bitValue, const Stream& stream) const;
    void WaitValue(u32 value, u32 timeout, const aclrtStream& rtStream) const;
    void WaitValue(u32 value, u32 timeout, const Stream& stream) const;

    std::string Describe() const;

    u32 GetId() const { return id; }

    u64 GetAddr() const { return addr; }

    u32 GetSize() const { return size; }

    std::vector<char> GetUniqueId() const;

private:
    u32 deviceId;
    u32 devPhyId;
    RtCntNotify_t handle{nullptr};
    u32 id{0};
    u64 addr{0};
    u32 size{0};
};

struct UbCntNotifyExchangeData {
    u32 userData;
    u64 addr{0};
    u32 id{0};
    u32 size{0};
    u8 key[HRT_UB_MEM_KEY_MAX_LEN]{0};
    u32 tokenValue{0};
    u32 tokenId{0};
    u32 keySize{0};
};

} // namespace Hccl

#endif // HCCLV2_RTS_CNT_NOTIFY_H

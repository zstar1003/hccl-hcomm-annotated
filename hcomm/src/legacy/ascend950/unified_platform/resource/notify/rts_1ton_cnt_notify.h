/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RTS_1TON_CNT_NOTIFY_H
#define HCCLV2_RTS_1TON_CNT_NOTIFY_H

#include <memory>
#include <string>
#include "stream.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

class BaseTask;
class Rts1ToNCntNotify {
public:
    Rts1ToNCntNotify();
    ~Rts1ToNCntNotify();
    std::unique_ptr<BaseTask> PostValue(u32 value);
    std::unique_ptr<BaseTask> WaitBits(u32 bitValue);
    void PostValue(u32 value, const aclrtStream& rtStream) const;
    void PostValue(u32 value, const Stream& stream) const;
    void WaitBits(u32 bitValue, u32 timeout, const Stream& stream) const;

    std::string Describe() const;

    u32 GetId() const { return id; }

    std::vector<char> GetUniqueId() const;

private:
    u32 deviceId;
    u32 devPhyId;
    RtCntNotify_t handle{nullptr};
    u32 id{0};
};

} // namespace Hccl

#endif // HCCLV2_RTS_1TON_CNT_NOTIFY_H

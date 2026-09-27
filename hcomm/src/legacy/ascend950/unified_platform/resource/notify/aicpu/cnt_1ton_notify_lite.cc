/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cnt_1ton_notify_lite.h"
#include "binary_stream.h"
#include "string_util.h"
#include "log.h"

namespace Hccl {

Cnt1tonNotifyLite::Cnt1tonNotifyLite(std::vector<char>& uniqueId)
{
    BinaryStream binaryStream(uniqueId);
    binaryStream >> notifyId;
    binaryStream >> devPhyId;
    HCCL_INFO("Cnt1tonNotifyLite::Cnt1tonNotifyLite:%s", Describe().c_str());
}

u32 Cnt1tonNotifyLite::GetId() const { return notifyId; }

u32 Cnt1tonNotifyLite::GetDevPhyId() const { return devPhyId; }

std::string Cnt1tonNotifyLite::Describe() const
{
    return StringFormat("Cnt1tonNotifyLite[notifyId=%u, devPhyId=%u]", notifyId, devPhyId);
}

} // namespace Hccl

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "notify_lite.h"
#include "binary_stream.h"
#include "string_util.h"
#include "log.h"
namespace Hccl {

NotifyLite::NotifyLite(std::vector<char>& uniqueId)
{
    BinaryStream binaryStream(uniqueId);
    binaryStream >> notifyId;
    binaryStream >> devPhyId;
    HCCL_INFO("NotifyLite::NotifyLite:%s", Describe().c_str());
}

u32 NotifyLite::GetId() const { return notifyId; }

u32 NotifyLite::GetDevPhyId() const { return devPhyId; }

std::string NotifyLite::Describe() const
{
    return StringFormat("NotifyLite[notifyId=%u, devPhyId=%u]", notifyId, devPhyId);
}

} // namespace Hccl

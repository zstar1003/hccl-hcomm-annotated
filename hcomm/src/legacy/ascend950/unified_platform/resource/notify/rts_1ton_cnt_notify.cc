/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rts_1ton_cnt_notify.h"
#include "exception_util.h"
#include "task.h"

namespace Hccl {

Rts1ToNCntNotify::Rts1ToNCntNotify()
    : deviceId(HrtGetDevice()),
      devPhyId(HrtGetDevicePhyIdByUserDevId(HrtGetDevice())),
      handle(HrtCntNotifyCreate(deviceId)),
      id(HrtGetCntNotifyId(handle))
{}

Rts1ToNCntNotify::~Rts1ToNCntNotify() { DECTOR_TRY_CATCH("Rts1ToNCntNotify", HrtCntNotifyDestroy(handle)); }

std::unique_ptr<BaseTask> Rts1ToNCntNotify::WaitBits(u32 bitValue)
{
    return std::make_unique<TaskWaitBits>(this, bitValue);
}

std::unique_ptr<BaseTask> Rts1ToNCntNotify::PostValue(u32 value)
{
    return std::make_unique<TaskPostValue>(this, value);
}

void Rts1ToNCntNotify::WaitBits(u32 bitValue, u32 timeout, const Stream& stream) const
{
    HrtCntNotifyWaitWithTimeOut(handle, stream.GetPtr(), HrtCntNotifyWaitMode::BITMAP, bitValue, timeout);
}

void Rts1ToNCntNotify::PostValue(u32 value, const aclrtStream& rtStream) const
{
    HrtCntNotifyRecord(handle, rtStream, HrtCntNotifyRecordMode::STORE, value);
}

void Rts1ToNCntNotify::PostValue(u32 value, const Stream& stream) const { PostValue(value, stream.GetPtr()); }

std::string Rts1ToNCntNotify::Describe() const
{
    return StringFormat("Rts1ToNCntNotify[handle=%p, id=%d]", handle, id);
}

std::vector<char> Rts1ToNCntNotify::GetUniqueId() const
{
    BinaryStream binaryStream;
    binaryStream << id;
    binaryStream << devPhyId;
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

} // namespace Hccl

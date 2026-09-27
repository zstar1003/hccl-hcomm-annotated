/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rts_cnt_notify.h"
#include "exception_util.h"
#include "task.h"
#include "dev_capability.h"

namespace Hccl {

RtsCntNotify::RtsCntNotify()
    : deviceId(HrtGetDevice()),
      devPhyId(HrtGetDevicePhyIdByUserDevId(HrtGetDevice())),
      handle(HrtCntNotifyCreate(deviceId)),
      id(HrtGetCntNotifyId(handle))
{
    HrtDevResInfo devResInfo;
    devResInfo.dieId = 0;
    devResInfo.procType = HrtDevResProcType::PROCESS_HCCP;
    devResInfo.resType = HrtDevResType::RES_TYPE_STARS_CNT_NOTIFY_BIT_WR;
    devResInfo.resId = id;
    devResInfo.flag = 0;
    auto resAddrInfo = HrtGetDevResAddress(devResInfo);
    addr = resAddrInfo.address;
    size = DevCapability::GetInstance().GetNotifySize();
}

RtsCntNotify::~RtsCntNotify()
{
    HrtDevResInfo devResInfo;
    devResInfo.dieId = 0;
    devResInfo.procType = HrtDevResProcType::PROCESS_HCCP;
    devResInfo.resType = HrtDevResType::RES_TYPE_STARS_CNT_NOTIFY_BIT_WR;
    devResInfo.resId = id;
    devResInfo.flag = 0;
    DECTOR_TRY_CATCH("RtsCntNotify", HrtReleaseDevResAddress(devResInfo));
    DECTOR_TRY_CATCH("RtsCntNotify", HrtCntNotifyDestroy(handle));
}

std::unique_ptr<BaseTask> RtsCntNotify::PostBits(u32 bitValue)
{
    return std::make_unique<TaskPostBits>(this, bitValue);
}

std::unique_ptr<BaseTask> RtsCntNotify::WaitValue(u32 value) { return std::make_unique<TaskWaitValue>(this, value); }

void RtsCntNotify::PostBits(u32 bitValue, const Stream& stream) const
{
    HrtCntNotifyRecord(handle, stream.GetPtr(), HrtCntNotifyRecordMode::WRITE_BIT, bitValue);
}

void RtsCntNotify::WaitValue(u32 value, u32 timeout, const aclrtStream& rtStream) const
{
    HrtCntNotifyWaitWithTimeOut(handle, rtStream, HrtCntNotifyWaitMode::EQUAL, value, timeout);
}

void RtsCntNotify::WaitValue(u32 value, u32 timeout, const Stream& stream) const
{
    WaitValue(value, timeout, stream.GetPtr());
}

std::string RtsCntNotify::Describe() const
{
    return StringFormat("RtsCntNotify[handle=%p, id=%u, addr=0x%llx, size=%u]", handle, id, addr, size);
}

std::vector<char> RtsCntNotify::GetUniqueId() const
{
    BinaryStream binaryStream;
    binaryStream << id;
    binaryStream << devPhyId;
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

} // namespace Hccl

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task.h"
#include "local_notify_v2.h"
namespace Hccl {

std::string TaskLocalCopy::Describe() const
{
    return StringFormat(
        "Task %s: src[0x%llx] dst[0x%llx] size[0x%llx] kind %s", type.Describe().c_str(), srcAddr, dstAddr, size,
        kind.Describe().c_str());
}

std::string TaskP2pMemcpy::Describe() const
{
    return StringFormat(
        "Task %s: src[0x%llx] dst[0x%llx] size[0x%llx] kind %s", type.Describe().c_str(), srcAddr, dstAddr, size,
        kind.Describe().c_str());
}

std::string TaskRemoteRecord::Describe() const
{
    return StringFormat("Task %s: notify[%s]", type.Describe().c_str(), notify->Describe().c_str());
}

std::string TaskWait::Describe() const
{
    return StringFormat("Task %s: notify[%s]", type.Describe().c_str(), notify->Describe().c_str());
}

std::string TaskLocalRecord::Describe() const
{
    return StringFormat("Task %s: notify[%s]", type.Describe().c_str(), notify->Describe().c_str());
}

std::string TaskWaitValue::Describe() const
{
    return StringFormat("Task %s: notify[%s] value[%u]", type.Describe().c_str(), notify->Describe().c_str(), value);
}

std::string TaskPostBits::Describe() const
{
    return StringFormat(
        "Task %s: notify[%s] bitValue[%u]", type.Describe().c_str(), notify->Describe().c_str(), bitValue);
}

std::string TaskSdmaReduce::Describe() const
{
    return StringFormat(
        "Task %s: src[0x%llx] dst[0x%llx] size[0x%llx] %s, %s", type.Describe().c_str(), srcAddr, dstAddr, size,
        dataType.Describe().c_str(), reduceOp.Describe().c_str());
}

std::string TaskLocalReduce::Describe() const
{
    return StringFormat(
        "Task %s: src1[0x%llx] src2[0x%llx] dst[0x%llx] "
        "size[0x%llx] dataCount[0x%llx] %s, %s",
        type.Describe().c_str(), srcAddr1, srcAddr2, dstAddr, size, GetDataCount(), dataType.Describe().c_str(),
        reduceOp.Describe().c_str());
}

std::string TaskRdmaSend::Describe() const
{
    if (IsTemplateMode()) {
        return StringFormat("Task %s: qpn[%u], wqeIndex[%u]", type.Describe().c_str(), qpn, wqeIndex);
    } else {
        return StringFormat("Task %s: dbIndex[%u], dbInfo[%llu]", type.Describe().c_str(), dbIndex, dbInfo);
    }
}

std::string TaskLocalAddrCopy::Describe() const
{
    return StringFormat(
        "Task %s: src[0x%llx] dst[0x%llx] size[0x%llx]", type.Describe().c_str(), srcAddr, dstAddr, size);
}

std::string TaskUbDbSend::Describe() const
{
    return StringFormat(
        "Task %s: jettyId[0x%llx] funcId[%u] piVal[%u] dieId[%u]", type.Describe().c_str(), jettyId, funcId, piVal,
        dieId);
}

TaskUbDirectSend::TaskUbDirectSend(u32 funcId, u32 dieId, u32 jettyId, u32 dwqeSize, const u8* dwqe)
    : BaseTask(TaskType::UB_DIRECT_SEND),
      funcId(funcId),
      dieId(dieId),
      jettyId(jettyId),
      dwqeSize(dwqeSize)
{
    if (dwqeSize != DWQE_SIZE_64 && dwqeSize != DWQE_SIZE_128) {
        std::string msg = StringFormat("Invalid dwqe size, dwqeSize=[%u]", dwqeSize);
        THROW<InternalException>(msg);
    }

    s32 ret = memcpy_s(this->dwqe, DWQE_MAX_LEN, dwqe, dwqeSize);
    if (ret != 0) {
        std::string msg = StringFormat("TaskUbDirectSend constructor copy dwqe failed, ret=[%d]", ret);
        THROW<InternalException>(msg);
    }
}
std::string TaskUbDirectSend::Describe() const
{
    return StringFormat(
        "Task %s: jettyId[0x%x] funcId[%u] dieId[%u] dwqeSize[%u]", type.Describe().c_str(), jettyId, funcId, dieId,
        dwqeSize);
}

std::string TaskWriteValue::Describe() const
{
    return StringFormat("Task %s: dbAddr[%llx] piVal[%u]", type.Describe().c_str(), dbAddr, piVal);
}

std::string TaskWaitBits::Describe() const
{
    return StringFormat(
        "Task %s: notify[%s] bitValue[%u]", type.Describe().c_str(), notify->Describe().c_str(), bitValue);
}

std::string TaskPostValue::Describe() const
{
    return StringFormat("Task %s: notify[%s] value[%u]", type.Describe().c_str(), notify->Describe().c_str(), value);
}
} // namespace Hccl

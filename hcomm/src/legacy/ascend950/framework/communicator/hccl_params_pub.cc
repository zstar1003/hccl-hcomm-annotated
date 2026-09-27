/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl_params_pub.h>
#include "exception_util.h"
#include "not_support_exception.h"
#include "string_util.h"
namespace Hccl {
std::string CollOpParams::Describe() const
{
    if (descOpMap.find(opType) != descOpMap.end()) {
        return descOpMap.at(opType).operator()(*this);
    } else {
        std::string msg = StringFormat("Does not support this operator=%s, please check.", opType.Describe().c_str());
        MACRO_THROW(NotSupportException, msg);
    }
}

std::string CollOpParams::DescReduceScatter(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, dataType=%s, reduceOp=%s, recvCount=%llu, sendBuf=%p, recvBuf=%p]",
        opType.Describe().c_str(), opParams.dataType.Describe().c_str(), opParams.reduceOp.Describe().c_str(),
        opParams.count, opParams.sendBuf, opParams.recvBuf);
}

std::string CollOpParams::DescReduce(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, dataType=%s, reduceOp=%s, recvCount=%llu, root=%u, sendBuf=%p, recvBuf=%p]",
        opType.Describe().c_str(), opParams.dataType.Describe().c_str(), opParams.reduceOp.Describe().c_str(),
        opParams.count, opParams.root, opParams.sendBuf, opParams.recvBuf);
}

std::string CollOpParams::DescAllreduce(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, dataType=%s, reduceOp=%s, count=%llu, sendBuf=%p, recvBuf=%p]",
        opType.Describe().c_str(), opParams.dataType.Describe().c_str(), opParams.reduceOp.Describe().c_str(),
        opParams.count, opParams.sendBuf, opParams.recvBuf);
}

std::string CollOpParams::DescAllgather(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, dataType=%s, sendCount=%llu, sendBuf=%p, recvBuf=%p]", opType.Describe().c_str(),
        opParams.dataType.Describe().c_str(), opParams.count, opParams.sendBuf, opParams.recvBuf);
}

std::string CollOpParams::DescScatter(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, dataType=%s, sendCount=%llu, sendBuf=%p, recvBuf=%p, root=%u]",
        opType.Describe().c_str(), opParams.dataType.Describe().c_str(), opParams.count, opParams.sendBuf,
        opParams.recvBuf, opParams.root);
}

std::string CollOpParams::DescAlltoall(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, sendCount=%llu, recvCount=%llu, sendType=%s, recvType=%s]", opType.Describe().c_str(),
        opParams.all2AllDataDes.sendCount, opParams.all2AllDataDes.recvCount,
        opParams.all2AllDataDes.sendType.Describe().c_str(), opParams.all2AllDataDes.recvType.Describe().c_str());
}

std::string CollOpParams::DescAlltoallV(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, sendType=%s, recvType=%s]", opType.Describe().c_str(),
        opParams.all2AllVDataDes.sendType.Describe().c_str(), opParams.all2AllVDataDes.recvType.Describe().c_str());
}

std::string CollOpParams::DescAlltoallVC(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, sendType=%s, recvType=%s]", opType.Describe().c_str(),
        opParams.all2AllVCDataDes.sendType.Describe().c_str(), opParams.all2AllVCDataDes.recvType.Describe().c_str());
}

std::string CollOpParams::DescSend(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, dataType=%s, sendBuf=%p]", opType.Describe().c_str(),
        opParams.dataType.Describe().c_str(), opParams.sendBuf);
}

std::string CollOpParams::DescRecv(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, dataType=%s, recvBuf=%p]", opType.Describe().c_str(),
        opParams.dataType.Describe().c_str(), opParams.recvBuf);
}

std::string CollOpParams::DescBroadcast(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, dataType=%s, reduceOp=%s, recvCount=%llu, sendBuf=%p, recvBuf=%p]",
        opType.Describe().c_str(), opParams.dataType.Describe().c_str(), opParams.reduceOp.Describe().c_str(),
        opParams.count, opParams.sendBuf, opParams.recvBuf);
}

std::string CollOpParams::DescBatchSendRecv(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, itemNum=%u", opType.Describe().c_str(), opParams.batchSendRecvDataDes.itemNum);
}

std::string CollOpParams::DescAllGatherV(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, dataType=%s]", opType.Describe().c_str(),
        opParams.vDataDes.dataType.Describe().c_str());
}

std::string CollOpParams::DescReduceScatterV(const CollOpParams& opParams)
{
    return StringFormat(
        "CollOpParams[opType=%s, dataType=%s]", opType.Describe().c_str(),
        opParams.vDataDes.dataType.Describe().c_str());
}
} // namespace Hccl

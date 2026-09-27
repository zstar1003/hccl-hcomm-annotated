/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_virtural.h"
#include "task_logic_info_pub.h"

namespace hccl {

TransportVirtural::TransportVirtural(
    DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
    std::chrono::milliseconds timeout, u32 index)
    : TransportBase(dispatcher, notifyPool, machinePara, timeout),
      currentIndex_(index)
{}

TransportVirtural::~TransportVirtural() { HCCL_DEBUG("~TransportVirtural Enter!"); }

/* 发送ack消息(同步模式) */
HcclResult TransportVirtural::TxAck(Stream& stream)
{
    TaskLogicInfo info(currentIndex_, TaskLogicType::TRANSPORT_TYPE, TaskLogicFuncType::TRANSPORT_TXACK_TYPE);
    stream.PushTaskLogicInfo(info);
    return HCCL_SUCCESS;
}

/* 接收ack消息(同步模式) */
HcclResult TransportVirtural::RxAck(Stream& stream)
{
    TaskLogicInfo info(currentIndex_, TaskLogicType::TRANSPORT_TYPE, TaskLogicFuncType::TRANSPORT_RXACK_TYPE);
    stream.PushTaskLogicInfo(info);
    return HCCL_SUCCESS;
}

HcclResult TransportVirtural::TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream)
{
    TaskLogicInfo info(currentIndex_, TaskLogicType::TRANSPORT_TYPE, TaskLogicFuncType::TRANSPORT_TXASYNC_TYPE, txMems);
    stream.PushTaskLogicInfo(info);
    return HCCL_SUCCESS;
}

HcclResult TransportVirtural::RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream)
{
    TaskLogicInfo info(currentIndex_, TaskLogicType::TRANSPORT_TYPE, TaskLogicFuncType::TRANSPORT_RXASYNC_TYPE, rxMems);
    stream.PushTaskLogicInfo(info);
    return HCCL_SUCCESS;
}

HcclResult TransportVirtural::TxDataSignal(Stream& stream)
{
    TaskLogicInfo info(currentIndex_, TaskLogicType::TRANSPORT_TYPE, TaskLogicFuncType::TRANSPORT_TXDATASIGNAL_TYPE);
    stream.PushTaskLogicInfo(info);
    return HCCL_SUCCESS;
}

HcclResult TransportVirtural::RxDataSignal(Stream& stream)
{
    TaskLogicInfo info(currentIndex_, TaskLogicType::TRANSPORT_TYPE, TaskLogicFuncType::TRANSPORT_RXDATASIGNAL_TYPE);
    stream.PushTaskLogicInfo(info);
    return HCCL_SUCCESS;
}

HcclResult TransportVirtural::TxPrepare([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }
HcclResult TransportVirtural::RxPrepare([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportVirtural::TxData(
    [[maybe_unused]] UserMemType dstMemType, [[maybe_unused]] u64 dstOffset, [[maybe_unused]] const void* src,
    [[maybe_unused]] u64 len, [[maybe_unused]] Stream& stream)
{
    return HCCL_SUCCESS;
}
HcclResult TransportVirtural::RxData(
    [[maybe_unused]] UserMemType srcMemType, [[maybe_unused]] u64 srcOffset, [[maybe_unused]] void* dst,
    [[maybe_unused]] u64 len, [[maybe_unused]] Stream& stream)
{
    return HCCL_SUCCESS;
}

HcclResult TransportVirtural::TxDone([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }
HcclResult TransportVirtural::RxDone([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }
} // namespace hccl

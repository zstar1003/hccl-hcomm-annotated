/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ERROR_MESSAGE_H
#define ERROR_MESSAGE_H
#include "task_param.h"
#include "kernel_param_lite.h"
namespace Hccl {

constexpr u32 OP_TAG_MAX_LENGTH = 288;
constexpr u32 GROUP_NAME_MAX_LENTH = 127; // 最大的group name 长度

struct ErrorMessageReport {
    char tag[OP_TAG_MAX_LENGTH] = {0};
    char group[GROUP_NAME_MAX_LENTH + 1] = {0};
    u32 remoteUserRank = 0;
    s32 streamId = 0;
    u32 taskId = 0;
    u64 notifyId = 0;
    s32 stage = 0;
    u32 notifyValue = 0;
    u32 rankId = 0;
    u32 rankSize = 0;
    TaskParamType taskType = TaskParamType::TASK_SDMA;
    DfxLinkType linkType = DfxLinkType::RESERVED;
    std::size_t size = 0;
    uint64_t count = 0;
    uint64_t dstAddr = 0;
    uint64_t srcAddr = 0;
    uint32_t opIndex = 0;      // 记录index
    uint32_t reduceType = 255; // 255 为 HcclReduceOp::HCCL_REDUCE_RESERVED
    uint8_t dataType = 0;

    Eid locEid{};
    Eid rmtEid{};
    uint64_t taskDstAddr = 0;
    uint64_t taskSrcAddr = 0;

    uint8_t rtCqErrorType = 0;
    uint32_t rtCqErrorCode = 0;
    uint16_t ubCqeStatus = 0;
    uint8_t opType = 0;
    uint64_t jettyHandle = 0;
    uint32_t jettyId = 0;
    uint32_t tpn = 0;

    std::string Describe() const
    {
        return Hccl::StringFormat(
            "tag[%s], group[%s], remoteUserRank[%u], streamId[%d], taskId[%u], notifyId[%llu], "
            "notifyValue[%u], stage[%d], rankId[%u], rankSize[%u], taskType[%s], linkType[%s], "
            "size[%zu], count[%llu], dstAddr[0x%llx], srcAddr[0x%llx], opIndex[%u], reduceType[%u], "
            "dataType[%u], locEid[%s], rmtEid[%s], taskDstAddr[0x%llx], taskSrcAddr[0x%llx], "
            "rtCqErrorType[%u], rtCqErrorCode[%u], ubCqeStatus[%u], opType[%u], jettyHandle[0x%llx], "
            "jettyId[%u], tpn[%u]",
            tag, group, remoteUserRank, streamId, taskId, static_cast<unsigned long long>(notifyId), notifyValue, stage,
            rankId, rankSize, taskType.Describe().c_str(), linkType.Describe().c_str(), size,
            static_cast<unsigned long long>(count), static_cast<unsigned long long>(dstAddr),
            static_cast<unsigned long long>(srcAddr), opIndex, reduceType, dataType, locEid.Describe().c_str(),
            rmtEid.Describe().c_str(), static_cast<unsigned long long>(taskDstAddr),
            static_cast<unsigned long long>(taskSrcAddr), rtCqErrorType, rtCqErrorCode, ubCqeStatus, opType,
            static_cast<unsigned long long>(jettyHandle), jettyId, tpn);
    }
};

} // namespace Hccl

#endif

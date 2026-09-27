/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_share_data_manager.h"
namespace hccl {
HcclResult AicpuShareDataManager::Init(u64 addr, u64 size)
{
    // 初始化公共数据
    aicpuCustomParam_ = reinterpret_cast<AicpuCustomParam*>(addr);
    if (aicpuCustomParam_ == nullptr || size != sizeof(AicpuCustomParam)) {
        HCCL_ERROR(
            "%s fail, addr[%p] is null or size[%llu] is not equal to the size[%llu] of AicpuCustomParam", __func__,
            aicpuCustomParam_, size, sizeof(AicpuCustomParam));
        return HCCL_E_PARA;
    }
    HCCL_INFO("%s success, aicpuCustomParam:%p, size:%llu", __func__, aicpuCustomParam_, size);
    return HCCL_SUCCESS;
}

u32 AicpuShareDataManager::GetOpRingBufferIdx()
{
    CHK_PRT_RET(aicpuCustomParam_ == nullptr, HCCL_ERROR("%s fail, aicpuCustomParam is nullptr", __func__), 0);
    return aicpuCustomParam_->taskExceptionParam.opRingBufferIdx;
}

HcclResult AicpuShareDataManager::RecordOpInfo(
    const std::string& newTag, OpParam& opParam, u32 opExecIndex, u32 userRank, bool isCustom)
{
    CHK_PRT_RET(aicpuCustomParam_ == nullptr, HCCL_ERROR("%s fail, aicpuCustomParam is nullptr", __func__), HCCL_E_PTR);
    auto& aicpuOpInfo = aicpuCustomParam_->taskExceptionParam.opInfo;
    // opRingBufferIdx是算子信息在aicpuOpInfo数组中的索引，记录的sqe信息也会记录这个值，实现sqe和算子信息的匹配
    u32& opRingBufferIdx = aicpuCustomParam_->taskExceptionParam.opRingBufferIdx;

    CHK_SAFETY_FUNC_RET(strcpy_s(aicpuOpInfo[opRingBufferIdx].tagBuff, HCCL_TAG_SIZE, newTag.c_str()));
    aicpuOpInfo[opRingBufferIdx].opIndex = opParam.index;
    aicpuOpInfo[opRingBufferIdx].opExecIndex = opExecIndex;
    HCCL_DEBUG(
        "%s tag[%s] opRingBufferIdx[%u] opIndex[%u] rootId[%u] opType[%u] srcAddr[0x%x] dstAddr[0x%x]", __func__,
        aicpuOpInfo[opRingBufferIdx].tagBuff, opRingBufferIdx, aicpuOpInfo[opRingBufferIdx].opIndex, opParam.root,
        opParam.opType, opParam.inputPtr, opParam.outputPtr);
    if (opParam.opType == HcclCMDType::HCCL_CMD_INVALID) {
        return HCCL_E_PARA;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        aicpuOpInfo[opRingBufferIdx].count = SYS_MAX_COUNT;
        aicpuOpInfo[opRingBufferIdx].dataType = HCCL_DATA_TYPE_RESERVED;
    } else if (
        opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC
        || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        aicpuOpInfo[opRingBufferIdx].count = opParam.All2AllDataDes.sendCount;
        aicpuOpInfo[opRingBufferIdx].dataType = opParam.All2AllDataDes.sendType;
    } else if (
        opParam.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V
        || opParam.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        aicpuOpInfo[opRingBufferIdx].count = static_cast<u64*>(opParam.VDataDes.counts)[userRank];
        aicpuOpInfo[opRingBufferIdx].dataType = opParam.VDataDes.dataType;
    } else {
        aicpuOpInfo[opRingBufferIdx].count = opParam.DataDes.count;
        aicpuOpInfo[opRingBufferIdx].dataType = opParam.DataDes.dataType;
    }
    HCCL_DEBUG(
        "[HcclCommAicpu][RecordOpInfo] count[%llu] dataType[%u]", aicpuOpInfo[opRingBufferIdx].count,
        aicpuOpInfo[opRingBufferIdx].dataType);

    aicpuOpInfo[opRingBufferIdx].opType = static_cast<uint8_t>(opParam.opType);
    aicpuOpInfo[opRingBufferIdx].rootId = opParam.root;
    aicpuOpInfo[opRingBufferIdx].dstAddr = reinterpret_cast<uint64_t>(opParam.outputPtr);
    aicpuOpInfo[opRingBufferIdx].srcAddr = reinterpret_cast<uint64_t>(opParam.inputPtr);
    aicpuOpInfo[opRingBufferIdx].reduceType = opParam.reduceType;
    aicpuOpInfo[opRingBufferIdx].isCustom = isCustom;
    opRingBufferIdx++; // +1后为下一个算子的index
    opRingBufferIdx = opRingBufferIdx % OPINFO_RING_BUFFER_MAX;
    return HCCL_SUCCESS;
}

const AicpuOpInfo* AicpuShareDataManager::GetAicpuOpInfo(u32 opRingBufferIdx)
{
    CHK_PRT_RET(aicpuCustomParam_ == nullptr, HCCL_ERROR("%s fail, aicpuCustomParam is nullptr", __func__), nullptr);
    CHK_PRT_RET(
        opRingBufferIdx >= OPINFO_RING_BUFFER_MAX,
        HCCL_ERROR(
            "%s fail, opRingBufferIdx[%u] should be smaller than %u", __func__, opRingBufferIdx,
            OPINFO_RING_BUFFER_MAX),
        nullptr);

    return &(aicpuCustomParam_->taskExceptionParam.opInfo[opRingBufferIdx]);
}
} // namespace hccl

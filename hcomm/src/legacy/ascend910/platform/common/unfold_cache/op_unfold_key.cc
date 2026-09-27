/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>

#include "op_unfold_key.h"

#include "log.h"

namespace hccl {
OpUnfoldKey::OpUnfoldKey()
    : opType(HcclCMDType::HCCL_CMD_INVALID),
      dataType(HcclDataType::HCCL_DATA_TYPE_RESERVED),
      reduceType(HcclReduceOp::HCCL_REDUCE_RESERVED),
      isZeroCopy(false),
      isSymmetricMemory(false),
      inputSize(0),
      isInplacePreSync(false),
      workflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED),
      isCapture(false),
      root(0)
{}

OpUnfoldKey::OpUnfoldKey(const OpUnfoldKey& other)
    : opType(other.opType),
      dataType(other.dataType),
      reduceType(other.reduceType),
      isZeroCopy(other.isZeroCopy),
      isSymmetricMemory(other.isSymmetricMemory),
      inputSize(other.inputSize),
      isInplacePreSync(other.isInplacePreSync),
      workflowMode(other.workflowMode),
      isCapture(other.isCapture),
      root(other.root)
{
    CHK_PRT_CONT(opType == HcclCMDType::HCCL_CMD_INVALID, HCCL_ERROR("[OpUnfoldKey][OpUnfoldKey] opType is invalid"));
    if (opType != HcclCMDType::HCCL_CMD_ALLTOALLV
        && opType != HcclCMDType::HCCL_CMD_ALLTOALLVC) { // 非V类算子, dataType一定不是RESERVED; 如果是alltoallv类算子
                                                         // (alltoallv/alltoallvc), dataType一定是RESERVED
        CHK_PRT_CONT(
            dataType == HcclDataType::HCCL_DATA_TYPE_RESERVED,
            HCCL_ERROR("[OpUnfoldKey][OpUnfoldKey] dataType is reserved"));
    }
    CHK_PRT_CONT(
        workflowMode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED,
        HCCL_ERROR("[[OpUnfoldKey][OpUnfoldKey]] workflowMode is reserved"));

    // 注意: 当算子不涉及reduce操作时, reduceType为HcclReduceOp::HCCL_REDUCE_RESERVED
}

HcclResult OpUnfoldKey::Init(
    const HcclCMDType curOpType, const HcclDataType curDataType, const HcclReduceOp curReduceType,
    const bool curIsZeroCopy, const bool curIsSymmetricMemory, const uint64_t curInputSize,
    const bool curIsInplacePreSync, const HcclWorkflowMode curWorkflowMode, const bool curIsCapture,
    const uint32_t curRoot)
{
    CHK_PRT_RET(
        curOpType == HcclCMDType::HCCL_CMD_INVALID, HCCL_ERROR("[OpUnfoldKey][OpUnfoldKey] opType is invalid"),
        HCCL_E_INTERNAL);
    if (curOpType != HcclCMDType::HCCL_CMD_ALLTOALLV
        && curOpType
               != HcclCMDType::HCCL_CMD_ALLTOALLVC) { // 非V类算子, dataType一定不是RESERVED; 如果是alltoallv类算子
                                                      // (alltoallv/alltoallvc), dataType一定是RESERVED
        CHK_PRT_RET(
            curDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED,
            HCCL_ERROR("[OpUnfoldKey][OpUnfoldKey] dataType is reserved"), HCCL_E_INTERNAL);
    }
    CHK_PRT_RET(
        curWorkflowMode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED,
        HCCL_ERROR("[[OpUnfoldKey][Init]] workflowMode is reserved"), HCCL_E_INTERNAL);

    // 注意: 当算子不涉及reduce操作时, reduceType为HcclReduceOp::HCCL_REDUCE_RESERVED

    opType = curOpType;
    dataType = curDataType;
    reduceType = curReduceType;
    isZeroCopy = curIsZeroCopy;
    isSymmetricMemory = curIsSymmetricMemory;
    inputSize = curInputSize;
    isInplacePreSync = curIsInplacePreSync;
    workflowMode = curWorkflowMode;
    isCapture = curIsCapture;
    // 注意: curRoot 默认值为 0, 对非root类算子无影响; 对 scatter/broadcast/reduce 会区分不同 root
    root = curRoot;

    return HCCL_SUCCESS;
}

std::string OpUnfoldKey::GetKeyString() const
{
    std::ostringstream oss;
    oss << "opType" << static_cast<uint32_t>(opType) << "-dataType" << static_cast<uint32_t>(dataType) << "-reduceType"
        << static_cast<uint32_t>(reduceType) << "-isZeroCopy" << isZeroCopy << "-isSymmetricMemory" << isSymmetricMemory
        << "-inputSize" << inputSize << "-isInplacePreSync" << isInplacePreSync << "-workflowMode"
        << static_cast<uint32_t>(workflowMode) << "-isCapture" << isCapture << "-root" << root;
    return oss.str();
}

bool OpUnfoldKey::operator==(const OpUnfoldKey& other) const
{
    return opType == other.opType && dataType == other.dataType && reduceType == other.reduceType
           && isZeroCopy == other.isZeroCopy && isSymmetricMemory == other.isSymmetricMemory
           && inputSize == other.inputSize && isInplacePreSync == other.isInplacePreSync
           && workflowMode == other.workflowMode && isCapture == other.isCapture && root == other.root;
}

const OpUnfoldKey& OpUnfoldKey::operator=(const OpUnfoldKey& other)
{
    if (this != &other) {
        this->opType = other.opType;
        this->dataType = other.dataType;
        this->reduceType = other.reduceType;
        this->isZeroCopy = other.isZeroCopy;
        this->isSymmetricMemory = other.isSymmetricMemory;
        this->inputSize = other.inputSize;
        this->isInplacePreSync = other.isInplacePreSync;
        this->workflowMode = other.workflowMode;
        this->isCapture = other.isCapture;
        this->root = other.root;
    }
    return *this;
}

}; // namespace hccl

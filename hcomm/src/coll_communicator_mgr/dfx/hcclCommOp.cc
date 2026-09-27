/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcclCommOp.h"

namespace hccl {

std::shared_ptr<Hccl::DfxOpInfo> ConvertToDfxOpInfo(const HcclDfxOpInfo& dfxOpInfo)
{
    auto dfxOpInfoOnce = std::make_shared<Hccl::DfxOpInfo>();
    dfxOpInfoOnce->op_.opMode = static_cast<Hccl::OpMode::Value>(dfxOpInfo.opMode);
    dfxOpInfoOnce->op_.oldOpType = dfxOpInfo.opType;     // 存A3的类型
    dfxOpInfoOnce->op_.oldReduceOp = dfxOpInfo.reduceOp; // 存A3的类型
    dfxOpInfoOnce->op_.oldDataType = dfxOpInfo.dataType; // 存A3的类型

    dfxOpInfoOnce->op_.dataCount = dfxOpInfo.dataCount;
    dfxOpInfoOnce->op_.root = dfxOpInfo.root;

    dfxOpInfoOnce->op_.inputAddr = dfxOpInfo.inputMemAddr;
    dfxOpInfoOnce->op_.inputSize = dfxOpInfo.inputMemSize;
    dfxOpInfoOnce->op_.outputAddr = dfxOpInfo.outputMemAddr;
    dfxOpInfoOnce->op_.outputSize = dfxOpInfo.outputMemSize;

    dfxOpInfoOnce->algTag_ = dfxOpInfo.algTag;
    dfxOpInfoOnce->beginTime_ = dfxOpInfo.beginTime;
    dfxOpInfoOnce->cpuWaitAicpuNotifyId_ = dfxOpInfo.cpuWaitAicpuNotifyId;

    return dfxOpInfoOnce;
}

} // namespace hccl

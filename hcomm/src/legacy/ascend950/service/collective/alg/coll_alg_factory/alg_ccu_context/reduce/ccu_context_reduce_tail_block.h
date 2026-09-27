/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_REDUCE_TAIL_BLOCK_H_
#define HCCLV2_CCU_CONTEXT_REDUCE_TAIL_BLOCK_H_

#include <vector>
#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"

namespace Hccl {

class CcuContextReduceTailBlock : public CcuContextAlgBase {
public:
    CcuContextReduceTailBlock(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextReduceTailBlock() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

private:
    void ExportVariables();
    void InitResource();
    void DoGroupReduce();
    void SyncMainBlock(uint32_t ctxSignalIndex);

    std::string notifySignal_ = "empty_signal";

    DataType dataType_;
    DataType outputDataType_;

    std::vector<CcuRep::Variable> token_;
    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;

    CcuRep::Variable inputOffset_;
    CcuRep::Variable outputOffset_;

    GroupOpSize groupOpSize_;

    CcuRep::MaskSignal mainBlockCtxSignal_;
    CcuRep::MaskSignal tailBlockCtxSignal_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_REDUCE_TAIL_BLOCK_H_

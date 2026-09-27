/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_TRANS_MEM_EXECUTOR_H
#define HCCL_SIM_TRANS_MEM_EXECUTOR_H

#include <cstdint>
#include <string>

#include "ccu_executor_base.h"
#include "ccu_microcode_common_v2.h"
#include "ccu_resource_manager.h"

class TransMemExecutor : public CcuExecutorBase {
public:
    explicit TransMemExecutor(
        int streamId, int rankId, int dieId, const hcomm::CcuRep::CcuInstr& instr, CcuSimulator* ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    TransMemExecutor() = default;
    ~TransMemExecutor() = default;

    void Parser() override;
    void Run() override;
    std::string Describe() override;
    CcuTrace::CcuInstrTraceDetail CollectTraceDetail() override;

private:
    uint16_t xdId_;
    uint16_t xdtId_;
    uint16_t xsId_;
    uint16_t xstId_;
    uint16_t xlId_;
    uint16_t xcId_;
    uint16_t xnId_;
    uint16_t xntId_;
    uint32_t value_;
    uint8_t udfType_;
    uint8_t reduceDataType_;
    uint8_t reduceOpCode_;
    uint8_t dmaOpCode_;
    uint8_t order_;
    uint8_t fence_;
    uint8_t cqe_;
    uint8_t nf_;
    uint8_t udfEnable_;
    uint8_t splitMode_;
    uint8_t se_;
    uint8_t rmtJettyType_;
    uint8_t srcMode_;
    uint8_t dstMode_;
    uint8_t msIdmode_;
    uint8_t targetHint_;
    uint16_t setCKEId_;
    uint16_t setCKEMask_;
};

#endif // HCCL_SIM_TRANS_MEM_EXECUTOR_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_REDUCE_MIN_EXECUTOR_H
#define HCCL_SIM_REDUCE_MIN_EXECUTOR_H

#include <cstdint>
#include <string>

#include "ccu_executor_base.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_resource_manager.h"

class ReduceMinExecutor : public CcuExecutorBase {
    enum DataType {
        ADD_FP32,
        ADD_FP16,
        ADD_BF16,
        ADD_HIF8,
        ADD_FP8_E4M3,
        ADD_FP8_E5M2,
        ADD_INT8,
        ADD_UINT8,
        ADD_INT16,
        ADD_INT32,
        ADD_RESERVED
    };

public:
    explicit ReduceMinExecutor(
        int streamId, int rankId, int dieId, const hcomm::CcuRep::CcuInstr& instr, CcuSimulator* ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {
        (void)memset(msId_, 0, sizeof(uint16_t) * hcomm::CcuRep::CCU_REDUCE_MAX_MS);
    }
    ReduceMinExecutor() = default;
    ~ReduceMinExecutor() = default;

    void Parser() override;
    void Run() override;
    void RunV1();
    void RunV2();
    void Process(CcuResourceManager& ccuResMgr) override;
    std::string Describe() override;
    CcuTrace::CcuInstrTraceDetail CollectTraceDetail() override;

private:
    uint16_t count_{0};
    uint16_t dataType_{0};
    uint16_t clearType_{0};
    uint16_t setCKEId_{0};
    uint16_t setCKEMask_{0};
    uint16_t waitCKEId_{0};
    uint16_t waitCKEMask_{0};
    uint16_t msId_[hcomm::CcuRep::CCU_REDUCE_MAX_MS];
};

#endif // HCCL_SIM_REDUCE_MIN_EXECUTOR_H

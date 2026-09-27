/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_TRANS_LOCMEM_TO_LOCMEM_EXECUTOR_H
#define HCCL_SIM_TRANS_LOCMEM_TO_LOCMEM_EXECUTOR_H

#include <cstdint>
#include <string>

#include "ccu_executor_base.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_microcode_common_v2.h"
#include "ccu_resource_manager.h"

class TransLocMemToLocMemExecutor : public CcuExecutorBase {
public:
    explicit TransLocMemToLocMemExecutor(
        int streamId, int rankId, int dieId, const hcomm::CcuRep::CcuInstr& instr, CcuSimulator* ccuSimulator)
        : CcuExecutorBase(streamId, rankId, dieId, instr, ccuSimulator)
    {}
    TransLocMemToLocMemExecutor() = default;
    ~TransLocMemToLocMemExecutor() = default;

    void Parser() override;
    void Run() override;
    void RunV1();
    void RunV2();
    void Process(CcuResourceManager& ccuResMgr) override;
    std::string Describe() override;
    CcuTrace::CcuInstrTraceDetail CollectTraceDetail() override;

private:
    uint16_t dstGSAId_{0};
    uint16_t dstXnId_{0};
    uint16_t srcGSAId_{0};
    uint16_t srcXnId_{0};
    uint16_t lengthXnId_{0};
    uint16_t channelId_{0};
    uint16_t clearType_{0};
    uint16_t lengthEn_{0};
    uint16_t setCKEId_{0};
    uint16_t setCKEMask_{0};
    uint16_t waitCKEId_{0};
    uint16_t waitCKEMask_{0};
    uint64_t transLength_{0};

    uint16_t xdId_{0};
    uint16_t xdtId_{0};
    uint16_t xsId_{0};
    uint16_t xstId_{0};
    uint16_t xlId_{0};
    uint16_t usedMSId_{0};
    uint16_t msNum_{0};
};

#endif // HCCL_SIM_TRANS_LOCMEM_TO_LOCMEM_EXECUTOR_H

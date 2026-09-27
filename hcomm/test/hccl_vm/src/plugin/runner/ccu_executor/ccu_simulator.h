/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_SIMULATOR_H
#define HCCL_SIM_CCU_SIMULATOR_H

#include <cstdint>

#include "ccu_resource_common.h"
#include "ccu_simulator_base.h"

class CcuSimulator {
public:
    explicit CcuSimulator(
        int rankId, int dieId, uint16_t startInstrId, uint16_t endInstrId, uint16_t instrCnt, RunnerCcuVersion version)
        : rankId_(rankId),
          dieId_(dieId),
          curInstrId_(startInstrId),
          startInstrId_(startInstrId),
          endInstrId_(endInstrId),
          instrCnt_(instrCnt),
          version_(version)
    {}
    CcuSimulator() = default;
    ~CcuSimulator() = default;

    bool Execute();
    bool ExecuteLoop();
    bool ExecuteLoopGroup();
    bool ExecuteInstr(uint16_t curInstrId);
    bool UpdateLoopStatus();

    void SetWaitCKEFlag(bool needCKE);
    void SetExecState(CcuExecState state);

    void Init(uint16_t startInstrId, uint16_t endInstrId, uint16_t instrCnt, RunnerCcuVersion version);
    void InitLoopGroupInfo(const LoopGroupInfo& loopGroupInfo);
    void InitLoopGroupInfo(uint16_t startLoopId, uint64_t offsetCfg, uint64_t repeatCfg);
    void InitLoopGroupInfoV2(uint16_t startLoopId, uint64_t xnValue, uint64_t xmValue, uint64_t xpValue);
    void InitLoopInfo(uint16_t startInstrId, uint16_t endInstrId, uint16_t execCount, uint32_t addrStep);
    void
    InitLoopInfoV2(uint16_t startInstrId, uint16_t endInstrId, uint64_t xnValue, uint64_t xmValue, uint64_t xpValue);
    void InitJumpStatus(uint16_t jumpInstrId);

    uint64_t GetLoopGsaAddrOffset();
    uint16_t GetLoopMsOffset();
    uint16_t GetLoopCKEOffset();
    uint16_t GetCurInstrId();
    uint16_t GetLoopXnIdOffset();
    uint16_t GetStartInstrId() { return startInstrId_; };

    uint16_t GetCurLoopCnt();
    uint32_t GetLoopIterStepGSA();
    uint16_t GetLoopExtendNum();
    uint32_t GetGSAOffset();

    CcuExecState GetState();

private:
    int rankId_{0};
    int dieId_{0};
    bool finished_{false};
    bool waitCKE_{false}; // 是否需要等待CKE
    uint16_t curInstrId_;
    uint16_t startInstrId_;
    uint16_t endInstrId_;
    uint16_t instrCnt_;
    uint16_t jumpInstrId_{0};
    uint16_t instrType_{0};   // 当前指令的类型(主要用于记录当前执行是否为Loop)
    bool initialized_{false}; // 是否已经初始化
    CcuExecState state_{CcuExecState::EXEC_NORMAL_INSTR}; // 当前ccu的执行状态
    RunnerCcuVersion version_{RunnerCcuVersion::CCU_V1};

    LoopGroupInfo loopGroupInfo_; // loopGroup指令信息
};

#endif // HCCL_SIM_CCU_SIMULATOR_H

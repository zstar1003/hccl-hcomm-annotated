/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_EXECUTOR_BASE_H
#define HCCL_SIM_CCU_EXECUTOR_BASE_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "ccu_microcode_common_v1.h"
#include "ccu_resource_manager.h"
#include "trace/ccu_trace_types.h"
#include "sim_log.h"

class CcuExecutorBase {
public:
    CcuExecutorBase(
        int streamId, int rankId, int dieId, const hcomm::CcuRep::CcuInstr& instr, CcuSimulator* ccuSimulator)
        : streamId_(streamId),
          rankId_(rankId),
          dieId_(dieId),
          instr_(instr),
          ccuSimulator_(ccuSimulator)
    {}
    virtual ~CcuExecutorBase() = default;

    virtual void Parser() = 0;
    virtual void Run() = 0;
    virtual std::string Describe() = 0;
    virtual void Process(CcuResourceManager& ccuResMgr) { return; }

    // 采集指令专属 trace 细节（仅含运行时动态参数）
    // 默认实现：仅包含 typeName，子类可 override 补充 args
    virtual CcuTrace::CcuInstrTraceDetail CollectTraceDetail()
    {
        CcuTrace::CcuInstrTraceDetail detail;
        detail.typeName = "Unknown";
        return detail;
    }

    RunnerCcuVersion GetVersion() const { return version_; }
    void SetVersion(RunnerCcuVersion version) { version_ = version; }

    std::string ParseMSList();
    void SetCkeSignal(CcuResourceManager& ccuResMgr, uint16_t setCKEId, uint16_t setCKEMask);
    void SetRmtCKESignal(
        CcuResourceManager& ccuResMgr, int rmtRank, int rmtDie, uint16_t setRmtCKEId, uint16_t setRmtCKEMask);
    void ClearCkeSignal(CcuResourceManager& ccuResMgr, uint16_t clearCKEId, uint16_t clearMask);
    void WaitCkeProcess(uint16_t waitCKEId, uint16_t waitCKEMask, uint16_t clearType, const std::string& instrName);

    uint16_t UpdateXnId(uint16_t xnIdField);
    uint16_t UpdateCkeId(uint16_t ckeId);
    uint16_t UpdateMSId(uint16_t msId);
    uint16_t GetXnId(uint16_t xnIdField);

    uint64_t UpdateAddressWithoutStride(uint64_t addr);

    uint64_t UpdateAddress(uint64_t addr, uint16_t addrExpandCoef = 0);

protected:
    void ValidateVersion(RunnerCcuVersion expectedVersion, const std::string& instrName)
    {
        if (version_ != expectedVersion) {
            HCCL_VM_ERROR(
                "[{}] Version mismatch: expected={}, actual={}", instrName, static_cast<int>(expectedVersion),
                static_cast<int>(version_));
        }
    }

    void ValidateVersionExclusive(RunnerCcuVersion allowedVersion, const std::string& instrName)
    {
        if (version_ != allowedVersion) {
            HCCL_VM_ERROR(
                "[{}] Instruction is exclusive to CCU_V{}, but running on V{}", instrName,
                static_cast<int>(allowedVersion), static_cast<int>(version_));
        }
    }

public:
    int rankId_{0};
    int dieId_{0};
    int streamId_{0};
    hcomm::CcuRep::CcuInstr instr_;       // 指令信息
    CcuSimulator* ccuSimulator_{nullptr}; // ccu指令模拟器
    RunnerCcuVersion version_{RunnerCcuVersion::CCU_INVALID};
    static std::map<std::string, uint32_t> s_blockingCountMap_;
};

#endif // HCCL_SIM_CCU_EXECUTOR_BASE_H

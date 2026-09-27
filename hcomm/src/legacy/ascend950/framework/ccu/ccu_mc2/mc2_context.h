/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_MC2_CONTEXT_H
#define HCCL_MC2_CONTEXT_H

#include <vector>
#include <array>
#include <unordered_map>
#include "ccu_ctx.h"
#include "mc2_type.h"

namespace Hccl {

class Mc2ContextBase : public CcuContext {
public:
    Mc2ContextBase() = default;
    ~Mc2ContextBase() override {}

    void Algorithm() override;

    void SetAlgoTemplateInfo(const std::map<uint64_t, uint32_t>& algoTemplateInfo);
    void SetMissionNumAndId(uint32_t miNum, uint32_t miIndex);
    void MissionPreSync(CcuRep::Variable& func);
    void MissionPostSync();

protected:
    void GenOpSelector();
    virtual void GenCircularQueue() = 0;

protected:
    // 同一个Die上的Mission数量
    uint32_t missionNum{1};
    // 同一个Die上的Mission的编号
    uint32_t missionIndex{0};
    // 用于Mission间同步信号
    std::vector<CcuRep::MaskSignal> exportMissoinSig;
    std::vector<CcuRep::MaskSignal> importMissionSig;
    // 用于Mission间同步变量
    std::vector<CcuRep::Variable> exportMissionVar;
    std::vector<CcuRep::Variable> importMissionVar;
    // 算子签名与起始地址Map <指令模板签名, 指令起始地址>，用于算子选择
    std::map<uint64_t, uint32_t> algoTemplateInfo_;
};

class Mc2Context : public Mc2ContextBase {
public:
    Mc2Context() = default;
    ~Mc2Context() override {}

    void SetCommAddr(uint64_t syncAddr, uint64_t paramAddr);
    void SetDieNum(uint32_t dieNum);

    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

protected:
    void GenCircularQueue() override;

private:
    void WaitTurnStartSig(const CcuRep::Variable& hbmSigAddr, CcuRep::Variable& turnStartSig);
    void SetTurnEndSig(const CcuRep::Variable& hbmSigAddr, const CcuRep::Variable& turnEndSig);
    void
    LoadFuncParamFromMemory(CcuRep::Variable& paramAddr, std::array<CcuRep::Variable, CCU_PARAM_NUM_PER_DIE>& param);

private:
    // HBM上的每轮开始信号的首地址
    uint64_t waitAddr_{0};
    // HBM上的每轮完成信号的首地址
    uint64_t recordAddr_{0};
    // HBM上的算子执行参数首地址
    uint64_t paramAddr_{0};

    // Die数量，用于判断单双Die，默认为1即单Die
    uint32_t dieNum_{1};
    // 用于Die间同步信号
    CcuRep::MaskSignal exportDieSig;
    CcuRep::MaskSignal importDieSig;
};

class Mc2SlaveContext : public Mc2ContextBase {
public:
    Mc2SlaveContext() = default;
    ~Mc2SlaveContext() override {}

    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

protected:
    void GenCircularQueue() override;
};
} // namespace Hccl

#endif // HCCL_MC2_CONTEXT_H

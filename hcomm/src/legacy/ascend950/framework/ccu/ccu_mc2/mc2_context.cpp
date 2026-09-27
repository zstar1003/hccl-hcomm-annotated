/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_context.h"
#include "ccu_task_arg_mc2.h"
#include "const_val.h"

namespace Hccl {

using namespace std;

const string OP_SELECTOR_LABEL = "OpSelector";
// HBM参数index
const uint32_t HBM_PARAM_IDX_0 = 0;
const uint32_t HBM_PARAM_IDX_1 = 1;
const uint32_t HBM_PARAM_IDX_2 = 2;
const uint32_t HBM_PARAM_IDX_3 = 3;

const uint32_t SINGLE_DIE = 1;                // 单Die数量
const uint32_t DOUBLE_DIE = 2;                // 双Die数量
const uint32_t DIE0_ID = 0;                   // Die0 ID
const uint32_t DIE1_ID = 1;                   // Die1 ID
const string DIE1_START_SIG = "Die1StartSig"; // 双Die场景，Die0通知Die1开始执行的信号
const string DIE1_END_SIG = "Die1EndSig";     // 双Die场景，Die1通知Die0执行完成的信号

void Mc2ContextBase::SetAlgoTemplateInfo(const map<uint64_t, uint32_t>& algoTemplateInfo)
{
    algoTemplateInfo_ = algoTemplateInfo;
    for (const auto& pair : algoTemplateInfo_) {
        HCCL_INFO("[Mc2Context::SetAlgoTemplateInfo] algoSignature[%llu] startInstr[%u]", pair.first, pair.second);
    }
}

void Mc2ContextBase::SetMissionNumAndId(uint32_t miNum, uint32_t miIndex)
{
    this->missionNum = miNum;
    this->missionIndex = miIndex;
    if (missionIndex >= miNum) {
        THROW<InvalidParamsException>("MC2 High Level API SetMissionNumAndId Failed: Invalid Mission Config");
    }
    if (miNum > 1) { // 多Mission场景才需要导入导出
        if (miIndex == 0) {
            // missionIndex = 0 为Master，需要missionNum - 1个导入导出信号以及missionNum - 1个导入变量
            for (uint32_t i = 0; i < miNum - 1; ++i) {
                exportMissoinSig.push_back(CreateMaskSignal());
                ExportMaskSignal(
                    exportMissoinSig[i], "master_sig_" + std::to_string(GetDieId()) + "_" + std::to_string(i + 1));
                importMissionSig.push_back(
                    ImportMaskSignal("slave_sig_" + std::to_string(GetDieId()) + "_" + std::to_string(i + 1)));
                importMissionVar.push_back(
                    ImportVariable("slave_var_" + std::to_string(GetDieId()) + "_" + std::to_string(i + 1)));
            }
        } else {
            // missionIndex > 0 为Slave，需要1个导入导出信号以及1个导出变量
            importMissionSig.push_back(
                ImportMaskSignal("master_sig_" + std::to_string(GetDieId()) + "_" + std::to_string(miIndex)));
            exportMissoinSig.push_back(CreateMaskSignal());
            ExportMaskSignal(
                exportMissoinSig[0], "slave_sig_" + std::to_string(GetDieId()) + "_" + std::to_string(miIndex));
            exportMissionVar.push_back(CreateVariable());
            ExportVariable(
                exportMissionVar[0], "slave_var_" + std::to_string(GetDieId()) + "_" + std::to_string(miIndex));
        }
    }
}

void Mc2ContextBase::MissionPreSync(CcuRep::Variable& func)
{
    if (missionNum == 1) {
        return;
    }
    if (missionIndex == 0) {
        for (uint32_t i = 0; i < missionNum - 1; ++i) {
            LocalCtxPostVar(func, importMissionVar[i], importMissionSig[i]);
        }
    } else {
        LocalWait(exportMissoinSig[0]);
        func = exportMissionVar[0];
    }
}

void Mc2ContextBase::MissionPostSync()
{
    if (missionNum == 1) {
        return;
    }
    if (missionIndex == 0) {
        for (uint32_t i = 0; i < missionNum - 1; ++i) {
            LocalWait(exportMissoinSig[i]);
        }
    } else {
        LocalCtxPost(importMissionSig[0]);
    }
}

void Mc2ContextBase::GenOpSelector()
{
    if (algoTemplateInfo_.empty()) {
        THROW<InvalidParamsException>("MC2 High Level API GenOpSelector Failed: Empty AlgoTemplateInfo");
    }

    {
        std::string funcName
            = OP_SELECTOR_LABEL + "_" + std::to_string(GetDieId()) + "_" + std::to_string(missionIndex);
        CcuRep::FuncBlock selectorFunc(this, funcName);

        CcuRep::Variable opCode = CreateVariable(); // 函数入参，算子FuncBlock的signature
        selectorFunc.DefineInArg(opCode);

        CcuRep::Variable opAddr = CreateVariable(); // 函数出参，命中算子FuncBlock的函数地址
        selectorFunc.DefineOutArg(opAddr);
        opAddr = INVALID_U64; // opAddr初值为非法值，如果命中算子则会被改为对应的函数地址

        for (auto entry : algoTemplateInfo_) {
            CCU_IF(opCode == entry.first) { opAddr = entry.second; }
        }
    }
}

void Mc2ContextBase::Algorithm()
{
    GenOpSelector();
    GenCircularQueue();
}

void Mc2Context::SetCommAddr(uint64_t syncAddr, uint64_t paramAddr)
{
    waitAddr_ = syncAddr;
    if (syncAddr > (UINT64_MAX - CCU_TASK_NUM_MAX * CCU_ONE_PARAM_SIZE)) {
        THROW<InvalidParamsException>("MC2 High Level API SetDieNum Failed: integer overflow occurs");
    }
    recordAddr_ = syncAddr + CCU_TASK_NUM_MAX * CCU_ONE_PARAM_SIZE; // 偏移8轮的总宽度
    paramAddr_ = paramAddr;
}

void Mc2Context::SetDieNum(uint32_t dieNum)
{
    dieNum_ = dieNum;
    // 参数合法值判断
    bool isDieNumValid = (dieNum_ == SINGLE_DIE || dieNum_ == DOUBLE_DIE);
    bool isDieIdValid = (GetDieId() == DIE0_ID || GetDieId() == DIE1_ID);
    if (!(isDieNumValid && isDieIdValid)) {
        THROW<InvalidParamsException>("MC2 High Level API SetDieNum Failed: Invalid Die Config");
    }

    if (dieNum_ == DOUBLE_DIE) { // 双Die场景才需要导入导出
        // 导出信号
        exportDieSig = CreateMaskSignal();
        // Die0: export完成信号给Die1，Die1: export开始信号给Die0
        const string& exportSigLabel = (GetDieId() == DIE1_ID) ? DIE1_START_SIG : DIE1_END_SIG;
        ExportMaskSignal(exportDieSig, exportSigLabel);

        // 导入信号
        const string& importSigLabel = (GetDieId() == DIE1_ID) ? DIE1_END_SIG : DIE1_START_SIG;
        importDieSig = ImportMaskSignal(importSigLabel);
    }
}

void Mc2Context::GenCircularQueue()
{
    // 存放Token的寄存器
    CcuRep::Variable token = CreateVariable();
    // 从SQE中载入Token
    Load(token);

    // 存放《选择函数返回的FuncCall地址》的寄存器，选择函数的出参，循环队列内部使用
    CcuRep::Variable opAddr = CreateVariable();

    // 存放《控制repeat循环执行的条件》的寄存器
    CcuRep::Variable repeatCond = CreateVariable();
    repeatCond = 0;

    // 存放《轮次执行开始信号》的寄存器，初值为 0
    CcuRep::Variable turnStartSig = CreateVariable();
    turnStartSig = 0;
    // 存放《轮次执行完成信号》的寄存器，在循环中固定为 1
    CcuRep::Variable turnEndSig = CreateVariable();
    turnEndSig = 1;

    CcuRep::Variable waitStartAddr = CreateVariable();
    waitStartAddr = waitAddr_;
    CcuRep::Variable recordStartAddr = CreateVariable();
    recordStartAddr = recordAddr_;
    CcuRep::Variable paramStartAddr = CreateVariable();
    paramStartAddr = paramAddr_;
    CcuRep::Variable waitAddr = CreateVariable();
    waitAddr = waitAddr_;
    CcuRep::Variable recordAddr = CreateVariable();
    recordAddr = recordAddr_;
    CcuRep::Variable paramAddr = CreateVariable();
    paramAddr = paramAddr_;

    CcuRep::Variable ckeSize = CreateVariable();
    ckeSize = CCU_ONE_PARAM_SIZE;
    CcuRep::Variable paramSize = CreateVariable();
    paramSize = CCU_PARAM_NUM_MAX * CCU_ONE_PARAM_SIZE;

    CcuRep::Variable queueIdx = CreateVariable();
    queueIdx = 0;
    CcuRep::Variable queueEnd = CreateVariable();
    queueEnd = CCU_TASK_NUM_MAX;
    CcuRep::Variable one = CreateVariable();
    one = 1;
    // 存放《每轮算子参数》的寄存器
    array<CcuRep::Variable, CCU_PARAM_NUM_PER_DIE> param;
    for (uint32_t i = 0; i < CCU_PARAM_NUM_PER_DIE; ++i) {
        param[i] = CreateContinuousVariable();
    }

    CCU_WHILE(repeatCond == 0)
    {
        // 在context中依次加入8轮指令
        if (waitAddr_ > (UINT64_MAX - (CCU_TASK_NUM_MAX - 1) * CCU_ONE_PARAM_SIZE)
            || recordAddr_ > (UINT64_MAX - (CCU_TASK_NUM_MAX - 1) * CCU_ONE_PARAM_SIZE)
            || paramAddr_ > (UINT64_MAX - (CCU_TASK_NUM_MAX - 1) * CCU_PARAM_NUM_MAX * CCU_ONE_PARAM_SIZE)) {
            THROW<InvalidParamsException>("MC2 High Level API SetDieNum Failed: integer overflow occurs");
        }
        // 等待本轮开始信号
        WaitTurnStartSig(waitAddr, turnStartSig);

        // 读取本轮参数
        LoadFuncParamFromMemory(paramAddr, param);

        MissionPreSync(param[HBM_PARAM_IDX_0]);

        // 第一个参数为opCode, 如果参数中opCode非法则跳出循环队列
        CCU_IF(param[HBM_PARAM_IDX_0] == INVALID_U64) { CCU_BREAK; }

        // 调用OpSelector
        std::string funcName
            = OP_SELECTOR_LABEL + "_" + std::to_string(GetDieId()) + "_" + std::to_string(missionIndex);
        auto selectFunc = Func(funcName);
        selectFunc.SetInArg(param[HBM_PARAM_IDX_0]);
        selectFunc.SetOutArg(opAddr);
        selectFunc.AppendToContext();

        // 检查OpSelector是否命中算子，如果没命中则跳出循环队列
        CCU_IF(opAddr == INVALID_U64) { CCU_BREAK; }

        // 调用算子Func
        auto opFunc = Func(opAddr);
        // 传入参 param[1-31] + token，token需要放在第三个
        opFunc.SetInArg(param[HBM_PARAM_IDX_1]);
        opFunc.SetInArg(param[HBM_PARAM_IDX_2]);
        opFunc.SetInArg(token);
        for (uint32_t i = HBM_PARAM_IDX_3; i < CCU_PARAM_NUM_PER_DIE; ++i) {
            opFunc.SetInArg(param[i]);
        }
        opFunc.AppendToContext();

        MissionPostSync();

        // Set本轮完成信号
        SetTurnEndSig(recordAddr, turnEndSig);
        waitAddr += ckeSize;
        recordAddr += ckeSize;
        paramAddr += paramSize;
        queueIdx += one;
        CCU_IF(queueIdx == static_cast<u64>(CCU_TASK_NUM_MAX))
        {
            waitAddr = waitStartAddr;
            recordAddr = recordStartAddr;
            paramAddr = paramStartAddr;
            queueIdx = 0;
        }
    }
}

void Mc2Context::WaitTurnStartSig(const CcuRep::Variable& hbmSigAddr, CcuRep::Variable& turnStartSig)
{
    if (dieNum_ == SINGLE_DIE) {
        // 单Die场景: 等待HBM中的信号
        CCU_WHILE(turnStartSig != 1)
        {
            // 循环读HBM对应地址的信号到Xn，直到Xn中的信号值为1
            LoadVariable(hbmSigAddr, turnStartSig);
        }
        turnStartSig = 0;                        // reset Xn
        StoreVariable(turnStartSig, hbmSigAddr); // reset HBM
    } else {
        // 双Die场景
        if (GetDieId() == DIE0_ID) {
            // 双Die场景Die0: 等待HBM中的信号，收到HBM信号之后再给Die1发信号，通知Die1开始
            CCU_WHILE(turnStartSig != 1)
            {
                // 循环读HBM对应地址的信号到Xn，直到Xn中的信号值为1
                LoadVariable(hbmSigAddr, turnStartSig);
            }
            turnStartSig = 0;                        // reset Xn
            StoreVariable(turnStartSig, hbmSigAddr); // reset HBM
            // 给Die1发开始信号
            LocalCtxPost(importDieSig, 1);
        } else if (GetDieId() == DIE1_ID) {
            // 双Die场景Die1: 等待Die0的信号
            LocalWait(exportDieSig, 1); // LocalWait会自动reset CKE
        }
    }
}

void Mc2Context::SetTurnEndSig(const CcuRep::Variable& hbmSigAddr, const CcuRep::Variable& turnEndSig)
{
    if (dieNum_ == SINGLE_DIE) {
        // 单Die场景: Set本轮完成信号到HBM
        StoreVariable(turnEndSig, hbmSigAddr);
    } else {
        // 双Die场景
        if (GetDieId() == DIE0_ID) {
            // 双Die场景Die0: 等待Die1执行完成信号，然后Set本轮完成信号到HBM
            LocalWait(exportDieSig, 1); // LocalWait会自动reset CKE
            StoreVariable(turnEndSig, hbmSigAddr);
        } else if (GetDieId() == DIE1_ID) {
            // 双Die场景Die1: 通知Die0执行完成
            LocalCtxPost(importDieSig, 1);
        }
    }
}

void Mc2Context::LoadFuncParamFromMemory(
    CcuRep::Variable& paramAddr, array<CcuRep::Variable, CCU_PARAM_NUM_PER_DIE>& param)
{
    // 双Die场景Die1需要读后32个参数，其他场景都是读前32个参数
    CcuRep::Variable doubleDie = CreateVariable();
    doubleDie = CCU_PARAM_NUM_PER_DIE * CCU_ONE_PARAM_SIZE;
    CcuRep::Variable addr = CreateVariable();
    addr = paramAddr;
    if (dieNum_ == DOUBLE_DIE && GetDieId() == DIE1_ID) {
        addr += doubleDie;
    }

    // 一次性读取本轮32个参数
    LoadVariable(addr, param[0], CCU_PARAM_NUM_PER_DIE);
}

vector<uint64_t> Mc2Context::GeneArgs(const CcuTaskArg& arg)
{
    const CcuTaskArgMc2* taskArg = dynamic_cast<const CcuTaskArgMc2*>(&arg);
    uint64_t tokenInfo = taskArg->token_;
    return {tokenInfo};
}

void Mc2SlaveContext::GenCircularQueue()
{
    // 算子签名
    CcuRep::Variable signature = CreateVariable();
    // 存放《选择函数返回的FuncCall地址》的寄存器，选择函数的出参，循环队列内部使用
    CcuRep::Variable opAddr = CreateVariable();

    // 存放《控制repeat循环执行的条件》的寄存器
    CcuRep::Variable repeatCond = CreateVariable();
    repeatCond = 0;

    CCU_WHILE(repeatCond == 0)
    {
        // 在context中依次加入8轮指令
        MissionPreSync(signature);

        // 第一个参数为opCode, 如果参数中opCode非法则跳出循环队列
        CCU_IF(signature == INVALID_U64) { CCU_BREAK; }

        // 调用OpSelector
        std::string funcName
            = OP_SELECTOR_LABEL + "_" + std::to_string(GetDieId()) + "_" + std::to_string(missionIndex);
        auto selectFunc = Func(funcName);
        selectFunc.SetInArg(signature);
        selectFunc.SetOutArg(opAddr);
        selectFunc.AppendToContext();

        // 检查OpSelector是否命中算子，如果没命中则跳出循环队列
        CCU_IF(opAddr == INVALID_U64) { CCU_BREAK; }

        // 调用算子Func
        auto opFunc = Func(opAddr);

        opFunc.AppendToContext();

        MissionPostSync();
    }
}

vector<uint64_t> Mc2SlaveContext::GeneArgs([[maybe_unused]] const CcuTaskArg& arg) { return {}; }

} // namespace Hccl

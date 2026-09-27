/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_translator_v1.h"

#include <algorithm>

#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_rep_loopcall_v1.h"
#include "ccu_rep_funccall_v1.h"
#include "ccu_rep_type_v1.h"
#include "ccu_rep_loop_v1.h"
#include "ccu_rep_loadarg_v1.h"
#include "ccu_assist_v1.h"

#include "ccu_dev_mgr_imp.h"
#include "dtype_common.h"

#include "ccu_ins_generator_base.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_ins_generator_v2.h"
#include "../../../ccu_device/ccu_res_specs.h"

#include "config_plf_log_v2.h"

#include "microcode_optimizer.h"

namespace hcomm {
namespace CcuRep {

    using Hccl::PLF_TASK;

    template <typename T>
    bool CheckType(const std::shared_ptr<CcuRepBlock>& refer)
    {
        if (refer == nullptr) {
            HCCL_ERROR("[%s] input refer is nullptr", __func__);
            return false;
        }
        HCCL_INFO("[CheckType] refer->Type() = %d", refer->Type());
        return false;
    }

    template <>
    bool CheckType<CcuRepFuncBlock>(const std::shared_ptr<CcuRepBlock>& refer)
    {
        if (refer == nullptr) {
            HCCL_ERROR("[%s] input refer is nullptr", __func__);
            return false;
        }
        return refer->Type() == CcuRepType::FUNC_BLOCK ? true : false;
    }

    template <>
    bool CheckType<CcuRepLoopBlock>(const std::shared_ptr<CcuRepBlock>& refer)
    {
        if (refer == nullptr) {
            HCCL_ERROR("[%s] input refer is nullptr", __func__);
            return false;
        }
        return refer->Type() == CcuRepType::LOOP_BLOCK ? true : false;
    }

    template <typename T1, typename T2>
    void CcuRepTranslator::BuildReference(const std::shared_ptr<CcuRepBase>& rep)
    {
        auto caller = std::static_pointer_cast<T1>(rep);
        auto label = caller->GetLabel();
        // 特例：针对函数地址调用，不需要依靠函数名索引
        if (label == "") {
            return;
        }
        auto refer = refManager->GetRefBlock(label);
        if (CheckType<T2>(refer)) {
            caller->Reference(std::static_pointer_cast<T2>(refer));
        } else {
            Hccl::THROW<Hccl::CcuApiException>("Invalid Reference: %s", label.c_str());
        }
    }

    CcuRepTranslator::CcuRepTranslator(
        int32_t userDevId, uint8_t dieId, std::shared_ptr<CcuRepReferenceManager> refManager,
        std::array<uint16_t, CCU_MAX_IODIE_NUM>& reserverChannalId, std::pair<uint64_t, uint64_t>& ccuTokenInfo,
        uint64_t hbmTokenInfo)
        : refManager(refManager)
    {
        transDep.userDevId = userDevId;
        transDep.dieId = dieId;
        s32 result = memcpy_s(
            transDep.reserveChannalId, sizeof(transDep.reserveChannalId), reserverChannalId.data(),
            sizeof(reserverChannalId));
        if (result != 0) {
            Hccl::THROW<Hccl::InternalException>(Hccl::StringFormat(
                "[NsRecovery] CcuRepTranslator::CcuRepTranslator: memcpy_s failed, ret = %d", result));
        }

        HcclResult ret = CcuDevMgrImp::GetCcuVersion(transDep.userDevId, ccuVersion);
        if (ret != HcclResult::HCCL_SUCCESS || ccuVersion == CcuVersion::CCU_INVALID) {
            Hccl::THROW<Hccl::CcuApiException>("[CcuRepTranslator] Constructor: Invalid CCU Type!");
        }

        // 获取本die的xn起始地址
        ret = CcuDevMgrImp::GetXnBaseAddr(userDevId, dieId, transDep.xnBaseAddr[dieId]);
        if (ret != HcclResult::HCCL_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>(
                "Failed to get xn base address. userDevId = %d, dieId = %u, ret = %d", userDevId, dieId, ret);
        }

        // 若另一die使能，在A6场景下获得其xn起始地址
        uint8_t anotherDieId = dieId == 0 ? 1 : 0;
        bool anotherDieEnable = false;
        if (static_cast<HcclResult>(CcuGetDieEnableInfo(userDevId, anotherDieId, anotherDieEnable))
            != HcclResult::HCCL_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>(
                "Failed to get CcuGetDieEnableInfo. userDevId = %d, dieId = %u", userDevId, dieId);
        }
        if (ccuVersion == CcuVersion::CCU_V2 && anotherDieEnable == true) {
            ret = CcuDevMgrImp::GetXnBaseAddr(userDevId, anotherDieId, transDep.xnBaseAddr[anotherDieId]);
            if (ret != HcclResult::HCCL_SUCCESS) {
                Hccl::THROW<Hccl::CcuApiException>(
                    "Failed to get xn base address. userDevId = %d, dieId = %u, "
                    "ret = %d",
                    userDevId, anotherDieId, ret);
            }
        }

        transDep.ccuResSpaceTokenInfo = CcuRep::GetToken(ccuTokenInfo.first, ccuTokenInfo.second, 1);
        // 获取hbm token信息
        transDep.memTokenInfo = hbmTokenInfo;
    }

    CcuRepTranslator::CcuRepTranslator(std::shared_ptr<CcuRepReferenceManager> refManager, const TransDep& transDep)
        : refManager(refManager),
          transDep(transDep)
    {
        HcclResult ret = CcuDevMgrImp::GetCcuVersion(transDep.userDevId, ccuVersion);
        if (ret != HcclResult::HCCL_SUCCESS || ccuVersion == CcuVersion::CCU_INVALID) {
            Hccl::THROW<Hccl::CcuApiException>("[CcuRepTranslator] Constructor: Invalid CCU Type!");
        }
    }

    uint32_t CcuRepTranslator::GetInstrNum(const int32_t userDevId)
    {
        CcuVersion tempCcuVersion = CcuVersion::CCU_INVALID;
        HcclResult ret = CcuDevMgrImp::GetCcuVersion(userDevId, tempCcuVersion);
        if (ret != HcclResult::HCCL_SUCCESS || tempCcuVersion == CcuVersion::CCU_INVALID) {
            Hccl::THROW<Hccl::CcuApiException>("[CcuRepTranslator] GetInstrNum: Invalid CCU Type!");
        }

        return tempCcuVersion == CcuVersion::CCU_V1 ?
                   4 // 4:翻译器翻译过程中额外需要的指令空间大小(插入3条通用操作指令+1条终止指令)
                   :
                   13; // 13:翻译器翻译过程中额外需要的指令空间大小(插入3条通用操作指令+1条终止指令+9条repJump)
    }

    CcuResReq CcuRepTranslator::GetResReq(const int32_t userDevId, uint8_t dieId)
    {
        // 申请离散 xn 资源
        // 需要申请若干xn、gsa、cke设置为固定值用于通用操作
        CcuVersion tempCcuVersion = CcuVersion::CCU_INVALID;
        HcclResult ret = CcuDevMgrImp::GetCcuVersion(userDevId, tempCcuVersion);
        if (ret != HcclResult::HCCL_SUCCESS || tempCcuVersion == CcuVersion::CCU_INVALID) {
            Hccl::THROW<Hccl::CcuApiException>("[CcuRepTranslator] GetResReq: Invalid CCU Type!");
        }

        CcuResReq resReq;
        int varNum = XN_NUM;
        int gsaNum = tempCcuVersion == CcuVersion::CCU_V1 ? GSA_NUM : 0;
        resReq.xnReq[dieId] = varNum;
        resReq.gsaReq[dieId] = gsaNum;
        resReq.ckeReq[dieId] = CKE_NUM;
        return resReq;
    }

    void CcuRepTranslator::GetRes(CcuRepResource& res)
    {
        int varNum = XN_NUM;
        int gsaNum = ccuVersion == CcuVersion::CCU_V1 ? GSA_NUM : 0;
        for (int i = 0; i < varNum; i++) {
            res.variable[transDep.dieId].push_back(var[i]);
        }
        for (int i = 0; i < gsaNum; i++) {
            res.address[transDep.dieId].push_back(addr[i]);
        }
        for (int i = 0; i < CKE_NUM; i++) {
            res.completedEvent[transDep.dieId].push_back(signal[i]);
        }
    }

    void CcuRepTranslator::PreProcess(std::shared_ptr<CcuRepBase> rep)
    {
        auto repType = rep->Type();
        if (repType == CcuRepType::FUNC_BLOCK) {
            auto funcBlock = std::static_pointer_cast<CcuRepFuncBlock>(rep);
            refManager->SetRefBlock(funcBlock->GetLabel(), funcBlock);
            funcBlock->SetFuncManager(refManager.get());
        } else if (repType == CcuRepType::LOOP_BLOCK) {
            auto loopBlock = std::static_pointer_cast<CcuRepLoopBlock>(rep);
            refManager->SetRefBlock(loopBlock->GetLabel(), loopBlock);
        } else if (repType == CcuRepType::FUNC_CALL) {
            BuildReference<CcuRepFuncCall, CcuRepFuncBlock>(rep);
            auto funcCall = std::static_pointer_cast<CcuRepFuncCall>(rep);
            funcCall->SetFuncManager(refManager.get());
        } else if (repType == CcuRepType::LOOP_CALL) {
            BuildReference<CcuRepLoopCall, CcuRepLoopBlock>(rep);
        } else if (repType == CcuRepType::LOOP) {
            BuildReference<CcuRepLoop, CcuRepLoopBlock>(rep);
        }
    }

    void CcuRepTranslator::Translate(
        CcuKernel* ccuKernel, const std::vector<std::shared_ptr<CcuRepBase>>& repVec, CcuInstr*& instr,
        uint16_t& instrId, std::function<bool(std::shared_ptr<CcuRepBase>)> filter)
    {
        constexpr uint32_t maxTryCount = 10; // 最大尝试次数10
        uint32_t tryCount = 0;
        uint32_t restCount = 0;

        auto funcInVar = refManager.get()->GetFuncIn();
        int funcArgIndex = 0;

        do {
            restCount = 0;
            for (uint32_t index = 0; index < repVec.size(); index++) {
                if (!filter(repVec[index])) {
                    continue;
                }

                if (repVec[index]->Translated()) {
                    continue;
                }

                if (repVec[index]->Type() == CcuRepType::LOAD_ARG && transDep.isFuncBlock) {
                    transDep.loadXnId = funcInVar[funcArgIndex++].Id();
                }

                PreProcess(repVec[index]);
                bool flag = repVec[index]->Translate(ccuKernel, instr, instrId, transDep);
                if (!flag) {
                    restCount++;
                }
            }
            tryCount++;
            HCCL_INFO("tryCount = %u, remaining representation = %u", tryCount, restCount);
        } while (restCount > 0 && tryCount < maxTryCount);

        if (tryCount == maxTryCount && restCount > 0) {
            HCCL_ERROR(
                "After translation, remaining representation: tryCount = %u, restCount = %u ", tryCount, restCount);
            for (uint32_t index = 0; index < repVec.size(); index++) {
                if (!repVec[index]->Translated()) {
                    HCCL_ERROR("index[%u], %s", index, repVec[index]->Describe().c_str());
                }
            }
            Hccl::THROW<Hccl::CcuApiException>("Translation Failed");
        }
    }

    namespace {

        // 收集 LOAD_ARG rep 并按全局 argId 升序排序, 确保 LoadSqeArgs 指令在 mission 切分时
        // 落入与其 slot id 匹配的 mission (避免用户乱序 LoadArg 导致取参错位).
        std::vector<std::shared_ptr<CcuRepBase>>
        CollectSortedLoadArgReps(const std::vector<std::shared_ptr<CcuRepBase>>& repVec)
        {
            std::vector<std::shared_ptr<CcuRepBase>> sortedLoadArgReps;
            sortedLoadArgReps.reserve(repVec.size());
            for (const auto& rep : repVec) {
                if (rep->Type() == CcuRepType::LOAD_ARG) {
                    sortedLoadArgReps.push_back(rep);
                }
            }
            std::stable_sort(
                sortedLoadArgReps.begin(), sortedLoadArgReps.end(),
                [](const std::shared_ptr<CcuRepBase>& a, const std::shared_ptr<CcuRepBase>& b) {
                    return std::static_pointer_cast<CcuRepLoadArg>(a)->GetFullArgId()
                           < std::static_pointer_cast<CcuRepLoadArg>(b)->GetFullArgId();
                });
            return sortedLoadArgReps;
        }

        // 调用 V2 后端优化器 (极简 CkeOnly 档: 只按 CKE 写后读补 NOP, 不重排 / 不重命名寄存器).
        // 默认关闭 (未定义 CCU_MICROCODE_OPT_ENABLE) 时此函数不被引用, 标注避免未使用告警.
        [[maybe_unused]] CcuInstrInfo
        RunV2BackendOptimizer(const CcuInstrInfo& instrInfo, const CcuKernel* ccuKernel, const TransDep& transDep)
        {
            HCCL_INFO("[CcuMicrocodeOpt] running backend optimizer (V2)");
            // 把 kernel 登记的 pinned 组 (array = 连续 XN 寄存器) 转成优化器侧轻量结构, 供 XN 写后读
            // (LoadX/StoreX/ClearX) 把散落 xn 操作数归约到所属 array 组. kernel 为空时退化为不归约.
            std::vector<CcuOpt::PinnedGroup> pinnedGroups;
            if (ccuKernel != nullptr) {
                const auto& groups = ccuKernel->GetPinnedRegGroups();
                pinnedGroups.reserve(groups.size());
                for (const auto& g : groups) {
                    pinnedGroups.push_back(CcuOpt::PinnedGroup{g.baseVar.Id(), g.count});
                }
            }
            CcuInstrInfo optimized
                = CcuOpt::MicrocodeOptimizer::Run(instrInfo, transDep.reserveXnId, transDep.reserveCkeId, pinnedGroups);
            HCCL_INFO(
                "[CcuMicrocodeOpt] before instrCount=%u, after instrCount=%u", instrInfo.instrCount,
                optimized.instrCount);
            return optimized;
        }

    } // namespace

    CcuInstrInfo CcuRepTranslator::Translate(
        CcuKernel* ccuKernel, const std::vector<std::shared_ptr<CcuRepBase>>& repVec, uint16_t startInstrId,
        bool isFuncBlock)
    {
        constexpr uint32_t defaultInstrCapacity = 32 * 1024; // 默认最大容量32 * 1024条
        CcuInstrInfo instrInfo;
        instrInfo.instrVec.resize(defaultInstrCapacity);
        CcuInstr* instr = instrInfo.instrVec.data();
        uint16_t curInstrId = startInstrId;

        BindResource(isFuncBlock);

        // 翻译LoopBlock
        Translate(ccuKernel, repVec, instr, curInstrId, [](std::shared_ptr<CcuRepBase> rep) -> bool {
            return rep->Type() == CcuRepType::LOOP_BLOCK;
        });

        // 翻译funcBlock
        Translate(ccuKernel, repVec, instr, curInstrId, [](std::shared_ptr<CcuRepBase> rep) -> bool {
            return rep->Type() == CcuRepType::FUNC_BLOCK;
        });

        uint16_t missionStartInstrId = curInstrId;

        // 翻译Load (按全局 argId 升序).
        std::vector<std::shared_ptr<CcuRepBase>> sortedLoadArgReps = CollectSortedLoadArgReps(repVec);
        Translate(ccuKernel, sortedLoadArgReps, instr, curInstrId, [](std::shared_ptr<CcuRepBase> rep) -> bool {
            return rep->Type() == CcuRepType::LOAD_ARG;
        });

        // 插入通用操作
        CommonProcess(ccuKernel, instr, curInstrId);

        // 翻译主体
        Translate(ccuKernel, repVec, instr, curInstrId, []([[maybe_unused]] std::shared_ptr<CcuRepBase> rep) -> bool {
            return true;
        });

        FinishMainBlock(instr, curInstrId);

        instrInfo.startInstrId = startInstrId;
        instrInfo.instrCount = curInstrId - startInstrId;
        instrInfo.missionStartInstrId = missionStartInstrId;
        instrInfo.missionInstrCount = curInstrId - missionStartInstrId;
        instrInfo.instrVec.resize(instrInfo.instrCount);

        DumpRep(repVec, instrInfo);
        DumpInstruction(instrInfo);

        if (ccuVersion == CcuVersion::CCU_V2) {
#if !defined(CCU_MICROCODE_OPT_DISABLE)
            // 编译期开启后端优化 (-DCCU_MICROCODE_OPT_ENABLE=ON): 按 CKE 写后读补 NOP.
            CcuInstrInfo optimizedInstrInfo = RunV2BackendOptimizer(instrInfo, ccuKernel, transDep);
            HCCL_INFO("[CcuMicrocodeOpt] dump instruction after backend optimizer:");
            DumpInstruction(optimizedInstrInfo);
            return optimizedInstrInfo;
#else
            // 默认关闭后端优化: 直接返回未优化指令序列, 不做 CKE 补 NOP.
            HCCL_INFO("[CcuMicrocodeOpt] backend optimizer disabled by default; return raw instrInfo");
#endif
        }

        return instrInfo;
    }

    void CcuRepTranslator::CommonProcess(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId)
    {
        if (ccuVersion == CcuVersion::CCU_INVALID) {
            Hccl::THROW<Hccl::InternalException>("[CcuRepTranslator] Unexpected ccuVersion[Invalid]");
        }
        if (ccuVersion == CcuVersion::CCU_V1) {
            LoadImdToXnInstr(instr++, var[0].Id(), 0);
            LoadImdToGSAInstr(instr++, addr[0].Id(), 0);
            SetCKEInstr(instr++, signal[0].Id(), 0xffff, 0, 0, 1);
        } else {
            CcuV2::AssignI(instr++, var[0].Id(), 0);
            CcuV2::AssignI(instr++, var[XN_NUM - 1].Id(), 0);
            SetCKEInstr(instr++, signal[0].Id(), 0xffff, 0, 0, 1);
        }

        // 遍历需要赋值的常量，A5场景下暂为空表
        std::unordered_map<uint64_t, CcuRep::Variable>& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
        u32 constValueNum = constValue2VarMap.size();
        for (auto elem : constValue2VarMap) {
            uint64_t constValue = elem.first;
            CcuRep::Variable curVariable = elem.second;
            if (ccuVersion == CcuVersion::CCU_V1) {
                LoadImdToXnInstr(instr++, curVariable.Id(), constValue);
            } else {
                CcuV2::AssignI(instr++, curVariable.Id(), constValue);
            }
        }

        u32 instrNum = 3 + constValueNum;
        if (instrId > UINT16_MAX - instrNum) {
            Hccl::THROW<Hccl::InternalException>("integer overflow occurs");
        }
        instrId += instrNum; // 插入3条指令
    }

    void CcuRepTranslator::FinishMainBlock(CcuInstr*& instr, uint16_t& instrId)
    {
        if (ccuVersion == CcuVersion::CCU_V1) {
            if (transDep.isFuncBlock) {
                JumpInstr(instr++, refManager.get()->GetFuncRet(FUNC_NEST_MAX).Id(), transDep.reserveXnId, 1);
            } else {
                LoadImdToXnInstr(instr++, var[0].Id(), 0);
            }
            instrId++;
        } else {
            if (transDep.isFuncBlock) {
                CcuV2::RelJmp(
                    instr, refManager.get()->GetFuncRet(FUNC_NEST_MAX).Id(), instrId + V2_RELJMP_INSTR_NUM,
                    transDep.commXn[0], transDep.commXn[1]);
                instr += V2_RELJMP_INSTR_NUM;
                CcuV2::Jump(
                    instr++, refManager.get()->GetFuncRet(FUNC_NEST_MAX).Id(), transDep.reserveXnId,
                    transDep.reserveXnId, 0);
            } else {
                for (int i = 0; i < V2_FINISH_BLOCK_INSTR_NUM; i++) {
                    HCCL_INFO("FinishMainBlock Add Nop: index[%d], instrId[%d]", i, instrId);
                    CcuV2::Nop(instr++);
                }
            }
            instrId += V2_FINISH_BLOCK_INSTR_NUM;
        }

        if (instrId > UINT16_MAX - 1) {
            Hccl::THROW<Hccl::InternalException>("integer overflow occurs");
        }
    }

    void CcuRepTranslator::DumpInstruction(const CcuInstrInfo& instrInfo) const
    {
        HCCL_INFO(
            "CcuInstrInfo: startInstrId = %u, instrCount = %u, missionStartInstrId = %u, missionInstrCount = %u",
            instrInfo.startInstrId, instrInfo.instrCount, instrInfo.missionStartInstrId, instrInfo.missionInstrCount);
        for (uint16_t index = 0; index < instrInfo.instrVec.size(); index++) {
            if (ccuVersion == CcuVersion::CCU_V1) {
                HCCL_INFO(
                    "%d: %s", instrInfo.startInstrId + index, ParseInstr(instrInfo.instrVec.data() + index).c_str());
            } else {
                HCCL_INFO(
                    "%d: %s", instrInfo.startInstrId + index,
                    CcuV2::ParseInstrV2(instrInfo.instrVec.data() + index).c_str());
            }
        }
    }

    void CcuRepTranslator::DumpRep(
        const std::vector<std::shared_ptr<CcuRepBase>>& repVec, const CcuInstrInfo& instrInfo) const
    {
        PLF_CONFIG_INFO(PLF_TASK, "Translated Ccu Rep:");
        for (uint32_t index = 0; index < repVec.size(); index++) {
            uint16_t startInstrId = repVec[index]->StartInstrId();
            uint32_t sum = static_cast<uint32_t>(startInstrId) + repVec[index]->InstrCount();
            if (sum > UINT16_MAX) {
                HCCL_ERROR(
                    "instrId overflow: startInstrId[%u] + InstrCount[%u] = %u exceeds UINT16_MAX", startInstrId,
                    repVec[index]->InstrCount(), sum);
                continue;
            }
            uint16_t endInstrId = static_cast<uint16_t>(sum);
            PLF_CONFIG_INFO(
                PLF_TASK, "rep[%u]: %s Instr[%u--%u]", index, repVec[index]->Describe().c_str(), startInstrId,
                endInstrId);
            for (uint16_t instrId = startInstrId; instrId < endInstrId; instrId++) {
                if (instrId < instrInfo.startInstrId) {
                    HCCL_ERROR("instrId[%u] less than startInstrId[%u]", instrId, instrInfo.startInstrId);
                    continue;
                }
                if (ccuVersion == CcuVersion::CCU_V1) {
                    PLF_CONFIG_INFO(
                        PLF_TASK, "microcode[%u]: %s", instrId,
                        ParseInstr(instrInfo.instrVec.data() + (instrId - instrInfo.startInstrId)).c_str());
                } else {
                    PLF_CONFIG_INFO(
                        PLF_TASK, "microcode[%u]: %s", instrId,
                        CcuV2::ParseInstrV2(instrInfo.instrVec.data() + (instrId - instrInfo.startInstrId)).c_str());
                }
            }
        }
    }

    void CcuRepTranslator::BindResource(bool isFuncBlock)
    {
        // 在V2场景下不会为addr数组申请资源，故reserveGsaId和commGsa中实际填的都是默认值0
        transDep.reserveXnId = var[0].Id();
        transDep.reserveGsaId = addr[0].Id();
        transDep.reserveCkeId = signal[0].Id();
        for (int i = 0; i < XN_NUM - 1; i++) {
            transDep.commXn[i] = var[i + 1].Id();
        }
        for (int i = 0; i < GSA_NUM - 1; i++) {
            transDep.commGsa[i] = addr[i + 1].Id();
        }
        transDep.commSignal = signal[1].Id();
        transDep.isFuncBlock = isFuncBlock;
        HCCL_INFO(
            "TransDep info: userDevId = %d, dieId = %u, reserveXnId = %u, reserveGsaId = %u, reserveCkeId = %u, "
            "innerDieChannelId = %u, interDieChannelId = %u",
            transDep.userDevId, transDep.dieId, transDep.reserveXnId, transDep.reserveGsaId, transDep.reserveCkeId,
            transDep.reserveChannalId[0], transDep.reserveChannalId[1]);
    }
}; // namespace CcuRep
}; // namespace hcomm

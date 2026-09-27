/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ins_preprocessor.h"
#include "ccu_ctx_mgr.h"
#include "ccu_ins_group.h"
#include "ccu_transport.h"
#include "ccu_communicator.h"
#include "internal_exception.h"
#include "ccu_transport_manager.h"
#include "ccu_registered_ctx_mgr.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_transport_group_manager.h"

namespace Hccl {

constexpr u32 TEMP_MAX_CNTCKE_NUM = 16;

static bool CreateCcuJettys(CcuCommunicator& ccuComm, const std::vector<LinkData>& links, bool& createStatus)
{
    auto ret = ccuComm.GetCcuJettyMgr()->PrepareCreate(links);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        createStatus = false; // 预留处理资源不足回退情况，当前不支持回退
        HCCL_WARNING(
            "[CcuInsPreprocessor][%s] create ccu jettys failed, "
            "ccu resource is unavailable, please check.",
            __func__);
        return false;
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        createStatus = false;
        HCCL_ERROR(
            "[CcuInsPreprocessor][%s] create ccu jettys failed, "
            "unexpected error, please check.",
            __func__);
        THROW<InternalException>("[CreateCcuJettys]create ccu jettys failed, unexpected error, please check.");
    }

    return true;
}

static bool CreateCcuTransports(
    CcuCommunicator& ccuComm, const std::vector<LinkData>& links, bool& createStatus,
    std::vector<CcuTransport*>& transports)
{
    for (auto& link : links) {
        CcuTransport* tansport = nullptr;
        auto ret = ccuComm.GetCcuTransportMgr()->PrepareCreate(link, tansport);
        if (ret == HcclResult::HCCL_E_UNAVAIL) {
            createStatus = false; // 预留处理资源不足回退情况，当前不支持回退
            HCCL_WARNING(
                "[CcuInsPreprocessor][%s] create ccu transports failed, "
                "ccu resource is unavailable, please check.",
                __func__);
            return false;
        }

        if (ret != HcclResult::HCCL_SUCCESS) {
            createStatus = false;
            HCCL_ERROR(
                "[CcuInsPreprocessor][%s] create ccu transports failed, "
                "unexpected error, please check.",
                __func__);
            THROW<InternalException>(
                "[CreateCcuTransports]create ccu transports failed, unexpected error, please check.");
        }
        transports.emplace_back(tansport);
    }

    return true;
}

std::unique_ptr<CcuContext> CcuInsPreprocessor::CreateCcuCtx(const CcuInstruction& ccuInst, bool& createStatus)
{
    HCCL_INFO("[CcuInsPreprocessor::%s] start, ccuInst[%s].", __func__, ccuInst.Describe().c_str());

    std::vector<LinkData> links = ccuInst.GetLinks();

    if (!CreateCcuJettys(ccuComm, links, createStatus)) {
        return nullptr;
    }

    std::vector<CcuTransport*> transports;
    if (!CreateCcuTransports(ccuComm, links, createStatus, transports)) {
        return nullptr;
    }

    // 排序links, 使用RemoteRankId排序, 确保相同的通信模式每次生成的LinkGroup顺序一致, 避免重复创建TransportGroup
    std::sort(links.begin(), links.end(), [](const LinkData& a, const LinkData& b) {
        return a.GetRemoteRankId() < b.GetRemoteRankId();
    });

    vector<LinkInfo> linkInfos{};
    linkInfos.resize(links.size());
    std::transform(links.begin(), links.end(), linkInfos.begin(), [](const LinkData& link) {
        return LinkInfo{link};
    });

    LinkGroup linkGroup{linkInfos};
    u32 cntCkeNum = TEMP_MAX_CNTCKE_NUM; // 临时规避多轮不同算子导致CNTCKE资源不足，待后续正式方案修改
    CcuTransportGroup* transportGrp = ccuComm.GetCcuTransportGrpMgr()->PrepareCreate(linkGroup, cntCkeNum);
    if (transportGrp == nullptr) {
        createStatus = false; // transportGroup当前未适配资源不足场景，需重构
        HCCL_WARNING(
            "[CcuInsPreprocessor::%s] transportGrp alloc resource fail, but fallback, "
            "transports size[%zu],rankGroup size[%zu], cntCkeNum[%u]",
            __func__, transports.size(), linkGroup.GetLinks().size(), cntCkeNum);
        return nullptr;
    }

    HCCL_INFO(
        "[CcuInsPreprocessor::%s] create ccuContext end, transports size[%zu], createStatus[%d], "
        "rankGroup size[%zu], cntCkeNum[%u], createStatus[%d], ccuInst[%s].",
        __func__, transports.size(), createStatus, linkGroup.GetLinks().size(), cntCkeNum, createStatus,
        ccuInst.GetInstType().Describe().c_str());

    std::unique_ptr<CcuCtxArg> ctxArg = ccuInst.GetCtxArg();
    CHECK_NULLPTR(ctxArg, "[CcuInsPreprocessor::CreateCcuCtx] ctxArg is nullptr!");
    return CcuCtxCreatorRegistry::GetInstance().GetCreateFunc(ccuInst.GetInstType())(
        *ctxArg, transports, *transportGrp);
}

void CcuInsPreprocessor::CreateCcuCtxGroup(
    const CcuInstruction& ccuIns, std::unique_ptr<CcuCtxGroup>& ccuCtxGroupPtr, bool& createStatus)
{
    HCCL_INFO("[CcuInsPreprocessor::%s] start, ccuIns[%s].", __func__, ccuIns.Describe().c_str());

    CcuInstType insType = ccuIns.GetInstType();
    if (insType == CcuInstType::CCU_INS_GROUP) {
        const CcuInsGroup& ccuInsGroup = dynamic_cast<const CcuInsGroup&>(ccuIns);
        for (auto& ins : ccuInsGroup.GetCcuInstructions()) {
            std::unique_ptr<CcuContext> ctxPtr = CreateCcuCtx(*ins, createStatus);
            if (ctxPtr == nullptr) {
                return;
            }
            ccuCtxGroupPtr->ctxs.emplace_back(std::move(ctxPtr));
        }
    } else {
        std::unique_ptr<CcuContext> ctxPtr = CreateCcuCtx(ccuIns, createStatus);
        if (ctxPtr == nullptr) {
            return;
        }
        ccuCtxGroupPtr->ctxs.emplace_back(std::move(ctxPtr));
    }

    HCCL_INFO(
        "[CcuInsPreprocessor::%s] create ccuCtx end, insType[%s], ccuCtxGroup ctxs size[%zu], "
        "createStatus[%d].",
        __func__, insType.Describe().c_str(), ccuCtxGroupPtr->ctxs.size(), createStatus);
}

bool CcuInsPreprocessor::CheckCtxTransportStatus(bool resAllocSuccess)
{
    if (!resAllocSuccess) {
        HCCL_WARNING(
            "[CcuInsPreprocessor::%s] CreateCcuCtx alloc local resource fail, ccuCtxGroups"
            " size[%zu], resPackIdxs size[%zu], ctxSignatures size[%zu], insPtrs size[%zu]",
            __func__, ccuCtxGroups.size(), resPackIdxs.size(), ctxSignatures.size(), insPtrs.size());
        return false;
    }

    HCCL_INFO(
        "[CcuInsPreprocessor::%s] CreateCcuCtx success, ccuCtxGroups size[%zu], resPackIdxs"
        " size[%zu], ctxSignatures size[%zu], insPtrs size[%zu]",
        __func__, ccuCtxGroups.size(), resPackIdxs.size(), ctxSignatures.size(), insPtrs.size());
    return true;
}

void CcuInsPreprocessor::InsPreprocess(InsIterator& insIter, u32 resPackIndex, bool isMc2)
{
    CcuInstruction& ccuIns = dynamic_cast<CcuInstruction&>(*insIter);
    // 获取Signature
    CcuCtxSignature ctxSignature = ccuIns.GetCtxSignature();
    if (isMc2) {
        ctxSignature.Append("_Mc2");
    }
    // 获取ResPack
    CcuResPack& ccuResPack = ccuComm.GetCcuResPackMgr()->GetCcuResPack(resPackIndex);
    uintptr_t resPackId = ccuResPack.GetId();

    u64 execId = 0;
    if (!ccuComm.GetRegisteredCcuCtxMgr()->HasRegistered(ctxSignature, resPackId, execId)) {
        needHandShake = true;

        resPackIdxs.emplace_back(resPackIndex);
        insPtrs.emplace_back(insIter);
        ctxSignatures.emplace_back(ctxSignature);
        if ((ccuCtxGroups.find(ctxSignature) != ccuCtxGroups.end())
            && (ccuCtxGroups[ctxSignature].find(resPackIndex) != ccuCtxGroups[ctxSignature].end())) {
            return;
        }

        // 根据CCU扩展指令创建CcuContext实例
        std::unique_ptr<CcuCtxGroup> ccuCtxGroupPtr = std::make_unique<CcuCtxGroup>();
        CHECK_NULLPTR(ccuCtxGroupPtr, "[CcuInsPreprocessor::InsPreprocess] ccuCtxGroupPtr is nullptr!");
        bool createStatus = true;
        CreateCcuCtxGroup(ccuIns, ccuCtxGroupPtr, createStatus);

        // 保存一些中间信息，用于后续的注册回退
        ccuCtxGroups[ctxSignature][resPackIndex] = std::move(ccuCtxGroupPtr);
        resAllocSuccess = createStatus && resAllocSuccess;
        if (!CheckCtxTransportStatus(resAllocSuccess)) {
            return;
        }

        // 调用平台层接口，为CcuContext实例分配本地CCU资源
        HcclResult res
            = CcuCtxMgr::AllocRes(ccuComm.GetDeviceLogicId(), *(ccuCtxGroups[ctxSignature][resPackIndex]), ccuResPack);
        if (res != HcclResult::HCCL_SUCCESS) {
            if (res != HcclResult::HCCL_E_UNAVAIL) {
                THROW<InternalException>("[CcuCtxMgr::AllocRes]AllocRes failed, unexpected error, please check.");
            }
            HCCL_WARNING("[CcuInsPreprocessor::%s] AllocRes failed, ret[%u]", __func__, res);
            resAllocSuccess = false;
        }
    } else {
        ccuIns.SetExecId(execId);
    }

    HCCL_INFO("[CcuInsPreprocessor::%s] end, ccuResPack handles size[%zu].", __func__, ccuResPack.handles.size());
}

void CcuInsPreprocessor::PrepareCcuCtx(std::shared_ptr<InsQueue>& insQueue, bool isMc2)
{
    HCCL_INFO("[CcuInsPreprocessor::%s] start, isMc2[%d].", __func__, isMc2);

    // 对每个从队列分配一个ResPack，对每个queue中每个ins进行预处理(构造ctx且分配本地资源)
    u32 resPackIndex = 0;
    for (auto slaveIter = insQueue->UnConstIterSlaves(); slaveIter.HasNext(); ++slaveIter) {
        for (auto ins = slaveIter->UnConstIter(); ins.HasNext(); ++ins) {
            if (ins->GetType() != InstructionType::CCU_INS) {
                HCCL_INFO(
                    "[CcuInsPreprocessor::%s] slave insQueue ins type[%s] not ccu type.", __func__,
                    ins->GetType().Describe().c_str());
                continue;
            }
            InsPreprocess(ins, resPackIndex, isMc2);
            if (needHandShake && !resAllocSuccess) {
                HCCL_WARNING(
                    "[CcuInsPreprocessor::%s] slave insQueue ins alloc local resource fail, "
                    "resPackIndex[%u], ins[%s].",
                    __func__, resPackIndex, ins->Describe().c_str());
                return;
            }
        }
        resPackIndex++;
    }

    // 主队列分配一个ResPack对每个ins进行预处理(构造ctx且分配本地资源)
    for (auto ins = insQueue->UnConstIter(); ins.HasNext(); ++ins) {
        if (ins->GetType() != InstructionType::CCU_INS) {
            HCCL_INFO(
                "[CcuInsPreprocessor::%s] master insQueue ins type[%s] not ccu type.", __func__,
                ins->GetType().Describe().c_str());
            continue;
        }
        InsPreprocess(ins, resPackIndex, isMc2);
        if (needHandShake && !resAllocSuccess) {
            HCCL_WARNING(
                "[CcuInsPreprocessor::%s] master insQueue ins alloc local resource fail, "
                "resPackIndex[%u], ins[%s].",
                __func__, resPackIndex, ins->Describe().c_str());
            return;
        }
    }

    HCCL_INFO(
        "[CcuInsPreprocessor::%s] prepare ccuContext end, needHandShake[%d], resAllocSuccess[%d].", __func__,
        needHandShake, resAllocSuccess);
}

void CcuInsPreprocessor::Confirm()
{
    HCCL_INFO("[CcuInsPreprocessor::%s] start.", __func__);

    // 建链并交换
    ccuComm.GetCcuResPackMgr()->Confirm();
    ccuComm.GetCcuTransportMgr()->Confirm();
    ccuComm.GetCcuTransportGrpMgr()->Confirm();
    ccuComm.GetCcuJettyMgr()->Confirm();

    HCCL_INFO("[CcuInsPreprocessor::%s] confirm resource end.", __func__);
}

void CcuInsPreprocessor::Fallback()
{
    HCCL_INFO("[CcuInsPreprocessor::%s] start.", __func__);

    // ccu回退流程resPack.handles为空，不需要删除resPack.handles
    ccuComm.GetCcuResPackMgr()->Fallback();
    ccuComm.GetCcuTransportMgr()->Fallback();
    ccuComm.GetCcuTransportGrpMgr()->Fallback();
    ccuComm.GetCcuJettyMgr()->Fallback();

    HCCL_INFO("[CcuInsPreprocessor::%s] fallback resource end.", __func__);
}

void CcuInsPreprocessor::RegisterCtx(bool isFuncBlock)
{
    HCCL_INFO("[CcuInsPreprocessor::%s] start, isFuncBlock[%d].", __func__, isFuncBlock);

    u32 size = insPtrs.size();
    u64 execId = 0;
    for (u32 index = 0; index < size; ++index) {
#ifdef HCCL_ALG_ANALYZER_DAVID
        // 为了算法分析器获取CCU微码序列，需要使用全局变量记录CCU指令和CCU微码，并建立映射关系
        extern Hccl::Instruction* g_ccuIns;
        g_ccuIns = &(*(insPtrs[index]));
#endif
        uintptr_t resPackId = ccuComm.GetCcuResPackMgr()->GetCcuResPack(resPackIdxs[index]).GetId();
        CcuCtxSignature signature = ctxSignatures[index];
        if (!ccuComm.GetRegisteredCcuCtxMgr()->HasRegistered(signature, resPackId, execId)) {
            execId = ccuComm.GetRegisteredCcuCtxMgr()->Register(
                std::move(ccuCtxGroups[signature][resPackIdxs[index]]), signature, resPackId, isFuncBlock);
        }
        CcuInstruction& ccuIns = dynamic_cast<CcuInstruction&>(*insPtrs[index]);
        ccuIns.SetExecId(execId);
    }

    HCCL_INFO("[CcuInsPreprocessor::%s] register ctx end, insPtrs size[%u].", __func__, size);
}

void CcuInsPreprocessor::ClearTmpResRecords()
{
    ccuCtxGroups.clear();
    resPackIdxs.clear();
    ctxSignatures.clear();
    insPtrs.clear();
    needHandShake = false;
    resAllocSuccess = true;
}

void CcuInsPreprocessor::Preprocess(std::shared_ptr<InsQueue>& insQueue, bool isMc2)
{
    HCCL_INFO("[CcuInsPreprocessor::%s] insQueue Preprocess start.", __func__);
    isRollback = false;
    // ccuResPack资源扩充
    u32 insQueueSize = insQueue->SizeOfSlaves() + 1; // 从流个数 + 主流
    ccuComm.GetCcuResPackMgr()->PrepareAlloc(insQueueSize);

    // 对insQ中每个ins,创建CcuContext实例且分配资源
    PrepareCcuCtx(insQueue, isMc2);
    bool isFuncBlock = isMc2; // mc2场景需要将ctxGroup注册为FuncBlock

    HCCL_INFO("[CcuInsPreprocessor::%s] resAllocSuccess is[%d]", __func__, resAllocSuccess);
    if (!resAllocSuccess) {
        HCCL_INFO("[CcuInsPreprocessor::%s] ResAlloc unsuccessful, accelerator fall back.", __func__);
        if (isMc2) {
            // mc2场景，CCU资源不足时不支持回退
            THROW<InternalException>(StringFormat("[CcuInsPreprocessor::%s] Alloc local resource failed", __func__));
        }
        // CCU资源不足时warning
        HCCL_WARNING("[CcuInsPreprocessor::%s] Alloc local resource failed", __func__);
        // 若本地资源申请失败,且非用户显式配置CCU模式，则进行握手回退
        Fallback();
        ClearTmpResRecords();
        ccuComm.AcceleratorFallback();
        isRollback = true;
        return;
    }
    // 若本地资源申请成功, 则进行握手
    // 资源确认
    TRY_CATCH_PROCESS_THROW(
        InternalException, Confirm(), "[CCU Confirm] Confirm Resources Error",
        // 建链失败时，清除临时创建的资源
        ClearTmpResRecords());

    // 注册
    RegisterCtx(isFuncBlock);
    ClearTmpResRecords();
    HCCL_INFO("[CcuInsPreprocessor::%s] insQueue Preprocess end.", __func__);
}

CcuCommunicator* CcuInsPreprocessor::GetCcuComm() { return &ccuComm; }

CcuInsPreprocessor::~CcuInsPreprocessor()
{
    ccuCtxGroups.clear();
    resPackIdxs.clear();
    ctxSignatures.clear();
    insPtrs.clear();
}

HcclResult CcuInsPreprocessor::RecoverCcuTransportCtx(
    const std::vector<LinkData>& links, vector<std::pair<LinkGroup, u32>> linkGroupPair)
{
    HCCL_INFO(
        "[CcuInsPreprocessor::%s] start, links size[%u], linkGroupPair size[%u]", __func__, links.size(),
        linkGroupPair.size());

    CHK_RET(ccuComm.GetCcuJettyMgr()->PrepareCreate(links));

    std::vector<CcuTransport*> transports;
    transports.reserve(links.size());
    for (auto& link : links) {
        CcuTransport* tansport = nullptr;
        CHK_RET(ccuComm.GetCcuTransportMgr()->PrepareCreate(link, tansport));
        transports.emplace_back(tansport);
    }

    // u32 cntCkeNum = TEMP_MAX_CNTCKE_NUM; // 临时规避多轮不同算子导致CNTCKE资源不足，待后续正式方案修改
    for (auto& iter : linkGroupPair) {
        LinkGroup linkGroup = iter.first;
        u32 cntCkeNum = iter.second;
        CcuTransportGroup* transportGrp = ccuComm.GetCcuTransportGrpMgr()->PrepareCreate(linkGroup, cntCkeNum);
        if (transportGrp == nullptr) {
            HCCL_ERROR(
                "[CcuInsPreprocessor::%s] transportGrp alloc resource fail, but fallback, "
                "transports size[%zu],rankGroup size[%zu], cntCkeNum[%u]",
                __func__, transports.size(), linkGroup.GetLinks().size(), cntCkeNum);
            return HCCL_E_INTERNAL;
        }
    }
    HCCL_INFO("[CcuInsPreprocessor::%s] end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CcuInsPreprocessor::RecoverCcuTransportConfirm()
{
    HCCL_INFO("[CcuInsPreprocessor::%s] start.", __func__);

    // 建链并交换
    ccuComm.GetCcuTransportMgr()->RecoverConfirm();
    ccuComm.GetCcuTransportGrpMgr()->Confirm();
    ccuComm.GetCcuJettyMgr()->Confirm();

    HCCL_INFO("[CcuInsPreprocessor::%s] confirm resource end.", __func__);
    return HCCL_SUCCESS;
}

bool CcuInsPreprocessor::IsRollback() const { return isRollback; }

} // namespace Hccl

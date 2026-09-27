/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_context_mgr_imp.h"

#include <unordered_set>

#include "ccu_ctx_mgr.h"
#include "ccu_res_pack_legacy.h"

#include "hccl_common_v2.h"
#include "orion_adapter_rts.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "orion_adapter_hccp.h"
#include "ccu_assist.h"
#include "dev_buffer.h"
#include "hccp_tlv_hdc_manager.h"
#include "config_plf_log_v2.h"

#ifdef HCCL_ALG_ANALYZER_DAVID
#include "instruction.h"
#endif

namespace Hccl {
CtxMgrImp::CtxMgrImp() {}

CtxMgrImp::~CtxMgrImp()
{
    if (initializedFlag_) {
        HCCL_INFO("[CtxMgrImp]~CtxMgrImp: deviceLogicId[%d], free addr[%p]", deviceLogicId_, instructionLoadDevMem_);
        if (instructionLoadDevMem_ != nullptr) {
            DECTOR_TRY_CATCH("CtxMgrImp", HrtFree(instructionLoadDevMem_));
            instructionLoadDevMem_ = nullptr;
        }
        ctxGroupMap_.clear();
        initializedFlag_ = false;
    }
}

CtxMgrImp& CtxMgrImp::GetInstance(s32 deviceLogicId)
{
    static CtxMgrImp contextManager[MAX_MODULE_DEVICE_NUM + 1];

    if (deviceLogicId < 0 || static_cast<u32>(deviceLogicId) > MAX_MODULE_DEVICE_NUM) {
        THROW<CcuApiException>("ProcessSharedResources failed deviceLogicId[%d]", deviceLogicId);
    }

    contextManager[deviceLogicId].deviceLogicId_ = deviceLogicId;
    return contextManager[deviceLogicId];
}

void CtxMgrImp::Init()
{
    std::unique_lock<std::mutex> lock(contextMapMutex_);
    if (initializedFlag_) {
        return;
    }

    initializedFlag_ = true;
    ctxGroupMap_.clear();
}

void CtxMgrImp::Deinit()
{
    std::unique_lock<std::mutex> lock(contextMapMutex_);
    translatorResPack.handles.clear();
    translatorResPack.count = 0;
    initializedFlag_ = false;
    ctxGroupMap_.clear();
    translators.clear();
    referenceMgrs.clear();
}

HcclResult CtxMgrImp::AllocRes(CcuCtxGroup& ctxGroup, CcuResPack& resPack)
{
    std::unique_lock<std::mutex> lock(contextMapMutex_);
    // 初始化ctx
    CtxInit(ctxGroup);

    // 获取ctxGroup使用到的所有dieId
    std::unordered_set<uint16_t> usedDieId;
    for (const auto& ctx : ctxGroup.ctxs) {
        usedDieId.insert(ctx->GetDieId());
    }
    for (auto dieId : usedDieId) {
        // 实例化translator，并且为RefManager和translator申请&绑定资源
        CHK_RET_UNAVAIL(InstantiationTranslator(dieId));
    }

    // 获取通信域当前所持有的资源
    CcuResReq totalRes;
    CHK_RET(GetResPackTotalResNum(resPack, totalRes));

    // 计算本次编排逻辑所需的资源
    CcuResReq resReq = GetCtxGroupResReq(ctxGroup);

    // 比较额外需要的资源
    CHK_RET_UNAVAIL(CompareResAndApplyAsNeeded(totalRes, resReq, resPack));

    // 申请指令空间资源
    CHK_RET_UNAVAIL(AllocInstrRes(ctxGroup));

    // 保存本次编排的资源信息到Ctx中
    resPack.count++;
    SaveResPackToCtx(ctxGroup, resPack);
    HCCL_INFO(
        "[CtxMgrImp:%s]cur resPack count[%u], resHandle[%u], handle size[%u]", __func__, resPack.count, resPack.GetId(),
        resPack.handles.size());
    return HcclResult::HCCL_SUCCESS;
}

// 在外侧调用的UnRegister函数中已加锁，所以当前函数不需要加锁
HcclResult CtxMgrImp::ReleaseRes(CcuCtxGroup& ctxGroup) const
{
    // 获取本次编排Ctx多对应的资源信息
    CcuResPack* resPack = ctxGroup.ctxs[0]->GetResPack();
    CHK_PTR_NULL(resPack);
    HCCL_INFO(
        "[CtxMgrImp:%s]cur resPack count[%u], resHandle[%u], handle size[%u]", __func__, resPack->count,
        resPack->GetId(), resPack->handles.size());
    if (resPack->count > 0) {
        resPack->count--;
    }

    // 释放资源
    if (resPack->count == 0) {
        for (CcuResHandle resHandle : resPack->handles) {
            HCCL_INFO("[CtxMgrImp]ReleaseRes: deviceLogicId[%d], resHandle[%p]", deviceLogicId_, resHandle);
            CHK_RET(CcuDeviceManager::ReleaseResHandle(deviceLogicId_, resHandle));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

uint64_t CtxMgrImp::Register(CcuCtxGroup& ctxGroup, bool isFuncBlock)
{
    // 多通信域场景需要加锁
    std::unique_lock<std::mutex> lock(contextMapMutex_);

    // REP编排虚拟资源和实际物理资源映射关联
    TransRepResToPhyRes(ctxGroup);

    // 构造翻译器,翻译本次编排的ctxGroup
    TransRepSequenceToMicrocode(ctxGroup, isFuncBlock);

    // 保存本次编排的ctxGroup和exeutorId关联关系
    executorId_++; // executorId_ = executorId_ > UINT64_MAX ? 0 : executorId_++;
    ctxGroupMap_[executorId_] = std::move(ctxGroup);
    // 清除referenceMgr中保存的引用关系
    for (auto& referenceMgrMap : referenceMgrs) {
        for (auto& referenceMgr : referenceMgrMap.second) {
            referenceMgr.second->ClearRepReference();
        }
    }
    return executorId_;
}

HcclResult CtxMgrImp::UnRegister(const uint64_t executorId)
{
    std::unique_lock<std::mutex> lock(contextMapMutex_);

    // 校验ctxGroupMap_中是否存在executorId对应的ctxGroup
    CHK_PRT_RET(
        ctxGroupMap_.find(executorId) == ctxGroupMap_.end(),
        HCCL_ERROR("[CtxMgrImp][UnRegister]executorId [%llu] is not exist", executorId), HcclResult::HCCL_E_NOT_FOUND);

    // 释放指令空间
    ReleaseInstrRes(ctxGroupMap_[executorId]);

    // 尝试释放掉ctx对应的handle
    CHK_RET(ReleaseRes(ctxGroupMap_[executorId]));

    ctxGroupMap_.erase(executorId);

    return HcclResult::HCCL_SUCCESS;
}

std::vector<std::vector<CcuTaskParam>> CtxMgrImp::GetTaskParam(CcuTaskArg& ccuTaskArg, const uint64_t executorId)
{
    // 根据executorId获取ctxGroup
    std::unique_lock<std::mutex> lock(contextMapMutex_);

    // 校验ctxGroupMap_中是否存在executorId对应的ctxGroup
    CHK_PRT_RET(
        ctxGroupMap_.find(executorId) == ctxGroupMap_.end(),
        HCCL_ERROR("[CtxMgrImp][GetTaskParam]executorId [%llu] is not exist", executorId),
        std::vector<std::vector<CcuTaskParam>>());

    // 获取每个ctx的taskParam信息
    std::vector<std::vector<CcuTaskParam>> taskParam;
    for (auto& ctx : ctxGroupMap_[executorId].ctxs) {
        std::vector<CcuTaskParam> tmp;
        auto ret = ctx->GeneTaskParam(ccuTaskArg, tmp);
        if (ret != HcclResult::HCCL_SUCCESS) {
            THROW<CcuApiException>("GeneTaskParam is failed. ret[%d]", ret);
        }
        taskParam.push_back(tmp);
    }

    return taskParam;
}

// ctx初始化
void CtxMgrImp::CtxInit(CcuCtxGroup& ctxGroup) const
{
    for (auto& ctx : ctxGroup.ctxs) {
        // 初始化ctx
        CHK_PRT_RET_NULL(ctx->Init(), HCCL_ERROR("Init failed"));
    }
    return;
}

// 申请指令空间资源
HcclResult CtxMgrImp::AllocInstrRes(CcuCtxGroup& ctxGroup) const
{
    for (auto& ctx : ctxGroup.ctxs) {
        ResInfo insInfo(0, 0);
        uint32_t instrCount = ctx->GetInstrCount() + CcuRepTranslator::GetInstrNum();
        CHK_RET_UNAVAIL(CcuDeviceManager::AllocIns(deviceLogicId_, ctx->GetDieId(), instrCount, insInfo));
        HCCL_INFO(
            "[CtxMgrImp]AllocInstrRes: deviceLogicId[%d], dieId[%u], startId[%u], count[%u]", deviceLogicId_,
            ctx->GetDieId(), insInfo.startId, insInfo.num);
        ctx->SetInstrId(insInfo.startId);
    }

    return HcclResult::HCCL_SUCCESS;
}

// 释放指令空间资源
HcclResult CtxMgrImp::ReleaseInstrRes(CcuCtxGroup& ctxGroup) const
{
    for (auto& ctx : ctxGroup.ctxs) {
        ResInfo insInfo(ctx->GetInstrId(), (ctx->GetInstrCount() + CcuRepTranslator::GetInstrNum()));
        HCCL_INFO(
            "[CtxMgrImp]ReleaseInstrRes: deviceLogicId[%d], dieId[%u], startId[%u], count[%u]", deviceLogicId_,
            ctx->GetDieId(), insInfo.startId, insInfo.num);
        CHK_RET(CcuDeviceManager::ReleaseIns(deviceLogicId_, ctx->GetDieId(), insInfo));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrImp::GetResPackTotalResNum(const CcuResPack& resPack, CcuResReq& totalRes) const
{
    CcuResRepository tmpResRepository;

    // 获取通信域当前所持有的资源
    for (CcuResHandle resHandle : resPack.handles) {
        CHK_RET(CcuDeviceManager::GetResource(deviceLogicId_, resHandle, tmpResRepository));

        // 合并获取的所持有的资源信息, 按照类型合并资源总和到totalRes的第0个vector中
        for (u32 i = 0; i < MAX_CCU_IODIE_NUM; i++) {
            totalRes.msReq[i] += GetResTotalNum(tmpResRepository.ms[i]);
            totalRes.blockMsReq[i] += GetResTotalNum(tmpResRepository.blockMs[i]);
            totalRes.ckeReq[i] += GetResTotalNum(tmpResRepository.cke[i]);
            totalRes.blockCkeReq[i] += GetResTotalNum(tmpResRepository.blockCke[i]);
            totalRes.loopEngineReq[i] += GetResTotalNum(tmpResRepository.loopEngine[i]);
            totalRes.blockLoopEngineReq[i] += GetResTotalNum(tmpResRepository.blockLoopEngine[i]);
            totalRes.gsaReq[i] += GetResTotalNum(tmpResRepository.gsa[i]);
            totalRes.blockGsaReq[i] += GetResTotalNum(tmpResRepository.blockGsa[i]);
            totalRes.xnReq[i] += GetResTotalNum(tmpResRepository.xn[i]);
            totalRes.blockXnReq[i] += GetResTotalNum(tmpResRepository.blockXn[i]);
            totalRes.missionReq.req[i] += GetResTotalNum(tmpResRepository.mission.mission[i]);
        }
    }

    DumpResReqInfo(totalRes);
    HCCL_INFO("GetResPackTotalResNum:dumpInfos success.");
    return HcclResult::HCCL_SUCCESS;
}

CcuResReq CtxMgrImp::GetCtxGroupResReq(CcuCtxGroup& ctxGroup) const
{
    CcuResReq totalResReq;

    std::unordered_set<uint16_t> usedDieId; // CCUCtxGroup使用到的所有dieId

    // 获取CCUCtxGroup所有ctx资源诉求
    for (auto& ctx : ctxGroup.ctxs) {
        auto dieId = ctx->GetDieId();
        usedDieId.insert(dieId);

        CcuResReq tmpResReq = ctx->GetResourceRequest();
        MergeCcuResReq(totalResReq, tmpResReq);
    }

    DumpResReqInfo(totalResReq);
    HCCL_INFO("CtxGroupResReq:dumpInfos success.");

    return totalResReq;
}

void CtxMgrImp::MergeCcuResReq(CcuResReq& resReqA, const CcuResReq& resReqB) const
{
    // 合并获取的所持有的资源信息, 按照类型合并资源总和到totalRes的第0个vector中
    for (u32 i = 0; i < MAX_CCU_IODIE_NUM; i++) {
        resReqA.msReq[i] += resReqB.msReq[i];
        resReqA.blockMsReq[i] += resReqB.blockMsReq[i];
        resReqA.ckeReq[i] += resReqB.ckeReq[i];
        resReqA.blockCkeReq[i] += resReqB.blockCkeReq[i];
        resReqA.loopEngineReq[i] += resReqB.loopEngineReq[i];
        resReqA.blockLoopEngineReq[i] += resReqB.blockLoopEngineReq[i];
        resReqA.gsaReq[i] += resReqB.gsaReq[i];
        resReqA.blockGsaReq[i] += resReqB.blockGsaReq[i];
        resReqA.xnReq[i] += resReqB.xnReq[i];
        resReqA.blockXnReq[i] += resReqB.blockXnReq[i];
        resReqA.missionReq.req[i] += resReqB.missionReq.req[i];

        if (resReqB.missionReq.req[i] > 0) {
            resReqA.missionReq.reqType = resReqB.missionReq.reqType;
        }
    }
    return;
}

HcclResult
CtxMgrImp::CompareResAndApplyAsNeeded(const CcuResReq& totalRes, const CcuResReq& resReq, CcuResPack& resPack) const
{
    // 比较额外需要的资源
    bool isNeedAlloc = false;
    CcuResReq needResReq;

    for (u32 i = 0; i < MAX_CCU_IODIE_NUM; i++) {
        needResReq.msReq[i] = GetReqResNum(resReq.msReq[i], totalRes.msReq[i]);
        needResReq.blockMsReq[i] = GetReqResNum(resReq.blockMsReq[i], totalRes.blockMsReq[i]);
        needResReq.ckeReq[i] = GetReqResNum(resReq.ckeReq[i], totalRes.ckeReq[i]);
        needResReq.blockCkeReq[i] = GetReqResNum(resReq.blockCkeReq[i], totalRes.blockCkeReq[i]);
        needResReq.loopEngineReq[i] = GetReqResNum(resReq.loopEngineReq[i], totalRes.loopEngineReq[i]);
        needResReq.blockLoopEngineReq[i] = GetReqResNum(resReq.blockLoopEngineReq[i], totalRes.blockLoopEngineReq[i]);
        needResReq.gsaReq[i] = GetReqResNum(resReq.gsaReq[i], totalRes.gsaReq[i]);
        needResReq.blockGsaReq[i] = GetReqResNum(resReq.blockGsaReq[i], totalRes.blockGsaReq[i]);
        needResReq.xnReq[i] = GetReqResNum(resReq.xnReq[i], totalRes.xnReq[i]);
        needResReq.blockXnReq[i] = GetReqResNum(resReq.blockXnReq[i], totalRes.blockXnReq[i]);
        needResReq.missionReq.req[i] = GetReqResNum(resReq.missionReq.req[i], totalRes.missionReq.req[i]);

        if (needResReq.missionReq.req[i] > 0) {
            needResReq.missionReq.reqType = resReq.missionReq.reqType;
        }

        if (needResReq.msReq[i] != 0 || needResReq.blockMsReq[i] != 0 || needResReq.ckeReq[i] != 0
            || needResReq.blockCkeReq[i] != 0 || needResReq.loopEngineReq[i] != 0
            || needResReq.blockLoopEngineReq[i] != 0 || needResReq.gsaReq[i] != 0 || needResReq.blockGsaReq[i] != 0
            || needResReq.xnReq[i] != 0 || needResReq.blockXnReq[i] != 0 || needResReq.missionReq.req[i] != 0) {
            isNeedAlloc = true;
        }
    }

    if (isNeedAlloc) {
        // 申请额外资源
        CcuResHandle handle;
        DumpResReqInfo(needResReq);

        CHK_RET_UNAVAIL(CcuDeviceManager::AllocResHandle(deviceLogicId_, needResReq, handle));
        // 将申请的资源保存到resPack中
        resPack.handles.push_back(handle);

        HCCL_INFO("ApplyAsNeeded:dumpInfos success deviceLogicId[%d] handle[%p].", deviceLogicId_, handle);
    }

    return HcclResult::HCCL_SUCCESS;
}

void CtxMgrImp::SaveResPackToCtx(CcuCtxGroup& ctxGroup, CcuResPack& resPack) const
{
    for (auto& ctx : ctxGroup.ctxs) {
        ctx->SetResPack(resPack);
    }
    return;
}

HcclResult CtxMgrImp::InstantiationTranslator(uint16_t dieId)
{
    if (translators.find(dieId) != translators.end()) {
        return HcclResult::HCCL_SUCCESS;
    }

    std::array<uint16_t, MAX_CCU_IODIE_NUM> tmpChannelId{};
    uint32_t chaneelId = 0;
    // 获取innerDieChannelId
    auto ret = CcuDeviceManager::GetLoopChannelId(deviceLogicId_, dieId, dieId, chaneelId);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>(
            "Failed to get inner die channel id. deviceLogicId = %d, dieId = %u, ret = %d", deviceLogicId_, dieId, ret);
    }
    tmpChannelId[0] = chaneelId;
    // 获取interDieChannelId
    uint8_t dstDieId = ((dieId == 0) ? 1 : 0);
    ret = CcuDeviceManager::GetLoopChannelId(deviceLogicId_, dieId, dstDieId, chaneelId);
    if (ret != HcclResult::HCCL_SUCCESS) {
        // 当前验证环境为单die环境，获取die间ChannelId会失败，打印WARNING日志。
        HCCL_WARNING(
            "Failed to get inter die channel id. deviceLogicId = %d, srcDieId = %u, dstDieId = %u, ret = %d",
            deviceLogicId_, dieId, dstDieId, ret);
    }
    tmpChannelId[1] = chaneelId;
    // 获取ccu token信息
    uint64_t tokenId = 0;
    uint64_t tokenValue = 0;
    ret = CcuDeviceManager::GetCcuResourceSpaceTokenInfoForLocal(deviceLogicId_, dieId, tokenId, tokenValue);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>(
            "Failed to get ccu resource space token info. deviceLogicId = %d, dieId = %u, ret = %d", deviceLogicId_,
            dieId, ret);
    }
    std::pair<uint64_t, uint64_t> ccuTokenInfo(tokenId, tokenValue);
    CcuResReq totalResReq;

    // 先获取hbm token，避免创建mission时循环获取
    DevBuffer tmpDevMem{1};
    auto hbmTokenInfo = GetTokenInfo(tmpDevMem.GetAddr(), 1);

    // 实例化CcuRepReferenceManager和CcuRepTranslator，并为CcuRepReferenceManager绑定物理资源
    for (uint32_t i = 0; i < 16; i++) { // mgr有16个
        referenceMgrs[dieId][i] = std::make_shared<CcuRepReferenceManager>(dieId);
        translators[dieId][i] = std::make_shared<CcuRepTranslator>(
            deviceLogicId_, dieId, referenceMgrs[dieId][i], tmpChannelId, ccuTokenInfo, hbmTokenInfo);

        // 统计&合并refManager和translator所有资源REQ
        auto refMangerResReq = CcuRep::CcuRepReferenceManager::GetResReq(dieId);
        auto transLatorResReq = CcuRep::CcuRepTranslator::GetResReq(dieId);
        MergeCcuResReq(totalResReq, refMangerResReq);
        MergeCcuResReq(totalResReq, transLatorResReq);
    }

    DumpResReqInfo(totalResReq);

    // 为refManager和translator申请物理资源
    CcuResHandle handle;
    CHK_RET_UNAVAIL(CcuDeviceManager::AllocResHandle(deviceLogicId_, totalResReq, handle));
    translatorResPack.handles.push_back(handle);

    CcuRepResource translatorRepRes;
    for (uint32_t i = 0; i < 16; i++) { // mgr有16个
        referenceMgrs[dieId][i]->GetRes(translatorRepRes);
        translators[dieId][i]->GetRes(translatorRepRes);
    }

    CcuResRepository totalResRepository;
    CHK_RET(GetResPackTotalResRepository(translatorResPack, totalResRepository));
    // 将ctx中的rep虚拟资源按类型进行和CCU物理资源映射
    ResetRepResourceToResRepository(translatorRepRes, totalResRepository);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrImp::TransRepResToPhyRes(CcuCtxGroup& ctxGroup) const
{
    // 获取ctxGroup中CCU物理资源句柄
    CcuResPack* resPack = ctxGroup.ctxs[0]->GetResPack();
    CHK_PTR_NULL(resPack);

    // 获取通信域当前所持有的资源
    CcuResRepository totalResRepository;
    CHK_RET(GetResPackTotalResRepository(*resPack, totalResRepository));

    // 遍历ctxGroup, 将每个ctx的虚拟资源进行合并
    CcuRepResource totalRepRes = GetTotalCcuRepResource(ctxGroup);

    // 将ctx中的rep虚拟资源按类型进行和CCU物理资源映射
    ResetRepResourceToResRepository(totalRepRes, totalResRepository);

    // 针对跨ctx的资源进行特殊映射处理
    ProcessInterCtxRes(ctxGroup);

    // missionId、mission key赋值给ctx
    CHK_RET(SaveCtxMissionInfo(ctxGroup, totalResRepository.mission.mission));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CtxMgrImp::GetResPackTotalResRepository(const CcuResPack& resPack, CcuResRepository& totalRes) const
{
    CcuResRepository tmpResRepository;

    // 获取通信域当前所持有的资源
    for (CcuResHandle resHandle : resPack.handles) {
        CHK_RET(CcuDeviceManager::GetResource(deviceLogicId_, resHandle, tmpResRepository));

        // 合并获取的所持有的资源信息, 按照类型合并资源总和到totalRes中
        for (u32 i = 0; i < MAX_CCU_IODIE_NUM; i++) {
            ExpandResInfo(totalRes.ms[i], tmpResRepository.ms[i]);
            ExpandResInfo(totalRes.blockMs[i], tmpResRepository.blockMs[i]);
            ExpandResInfo(totalRes.loopEngine[i], tmpResRepository.loopEngine[i]);
            ExpandResInfo(totalRes.blockLoopEngine[i], tmpResRepository.blockLoopEngine[i]);
            ExpandResInfo(totalRes.cke[i], tmpResRepository.cke[i]);
            ExpandResInfo(totalRes.blockCke[i], tmpResRepository.blockCke[i]);
            ExpandResInfo(totalRes.gsa[i], tmpResRepository.gsa[i]);
            ExpandResInfo(totalRes.blockGsa[i], tmpResRepository.blockGsa[i]);
            ExpandResInfo(totalRes.xn[i], tmpResRepository.xn[i]);
            ExpandResInfo(totalRes.blockXn[i], tmpResRepository.blockXn[i]);
            ExpandResInfo(totalRes.mission.mission[i], tmpResRepository.mission.mission[i]);
        }
        DumpResRepositoryInfo(totalRes);
        HCCL_INFO(
            "GetResPackTotalResRepository:dumpInfos success deviceLogicId[%d] resHandle[%p].", deviceLogicId_,
            resHandle);
    }
    return HcclResult::HCCL_SUCCESS;
}

CcuRepResource CtxMgrImp::GetTotalCcuRepResource(CcuCtxGroup& ctxGroup) const
{
    CcuRepResource totalRepRes;

    // 遍历ctxGroup, 将每个ctx的虚拟资源进行合并
    for (auto& ctx : ctxGroup.ctxs) {
        // 获取ctx的虚拟资源
        CcuRepResource tmpRepRes = ctx->GetResource();
        MergeCtxRepResource(totalRepRes, tmpRepRes);
    }

    return totalRepRes;
}

void CtxMgrImp::MergeCtxRepResource(CcuRepResource& repResourceA, CcuRepResource& repResourceB) const
{
    for (u32 i = 0; i < MAX_CCU_IODIE_NUM; i++) {
        // 合并repRes
        repResourceA.ccubuffers[i].insert(
            repResourceA.ccubuffers[i].end(), repResourceB.ccubuffers[i].begin(), repResourceB.ccubuffers[i].end());

        repResourceA.blockCcubuffers[i].insert(
            repResourceA.blockCcubuffers[i].end(), repResourceB.blockCcubuffers[i].begin(),
            repResourceB.blockCcubuffers[i].end());

        repResourceA.executor[i].insert(
            repResourceA.executor[i].end(), repResourceB.executor[i].begin(), repResourceB.executor[i].end());

        repResourceA.blockExecutor[i].insert(
            repResourceA.blockExecutor[i].end(), repResourceB.blockExecutor[i].begin(),
            repResourceB.blockExecutor[i].end());

        repResourceA.maskSignal[i].insert(
            repResourceA.maskSignal[i].end(), repResourceB.maskSignal[i].begin(), repResourceB.maskSignal[i].end());

        repResourceA.blockMaskSignal[i].insert(
            repResourceA.blockMaskSignal[i].end(), repResourceB.blockMaskSignal[i].begin(),
            repResourceB.blockMaskSignal[i].end());

        repResourceA.address[i].insert(
            repResourceA.address[i].end(), repResourceB.address[i].begin(), repResourceB.address[i].end());

        repResourceA.blockAddress[i].insert(
            repResourceA.blockAddress[i].end(), repResourceB.blockAddress[i].begin(),
            repResourceB.blockAddress[i].end());

        repResourceA.variable[i].insert(
            repResourceA.variable[i].end(), repResourceB.variable[i].begin(), repResourceB.variable[i].end());

        repResourceA.continuousVariable[i].insert(
            repResourceA.continuousVariable[i].end(), repResourceB.continuousVariable[i].begin(),
            repResourceB.continuousVariable[i].end());
    }
    return;
}

template <typename T1, typename T2>
void CtxMgrImp::ResetRepResourceTemplate(std::vector<T1>& resource, const std::vector<T2>& repository) const
{
    if (resource.size() > repository.size()) {
        THROW<CcuApiException>(
            "[CtxMgrImp][ResetRepResourceTemplate]resource size[%u] bigger repository size[%u] typeid[%s]",
            resource.size(), repository.size(), typeid(T1).name());
        return;
    }

    for (uint32_t j = 0; j < resource.size(); j++) {
        resource[j].Reset(repository[j].startId);
    }
}

void CtxMgrImp::ResetRepResourceToResRepository(
    CcuRepResource& totalRepRes, const CcuResRepository& totalResRepository) const
{
    // 遍历translatorRepRes, 将每个rep的虚拟资源翻译到实际物理资源上
    for (u32 i = 0; i < MAX_CCU_IODIE_NUM; i++) {
        ResetRepResourceTemplate(totalRepRes.ccubuffers[i], totalResRepository.ms[i]);
        ResetRepResourceTemplate(totalRepRes.blockCcubuffers[i], totalResRepository.blockMs[i]);
        ResetRepResourceTemplate(totalRepRes.executor[i], totalResRepository.loopEngine[i]);
        ResetRepResourceTemplate(totalRepRes.blockExecutor[i], totalResRepository.blockLoopEngine[i]);
        ResetRepResourceTemplate(totalRepRes.maskSignal[i], totalResRepository.cke[i]);
        ResetRepResourceTemplate(totalRepRes.blockMaskSignal[i], totalResRepository.blockCke[i]);
        ResetRepResourceTemplate(totalRepRes.address[i], totalResRepository.gsa[i]);
        ResetRepResourceTemplate(totalRepRes.blockAddress[i], totalResRepository.blockGsa[i]);
        ResetRepResourceTemplate(totalRepRes.variable[i], totalResRepository.xn[i]);
        ResetRepResourceTemplate(totalRepRes.continuousVariable[i], totalResRepository.blockXn[i]);
    }
}

template <typename T>
void CtxMgrImp::ProcessSharedResources(
    std::unordered_map<std::string, T>& resources, std::vector<std::unordered_map<std::string, T>>& exportedResources,
    uint32_t i) const
{
    for (auto& res : resources) {
        uint32_t j;
        for (j = 0; j < exportedResources.size(); j++) {
            if (i != j) {
                auto exportedRes = exportedResources[j].find(res.first);
                if (exportedRes != exportedResources[j].end()) {
                    res.second.Reset(exportedRes->second.Id(), exportedRes->second.DieId());
                    break;
                }
            }
        }

        if (j >= exportedResources.size()) {
            THROW<CcuApiException>("ProcessSharedResources failed tag[%s]", res.first.c_str());
        }
    }
}

void CtxMgrImp::ProcessInterCtxRes(CcuCtxGroup& ctxGroup) const
{
    // 针对跨ctx的资源进行特殊映射处理
    for (uint32_t i = 0; i < ctxGroup.ctxs.size(); i++) {
        // 获取当前ctx中导入的ccuShrRes
        CcuSharedResource importRes = ctxGroup.ctxs[i]->GetImportRes();

        // 提前计算所有ctx的exportRes，以减少不必要的重复计算
        std::vector<std::unordered_map<std::string, CcuRep::Variable>> exportVarResList(ctxGroup.ctxs.size());
        std::vector<std::unordered_map<std::string, CcuRep::MaskSignal>> exportSigResList(ctxGroup.ctxs.size());

        for (uint32_t j = 0; j < ctxGroup.ctxs.size(); j++) {
            if (i != j) {
                auto exportRes = ctxGroup.ctxs[j]->GetExportRes();
                exportVarResList[j] = exportRes.sharedVar;
                exportSigResList[j] = exportRes.sharedSig;
            }
        }

        // 处理sharedVar和sharedSig
        ProcessSharedResources(importRes.sharedVar, exportVarResList, i);
        ProcessSharedResources(importRes.sharedSig, exportSigResList, i);
    }
}

HcclResult
CtxMgrImp::SaveCtxMissionInfo(CcuCtxGroup& ctxGroup, array<vector<ResInfo>, MAX_CCU_IODIE_NUM>& missionId) const
{
    for (auto& ctx : ctxGroup.ctxs) {
        auto dieId = ctx->GetDieId();
        uint32_t missionKey;
        CHK_RET(CcuDeviceManager::GetMissionKey(deviceLogicId_, dieId, missionKey));

        HCCL_INFO("[SaveCtxMissionInfo]GetMissionKey:deviceLogicId[%d] dieId[%u]", deviceLogicId_, dieId);

        ctx->SetMissionKey(missionKey);
        // 从missionId中获取一个元素并从missionId中删除
        ctx->SetMissionId(missionId[dieId].back().startId);
        missionId[dieId].pop_back();
    }

    return HcclResult::HCCL_SUCCESS;
}

void CtxMgrImp::TransRepSequenceToMicrocode(CcuCtxGroup& ctxGroup, bool isFuncBlock)
{
    for (auto& ctx : ctxGroup.ctxs) {
        auto dieId = ctx->GetDieId();
        auto missionId = ctx->GetMissionId();
        // 翻译本ctx的REP序列
        auto instrInfo
            = translators[dieId][missionId]->Translate(ctx->GetRepSequence(), ctx->GetInstrId(), isFuncBlock);

#ifdef HCCL_ALG_ANALYZER_DAVID
        // 建立CCU指令和微码的映射关系
        extern std::map<const Hccl::Instruction*, std::vector<Hccl::CcuRep::CcuInstrInfo>> g_ccuIns2MicroCode;
        extern Hccl::Instruction* g_ccuIns;
        g_ccuIns2MicroCode[g_ccuIns].push_back(instrInfo);
#endif

        // 保存翻译后的指令信息
        ctx->SetCcuInstrInfo(instrInfo);

        // 加载指令到CCU指令空间
        LoadInstruction(instrInfo, dieId);
    }
    return;
}

void CtxMgrImp::LoadInstruction(CcuRep::CcuInstrInfo& instrInfo, uint32_t dieId)
{
    uint64_t instrInfoSize = instrInfo.instrVec.size() * sizeof(CcuInstr);

    if (!instructionLoadDevMem_) {
        uint32_t instrNum = 0;
        if (CcuDeviceManager::GetInstructionNum(deviceLogicId_, 0, instrNum) != HcclResult::HCCL_SUCCESS) {
            THROW<CcuApiException>("CtxMgrImp::Init failed");
        }
        HCCL_INFO("[CtxMgrImp]LoadInstruction: deviceLogicId[%d], instrNum[%u]", deviceLogicId_, instrNum);
        // rt接口申请device内存
        instructionLoadDevMem_ = HrtMalloc(instrNum * sizeof(CcuInstr), static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    }

    // rt接口memcpySync
    HrtMemcpy(
        instructionLoadDevMem_, instrInfoSize, instrInfo.instrVec.data(), instrInfoSize, RT_MEMCPY_HOST_TO_DEVICE);

    auto tlvHandle = HccpTlvHdcManager::GetInstance().GetTlvHandle(deviceLogicId_);
    CHECK_NULLPTR(
        tlvHandle, StringFormat("[CtxMgrImp][%s] tlvHandle is nullptr, deviceLogicId[%d]", __func__, deviceLogicId_));

    struct CustomChannelInfoIn inBuff;
    struct CustomChannelInfoOut outBuff;

    CcuDataTypeUnion tmp;
    tmp.insinfo.resourceAddr = reinterpret_cast<uint64_t>(instructionLoadDevMem_);

    // 设置操作码和通道数据
    inBuff.op = CcuOpcodeType::CCU_U_OP_SET_INSTRUCTION;
    inBuff.offsetStartIdx = instrInfo.startInstrId;
    inBuff.data.dataInfo.udieIdx = dieId;
    inBuff.data.dataInfo.dataArraySize = 1;
    inBuff.data.dataInfo.dataLen = instrInfoSize;

    // 复制通道数据
    (void)memcpy_s(inBuff.data.dataInfo.dataArray, sizeof(CcuDataTypeUnion), &tmp, sizeof(CcuDataTypeUnion));

    HrtRaTlvRequestForCustomChannel(
        tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, static_cast<void*>(&inBuff), static_cast<void*>(&outBuff));

    HCCL_RUN_INFO(
        "Entry-LoadInstruction: load instruction success startInstrId[%u] instrCount[%u]", instrInfo.startInstrId,
        instrInfo.instrCount);

    return;
}

void CtxMgrImp::DumpResReqInfo(const CcuResReq& totalRes) const
{
    for (uint32_t i = 0; i < MAX_CCU_IODIE_NUM; i++) {
        if (totalRes.msReq[i] != 0 || totalRes.blockMsReq[i] != 0 || totalRes.ckeReq[i] != 0
            || totalRes.blockCkeReq[i] != 0 || totalRes.loopEngineReq[i] != 0 || totalRes.blockLoopEngineReq[i] != 0
            || totalRes.gsaReq[i] != 0 || totalRes.blockGsaReq[i] != 0 || totalRes.xnReq[i] != 0
            || totalRes.blockXnReq[i] != 0 || totalRes.missionReq.req[i] != 0) {
            HCCL_INFO(
                "DumpResReqInfo: dieId[%u], msReq[%u], blockMsReq[%u], ckeReq[%u], blockCkeReq[%u], "
                "loopEngineReq[%u], blockLoopEngineReq[%u], gsaReq[%u], blockGsaReq[%u], "
                "xnReq[%u], blockXnReq[%u], missionReq[%u]",
                i, totalRes.msReq[i], totalRes.blockMsReq[i], totalRes.ckeReq[i], totalRes.blockCkeReq[i],
                totalRes.loopEngineReq[i], totalRes.blockLoopEngineReq[i], totalRes.gsaReq[i], totalRes.blockGsaReq[i],
                totalRes.xnReq[i], totalRes.blockXnReq[i], totalRes.missionReq.req[i]);
        }
    }
}

void CtxMgrImp::DumpResRepositoryInfo(const CcuResRepository& resRepo) const
{
    for (uint32_t i = 0; i < MAX_CCU_IODIE_NUM; i++) {
        if (resRepo.ms[i].size() != 0 || resRepo.blockMs[i].size() != 0 || resRepo.cke[i].size() != 0
            || resRepo.blockCke[i].size() != 0 || resRepo.loopEngine[i].size() != 0
            || resRepo.blockLoopEngine[i].size() != 0 || resRepo.gsa[i].size() != 0 || resRepo.blockGsa[i].size() != 0
            || resRepo.xn[i].size() != 0 || resRepo.blockXn[i].size() != 0 || resRepo.mission.mission[i].size() != 0) {
            HCCL_INFO(
                "DumpResRepository: dieId[%u], ms size[%u], blockMs size[%u], cke size[%u], blockCke size[%u], "
                "loopEngine size[%u], blockLoopEngine size[%u], gsa size[%u], blockGsa size[%u], xn size[%u], "
                "block xn size[%u], mission size[%u]",
                i, resRepo.ms[i].size(), resRepo.blockMs[i].size(), resRepo.cke[i].size(), resRepo.blockCke[i].size(),
                resRepo.loopEngine[i].size(), resRepo.blockLoopEngine[i].size(), resRepo.gsa[i].size(),
                resRepo.blockGsa[i].size(), resRepo.xn[i].size(), resRepo.blockXn[i].size(),
                resRepo.mission.mission[i].size());
        }
    }
}

std::vector<std::vector<CcuProfilingInfo>> CtxMgrImp::GetProfilingInfo(CcuTaskArg& ccuTaskArg, const uint64_t entityId)
{
    // 根据entityId获取ctxGroup
    std::unique_lock<std::mutex> lock(contextMapMutex_);

    // 校验ctxGroupMap_中是否存在entityId对应的ctxGroup
    CHK_PRT_RET(
        ctxGroupMap_.find(entityId) == ctxGroupMap_.end(),
        HCCL_ERROR("[CtxMgrImp][GetProfilingInfo]entityId [%llu] is not exist", entityId),
        std::vector<std::vector<CcuProfilingInfo>>());

    // 获取每个ctx的profiling信息
    std::vector<std::vector<CcuProfilingInfo>> ccuProfilingInfo;
    for (auto& ctx : ctxGroupMap_[entityId].ctxs) {
        std::vector<CcuProfilingInfo> tmp;
        auto ret = ctx->GetCcuProfilingInfo(ccuTaskArg, tmp);
        if (ret != HcclResult::HCCL_SUCCESS) {
            THROW<CcuApiException>("GetCcuProfilingInfo is failed. ret[%d]", ret);
        }
        ccuProfilingInfo.push_back(tmp);
    }

    return ccuProfilingInfo;
}

CcuContext* CtxMgrImp::GetCtx(uint64_t executorId, uint32_t dieId, uint32_t missionId)
{
    std::unique_lock<std::mutex> lock(contextMapMutex_);

    if (ctxGroupMap_.find(executorId) == ctxGroupMap_.end()) {
        HCCL_ERROR("[CtxMgrImp][GetCtx] executorId [%llu] is not exist", executorId);
        return nullptr;
    }

    for (auto& ctx : ctxGroupMap_[executorId].ctxs) {
        if (ctx->GetDieId() == dieId && ctx->GetMissionId() == missionId) {
            return ctx.get();
        }
    }

    HCCL_ERROR("[CtxMgrImp][GetCtx] CcuContext is not exist, dieId[%u], missionId[%u]", dieId, missionId);
    return nullptr;
}
}; // namespace Hccl

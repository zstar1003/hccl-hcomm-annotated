/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res_batch_allocator.h"

#include <array>
#include <memory>
#include <utility>
#include <iterator>
#include <algorithm>

#include "hccl_common.h"

#include "ccu_comp.h"
#include "ccu_res_specs.h"
#include "ccu_res_type_converter.h"
#include "ccu_device_res.h"
#include "ccu_rep_reference_manager_v1.h"
#include "ccu_rep_reference_manager.h"

namespace hcomm {

constexpr uint32_t REQ_RES_TYPE_NUM = 11;
constexpr uint32_t BLOCK_RES_TYPE_NUM = 5;
constexpr uint32_t CONS_RES_TYPE_NUM = 1;
constexpr uint32_t DISCRETE_RES_TYPE_NUM = 4;
constexpr uint32_t NON_BLOCK_TYPE_NUM = CONS_RES_TYPE_NUM + DISCRETE_RES_TYPE_NUM;
constexpr uint32_t BLOCK_SIZE_MS_AX_DIE0 = 128;
constexpr uint32_t CCUA_NUM = 4;

constexpr uint32_t CCU_REPREFMGR_NEW_XN_NUM
    = hcomm::CcuRep::FUNC_ARG_MAX + hcomm::CcuRep::FUNC_ARG_MAX + 1 + hcomm::CcuRep::FUNC_NEST_MAX + 1;
constexpr uint32_t CCU_REPREFMGR_LEGACY_XN_NUM
    = Hccl::CcuRep::FUNC_IN_MAX + Hccl::CcuRep::FUNC_OUT_MAX + 1 + Hccl::CcuRep::FUNC_NEST_MAX + 1;
constexpr uint32_t CCU_REP_TRANSLATOR_GSA_NUM = 3;
constexpr uint32_t CCU_REP_TRANSLATOR_CKE_NUM = 2;
constexpr uint32_t CCU_REP_TRANSLATOR_XN_NUM = 4;
// 建链预留数量 + 开源+legacy的CcuRepTranslator预留数量
constexpr uint32_t RESERVED_DISCRETE_CKE_NUM = 4 * 128 + (CCU_REP_TRANSLATOR_CKE_NUM * 16) * 2;
// 建链预留数量 + 开源+legacy的CcuRepReferenceManager预留数量 + 开源+legacy的CcuRepTranslator预留数量
constexpr uint32_t RESERVED_DISCRETE_XN_NUM
    = 4 * 128 + CCU_REPREFMGR_LEGACY_XN_NUM * 16 + CCU_REPREFMGR_NEW_XN_NUM * 16 + (CCU_REP_TRANSLATOR_XN_NUM * 16) * 2;
// 开源+legacy的CcuRepTranslator预留数量
constexpr uint32_t RESERVED_DISCRETE_GSA_NUM = (CCU_REP_TRANSLATOR_GSA_NUM * 16) * 2;

CcuResBatchAllocator& CcuResBatchAllocator::GetInstance(const int32_t userDevId)
{
    static CcuResBatchAllocator ccuResBatchAllocator[MAX_MODULE_DEVICE_NUM + 1];
    int32_t validUserDevId = userDevId;
    if (validUserDevId < 0 || static_cast<uint32_t>(validUserDevId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[CcuResBatchAllocator][%s] use the backup device, userDevId[%d] "
            "should be less than %u.",
            __func__, validUserDevId, MAX_MODULE_DEVICE_NUM);
        validUserDevId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }
    ccuResBatchAllocator[validUserDevId].userDevId_ = validUserDevId;
    return ccuResBatchAllocator[validUserDevId];
}

HcclResult CcuResBatchAllocator::Init()
{
    if (initFlag_) {
        return HcclResult::HCCL_SUCCESS;
    }

    dieEnableFlags_ = CcuComponent::GetInstance(userDevId_).GetDieEnableFlags();
    if (!dieEnableFlags_[0] && !dieEnableFlags_[1]) {
        HCCL_WARNING(
            "[CcuResBatchAllocator][%s] failed but passed, "
            "userDevId[%d] no usable die.",
            __func__, userDevId_);
        return HcclResult::HCCL_E_UNAVAIL;
    }

    auto ret = PreAllocBlockRes();
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING(
            "[CcuResBatchAllocator][%s] pre alloc block res failed but passed, "
            "some sources are not enough, userDevId[%d].",
            __func__, userDevId_);
        return ret;
    }
    CHK_RET(ret);

    ret = missionMgr_.PreAlloc(userDevId_, resStrategies_[0].missionNum, dieEnableFlags_);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING(
            "[CcuResBatchAllocator][%s] pre alloc mission res failed but passed, "
            "some sources are not enough, userDevId[%d].",
            __func__, userDevId_);
        return ret;
    }
    CHK_RET(ret);

    initFlag_ = true;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResBatchAllocator::Deinit()
{
    missionMgr_.Reset();
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        resBlocks_[dieId].clear();
    }

    handleMap_.clear();
    initFlag_ = false;
    return HcclResult::HCCL_SUCCESS;
}

static CcuResBlockNums GetPreAllocatedMaxBlockNums(
    const uint32_t userDevId, const uint8_t dieId,
    const std::array<CcuBlockResStrategy, CCU_MAX_IODIE_NUM>& resStrategies)
{
    CcuResBlockNums blockNums{};

    CcuResSpecifications& ccuResSepcs = CcuResSpecifications::GetInstance(userDevId);

    uint32_t loopNum = 0;
    (void)ccuResSepcs.GetLoopEngineNum(dieId, loopNum);
    blockNums.loopNum = loopNum / resStrategies[dieId].loopNum;

    uint32_t msNum = 0;
    (void)ccuResSepcs.GetMsNum(dieId, msNum);
    blockNums.msNum = msNum / resStrategies[dieId].msNum;

    uint32_t ckeNum = 0;
    (void)ccuResSepcs.GetCkeNum(dieId, ckeNum);
    if (ckeNum > RESERVED_DISCRETE_CKE_NUM) {
        blockNums.ckeNum = (ckeNum - RESERVED_DISCRETE_CKE_NUM) / resStrategies[dieId].ckeNum;
    } else {
        blockNums.ckeNum = 0;
    }

    uint32_t xnNum = 0;
    (void)ccuResSepcs.GetXnNum(dieId, xnNum);
    if (xnNum > RESERVED_DISCRETE_XN_NUM) {
        blockNums.xnNum = (xnNum - RESERVED_DISCRETE_XN_NUM) / resStrategies[dieId].xnNum;
    } else {
        blockNums.xnNum = 0;
    }

    if (ccuResSepcs.GetCcuVersion() == CcuVersion::CCU_V1) {
        uint32_t gsaNum = 0;
        (void)ccuResSepcs.GetGsaNum(dieId, gsaNum);
        if (gsaNum > RESERVED_DISCRETE_GSA_NUM) {
            blockNums.gsaNum = (gsaNum - RESERVED_DISCRETE_GSA_NUM) / resStrategies[dieId].gsaNum;
        } else {
            blockNums.gsaNum = 0;
        }
    } else {
        blockNums.gsaNum = 0;
    }

    HCCL_INFO(
        "[CcuResBatchAllocator][%s] batch allocator will alloc blocks resources: loop blocks[%u] "
        "ms blocks[%u] cke blocks[%u] xn blocks[%u] gsa blocks[%u], userDevId[%d] dieId[%u].",
        __func__, blockNums.loopNum, blockNums.msNum, blockNums.ckeNum, blockNums.xnNum, blockNums.gsaNum, userDevId,
        dieId);
    return blockNums;
}

HcclResult CcuResBatchAllocator::GetAllocatableMaxBlockResNum(ResType resType, uint8_t dieId, uint32_t& num) const
{
    switch (resType) {
        case ResType::LOOP:
            num = maxResBlockNums_.loopNum * resStrategies_[dieId].loopNum;
            break;
        case ResType::MS:
            num = maxResBlockNums_.msNum * resStrategies_[dieId].msNum;
            break;
        case ResType::CKE:
            num = maxResBlockNums_.ckeNum * resStrategies_[dieId].ckeNum;
            break;
        case ResType::XN:
            num = maxResBlockNums_.xnNum * resStrategies_[dieId].xnNum;
            break;
        case ResType::GSA:
            num = maxResBlockNums_.gsaNum * resStrategies_[dieId].gsaNum;
            break;
        default:
            HCCL_ERROR(
                "[CcuResBatchAllocator][%s] unsupported block res type[%s], userDevId[%d] dieId[%u].", __func__,
                resType.Describe().c_str(), userDevId_, dieId);
            return HcclResult::HCCL_E_PARA;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResBatchAllocator::PreAllocBlockRes()
{
    CcuComponent& ccuComponent = CcuComponent::GetInstance(userDevId_);
    const auto serveMode = CcuResSpecifications::GetInstance(userDevId_).GetServeMode();
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (!dieEnableFlags_[dieId]) {
            HCCL_WARNING(
                "[CcuResBatchAllocator][%s] userDevId[%d] dieId[%u] is not enable, "
                "will not pre-allocate block resource.",
                __func__, userDevId_, dieId);
            continue;
        }

        maxResBlockNums_ = GetPreAllocatedMaxBlockNums(userDevId_, dieId, resStrategies_);
        const std::array<std::tuple<ResType, uint32_t, uint32_t>, BLOCK_RES_TYPE_NUM> blockResReqs = {
            std::make_tuple(ResType::LOOP, maxResBlockNums_.loopNum, resStrategies_[dieId].loopNum),
            std::make_tuple(ResType::MS, maxResBlockNums_.msNum, resStrategies_[dieId].msNum),
            std::make_tuple(ResType::CKE, maxResBlockNums_.ckeNum, resStrategies_[dieId].ckeNum),
            std::make_tuple(ResType::XN, maxResBlockNums_.xnNum, resStrategies_[dieId].xnNum),
            std::make_tuple(ResType::GSA, maxResBlockNums_.gsaNum, resStrategies_[dieId].gsaNum),
        };

        for (auto& resReq : blockResReqs) {
            const ResType resType = std::get<0>(resReq);
            const uint32_t blockNum = std::get<1>(resReq);
            const uint32_t blockSize = std::get<2>(resReq);
            const uint32_t reqNum = blockNum * blockSize; // 生成时已保证不会溢出
            if (reqNum == 0) {
                HCCL_WARNING(
                    "[CcuResBatchAllocator][%s] userDevId[%d] dieId[%u], "
                    "resType[%s], request num is 0, passed.",
                    __func__, userDevId_, dieId, resType.Describe().c_str());
                continue;
            }

            std::vector<ResInfo> tempResInfos;
            auto ret = ccuComponent.AllocRes(dieId, resType, reqNum, true, tempResInfos);
            if (ret == HcclResult::HCCL_E_UNAVAIL) {
                HCCL_WARNING(
                    "[CcuResBatchAllocator][%s] failed, userDevId[%d] dieId[%u], "
                    "failed to pre allocate block type resource, resType[%s], num[%u].",
                    __func__, userDevId_, dieId, resType.Describe().c_str(), reqNum);
                return ret;
            }
            CHK_RET(ret);

            const bool avoidCcu0Flag = (serveMode == ServeMode::ARMX86 && dieId == 0 && resType == ResType::MS);
            std::vector<BlockInfo> tempBlocks;
            const uint32_t startId = tempResInfos[0].startId;
            for (uint32_t k = 0; k < blockNum; k++) {
                BlockInfo blockInfo;
                blockInfo.id = k;
                blockInfo.startId = startId + k * blockSize;
                blockInfo.num = blockSize;
                // A+X形态，PCIE连接到IOdie0，导致IOdie0上连接PCIE的CCUA0无法使用，分配MS资源时需要跳过CCUA0
                // 给要分给CCUA0的块，设置成已分配过，防止后续分给算法使用
                blockInfo.allocated = avoidCcu0Flag ? k % CCUA_NUM == 0 : false;
                blockInfo.handle = 0;
                tempBlocks.emplace_back(blockInfo);
            }
            resBlocks_[dieId][resType] = std::move(tempBlocks);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

static bool CheckReqValid(const CcuResReq& req, int32_t userDevId, std::array<bool, CCU_MAX_IODIE_NUM>& dieEnableFlags)
{
    bool ifValid = false;
    for (uint8_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        const std::array<uint32_t, REQ_RES_TYPE_NUM> reqs
            = {req.loopEngineReq[i], req.blockLoopEngineReq[i], req.msReq[i],         req.blockMsReq[i],
               req.ckeReq[i],        req.blockCkeReq[i],        req.xnReq[i],         req.blockXnReq[i],
               req.gsaReq[i],        req.blockGsaReq[i],        req.missionReq.req[i]};

        const bool ifReqEmpty = std::all_of(std::begin(reqs), std::end(reqs), [](uint32_t x) {
            return x == 0;
        });

        if (!dieEnableFlags[i] && !ifReqEmpty) { // 当前die未使能，但请求资源
            HCCL_ERROR(
                "[CcuResBatchAllocator][%s] failed, dieId[%u] is not enable, "
                "but resource request is not empty, userDevId[%d].",
                __func__, i, userDevId);
            return false;
        }

        // 当前die使能，并且请求资源即合法
        if (dieEnableFlags[i] && !ifReqEmpty) {
            ifValid = true;
        }
    }

    if (!ifValid) {
        HCCL_ERROR(
            "[CcuResBatchAllocator][%s] all dies resource request is empty, "
            "userDevId[%d].",
            __func__, userDevId);
    }

    return ifValid;
}

HcclResult CcuResBatchAllocator::AllocResHandle(const CcuResReq& resReq, CcuResHandle& resHandle)
{
    if (!CheckReqValid(resReq, userDevId_, dieEnableFlags_)) {
        resHandle = nullptr;
        HCCL_ERROR(
            "[CcuResBatchAllocator][%s] failed, userDevId[%d], invalid resource "
            "request, all resource request is empty.",
            __func__, userDevId_);
        return HcclResult::HCCL_E_PARA;
    }

    std::unique_ptr<CcuResRepository> resRepoPtr = nullptr;
    resRepoPtr.reset(new (std::nothrow) CcuResRepository());
    CHK_PTR_NULL(resRepoPtr);
    const uintptr_t handleKey = reinterpret_cast<uintptr_t>(resRepoPtr.get());
    // 申请分配临时资源
    HcclResult ret = TryAllocResHandle(handleKey, resReq, resRepoPtr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        resHandle = nullptr;
        HCCL_WARNING(
            "[CcuResBatchAllocator][%s] failed, userDevId[%d], failed to "
            "allocate resource handle, release temporary resources of this request.",
            __func__, userDevId_);

        // 释放申请的临时资源，由CcuResRepo对象对应的智能指针管理
        HcclResult releaseRet = ReleaseResource(resRepoPtr);
        if (releaseRet != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CcuResBatchAllocator][%s] failed, userDevId[%d], "
                "failed to release temporary resources of this request.",
                __func__, userDevId_);
            return releaseRet;
        }

        HCCL_INFO(
            "[CcuResBatchAllocator][%s] userDevId[%d], "
            "temporary resources released.",
            __func__, userDevId_);
        return ret;
    }
    // 保存资源信息
    resHandle = reinterpret_cast<CcuResHandle>(resRepoPtr.get());
    handleMap_[handleKey] = std::move(resRepoPtr);

    return HcclResult::HCCL_SUCCESS;
}

static HcclResult HandleBlockRes(
    const uintptr_t handleKey, const uint32_t num, const uint32_t blockSize, std::vector<BlockInfo>& blocks,
    std::vector<ResInfo>& resInfos)
{
    uint32_t blockNum = 1 + (num - 1) / blockSize;
    uint32_t blockMaxSize = blocks.size();
    uint32_t blockStartId = blockMaxSize;
    uint32_t freeNum = 0;
    bool allocatable = false;
    for (size_t k = 0; k < blockMaxSize; k++) {
        // 如果当前块已分配，说明当前分配不够，重置分配数量与起始id
        if (blocks[k].allocated) {
            blockStartId = blockMaxSize;
            freeNum = 0;
            continue;
        }
        // 如果是首个可分配块，记录起始id
        if (blockStartId == blockMaxSize) {
            blockStartId = k;
        }
        // 当前块未分配，更新可分配数量
        freeNum++;
        // 可分配数量足够则分配成功
        if (freeNum >= blockNum) {
            allocatable = true;
            break;
        }
    }
    if (!allocatable) {
        return HcclResult::HCCL_E_UNAVAIL;
    }
    // 更新所有新分配的块的信息
    for (size_t k = blockStartId; k < blockStartId + blockNum; k++) {
        blocks[k].handle = handleKey;
        blocks[k].allocated = true;
    }
    resInfos.emplace_back(ResInfo{blocks[blockStartId].startId, blockNum * blockSize});
    return HcclResult::HCCL_SUCCESS;
}

static void DumpBlockResInfo(ResType resType, const std::vector<BlockInfo>& blocks)
{
    HCCL_INFO("Dump ResType[%s] block resources info: ", resType.Describe().c_str());
    uint32_t blockNum = blocks.size();
    for (size_t k = 0; k < blockNum; k++) {
        HCCL_INFO(
            "Block[id[%u], startId[%u], num[%u], handle(uintptr_t)[%llu], allocated[%d]]", blocks[k].id,
            blocks[k].startId, blocks[k].num, static_cast<unsigned long long>(blocks[k].handle),
            static_cast<int>(blocks[k].allocated));
    }
}

HcclResult CcuResBatchAllocator::AllocBlockRes(
    const uintptr_t handleKey, const CcuResReq& resReq, std::unique_ptr<CcuResRepository>& resRepoPtr)
{
    using ResTypeReqNumBlockNumFunc = std::tuple<ResType::Value, uint32_t, uint32_t, std::vector<ResInfo>&>;

    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (!dieEnableFlags_[dieId]) {
            HCCL_WARNING(
                "[CcuResBatchAllocator][%s] userDevId[%d] dieId[%u] is not enable, "
                "will not allocate block resource.",
                __func__, userDevId_, dieId);
            continue;
        }

        std::array<ResTypeReqNumBlockNumFunc, BLOCK_RES_TYPE_NUM> blockReqParas
            = {std::make_tuple(
                   ResType::LOOP, resReq.blockLoopEngineReq[dieId], resStrategies_[dieId].loopNum,
                   std::ref(resRepoPtr->blockLoopEngine[dieId])),
               std::make_tuple(
                   ResType::MS, resReq.blockMsReq[dieId], resStrategies_[dieId].msNum,
                   std::ref(resRepoPtr->blockMs[dieId])),
               std::make_tuple(
                   ResType::CKE, resReq.blockCkeReq[dieId], resStrategies_[dieId].ckeNum,
                   std::ref(resRepoPtr->blockCke[dieId])),
               std::make_tuple(
                   ResType::XN, resReq.blockXnReq[dieId], resStrategies_[dieId].xnNum,
                   std::ref(resRepoPtr->blockXn[dieId])),
               std::make_tuple(
                   ResType::GSA, resReq.blockGsaReq[dieId], resStrategies_[dieId].gsaNum,
                   std::ref(resRepoPtr->blockGsa[dieId]))};

        for (uint32_t blockType = 0; blockType < BLOCK_RES_TYPE_NUM; blockType++) {
            const auto& req = blockReqParas[blockType];
            const uint32_t num = std::get<1>(req);
            if (num == 0) {
                continue;
            }

            const ResType resType = std::get<0>(req);
            const uint32_t blockSize = std::get<2>(req);
            auto& blocks = resBlocks_[dieId][resType];
            auto& resInfos = std::get<3>(req);
            auto ret = HandleBlockRes(handleKey, num, blockSize, blocks, resInfos);
            if (ret != HcclResult::HCCL_SUCCESS) {
                HCCL_WARNING(
                    "[CcuResBatchAllocator][%s] failed, userDevId[%d] dieId[%u], "
                    "failed to allocate [%s] block resource, remaining block resources are "
                    "not enough, request num[%u].",
                    __func__, userDevId_, dieId, resType.Describe().c_str(), num);
                DumpBlockResInfo(resType, resBlocks_[dieId][resType]);
                return ret;
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuResBatchAllocator::AllocConsecutiveRes(const CcuResReq& resReq, std::unique_ptr<CcuResRepository>& resRepoPtr) const
{
    using ResTypeReqNumResInfoTuple = std::tuple<ResType, uint32_t, std::vector<ResInfo>&>;

    CcuComponent& ccuComponent = CcuComponent::GetInstance(userDevId_);
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (!dieEnableFlags_[dieId]) {
            HCCL_WARNING(
                "[CcuResBatchAllocator][%s] userDevId[%d] dieId[%u] is not enable, "
                "will not allocate consecutive resource.",
                __func__, userDevId_, dieId);
            continue;
        }

        std::array<ResTypeReqNumResInfoTuple, CONS_RES_TYPE_NUM> reqParas
            = {std::make_tuple(ResType::XN, resReq.xnReq[dieId], std::ref(resRepoPtr->xn[dieId]))};

        for (const auto& req : reqParas) {
            if (std::get<1>(req) == 0) {
                continue;
            }

            std::vector<ResInfo> resInfos;
            auto ret = ccuComponent.AllocRes(dieId, std::get<0>(req), std::get<1>(req), true, resInfos);
            if (ret == HcclResult::HCCL_E_UNAVAIL) {
                HCCL_WARNING(
                    "[CcuResBatchAllocator][%s] failed, userDevId[%d] dieId[%u], "
                    "failed to allocate %s resource, num[%u].",
                    __func__, userDevId_, dieId, std::get<0>(req).Describe().c_str(), std::get<1>(req));
                return ret;
            }
            CHK_RET(ret);

            std::get<2>(req) = resInfos;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CcuResBatchAllocator::AllocDiscreteRes(const CcuResReq& resReq, std::unique_ptr<CcuResRepository>& resRepoPtr) const
{
    using ResTypeReqNumResInfoTuple = std::tuple<ResType, uint32_t, std::vector<ResInfo>&>;

    CcuComponent& ccuComponent = CcuComponent::GetInstance(userDevId_);
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (!dieEnableFlags_[dieId]) {
            HCCL_WARNING(
                "[CcuResBatchAllocator][%s] userDevId[%d] dieId[%u] is not enable, "
                "will not allocate discrete resource.",
                __func__, userDevId_, dieId);
            continue;
        }

        std::array<ResTypeReqNumResInfoTuple, DISCRETE_RES_TYPE_NUM> reqParas
            = {std::make_tuple(ResType::LOOP, resReq.loopEngineReq[dieId], std::ref(resRepoPtr->loopEngine[dieId])),
               std::make_tuple(ResType::MS, resReq.msReq[dieId], std::ref(resRepoPtr->ms[dieId])),
               std::make_tuple(ResType::CKE, resReq.ckeReq[dieId], std::ref(resRepoPtr->cke[dieId])),
               std::make_tuple(ResType::GSA, resReq.gsaReq[dieId], std::ref(resRepoPtr->gsa[dieId]))};

        for (const auto& req : reqParas) {
            if (std::get<1>(req) == 0) {
                continue;
            }

            std::vector<ResInfo> resInfos;
            auto ret = ccuComponent.AllocRes(dieId, std::get<0>(req), std::get<1>(req), false, resInfos);
            if (ret == HcclResult::HCCL_E_UNAVAIL) {
                HCCL_WARNING(
                    "[CcuResBatchAllocator][%s] failed, userDevId[%d] dieId[%u], "
                    "failed to allocate %s resource, num[%u].",
                    __func__, userDevId_, dieId, std::get<0>(req).Describe().c_str(), std::get<1>(req));
                return ret;
            }
            CHK_RET(ret);

            std::get<2>(req) = resInfos; // 2: resRepoPtr to resource
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResBatchAllocator::TryAllocResHandle(
    const uintptr_t handleKey, const CcuResReq& resReq, std::unique_ptr<CcuResRepository>& resRepoPtr)
{
    std::unique_lock<std::mutex> lock(innerMutex_);

    HcclResult ret = AllocBlockRes(handleKey, resReq, resRepoPtr);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING(
            "[CcuResBatchAllocator][%s] failed, userDevId[%d], "
            "failed to allocate block type resource.",
            __func__, userDevId_);
        return ret;
    }
    CHK_RET(ret);

    ret = missionMgr_.Alloc(handleKey, resReq.missionReq, resRepoPtr->mission);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING(
            "[CcuResBatchAllocator][%s] userDevId[%d], failed to allocate "
            "mission resource, remaining block resources are not enough.",
            __func__, userDevId_);
        return ret;
    }
    CHK_RET(ret);

    ret = AllocConsecutiveRes(resReq, resRepoPtr);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING(
            "[CcuResBatchAllocator][%s] userDevId[%d], failed to allocate "
            "consecutive resource.",
            __func__, userDevId_);
        return ret;
    }
    CHK_RET(ret);

    ret = AllocDiscreteRes(resReq, resRepoPtr);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING(
            "[CcuResBatchAllocator][%s] userDevId[%d], failed to allocate "
            "discrete resource.",
            __func__, userDevId_);
        return ret;
    }
    CHK_RET(ret);

    return HcclResult::HCCL_SUCCESS;
}

static void ReleaseBlockRes(const uint32_t blockSize, std::vector<BlockInfo>& blocks, std::vector<ResInfo>& resInfos)
{
    uint32_t startId = resInfos[0].startId;
    uint32_t num = resInfos[0].num;
    uint32_t startBlockId = (startId - blocks[0].startId) / blockSize;
    uint32_t blockNum = num / blockSize;

    for (uint32_t k = startBlockId; k < startBlockId + blockNum; k++) {
        blocks[k].handle = 0;
        blocks[k].allocated = false;
    }
    resInfos.clear();
}

HcclResult CcuResBatchAllocator::ReleaseResHandle(const CcuResHandle& handle)
{
    std::unique_lock<std::mutex> lock(innerMutex_);

    uintptr_t handleKey = reinterpret_cast<uintptr_t>(handle);
    if (handleMap_.find(handleKey) == handleMap_.end()) {
        HCCL_ERROR(
            "[CcuResBatchAllocator][%s] failed, userDevId[%d], "
            "failed to find resource repository, invalid resource handle(uintptr_t)[%llu]",
            __func__, userDevId_, static_cast<unsigned long long>(handleKey));
        return HcclResult::HCCL_E_PARA;
    }

    std::unique_ptr<CcuResRepository>& resRepoPtr = handleMap_[handleKey];

    auto ret = ReleaseResource(resRepoPtr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CcuResBatchAllocator][%s] failed, userDevId[%d], "
            "failed[%u] to release resource.",
            __func__, userDevId_, ret);
        return ret;
    }

    handleMap_.erase(handleKey);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResBatchAllocator::ReleaseResource(std::unique_ptr<CcuResRepository>& resRepoPtr)
{
    ReleaseBlockResource(resRepoPtr);
    missionMgr_.Release(resRepoPtr->mission);
    HcclResult ret = ReleaseNonBlockTypeRes(resRepoPtr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CcuResBatchAllocator][%s] failed, userDevId[%d], "
            "failed[%u] to release discrete resource.",
            __func__, userDevId_, ret);
        return ret;
    }

    return HcclResult::HCCL_SUCCESS;
}

void CcuResBatchAllocator::ReleaseBlockResource(std::unique_ptr<CcuResRepository>& resRepoPtr)
{
    using BlockSizeResNum = std::tuple<ResType, uint32_t, std::vector<ResInfo>&>;

    for (uint8_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        if (!dieEnableFlags_[i]) {
            continue;
        }

        const std::array<BlockSizeResNum, BLOCK_RES_TYPE_NUM> blockReqParas
            = {std::make_tuple(ResType::LOOP, resStrategies_[i].loopNum, std::ref(resRepoPtr->blockLoopEngine[i])),
               std::make_tuple(ResType::MS, resStrategies_[i].msNum, std::ref(resRepoPtr->blockMs[i])),
               std::make_tuple(ResType::CKE, resStrategies_[i].ckeNum, std::ref(resRepoPtr->blockCke[i])),
               std::make_tuple(ResType::XN, resStrategies_[i].xnNum, std::ref(resRepoPtr->blockXn[i])),
               std::make_tuple(ResType::GSA, resStrategies_[i].gsaNum, std::ref(resRepoPtr->blockGsa[i]))};

        for (uint32_t j = 0; j < BLOCK_RES_TYPE_NUM; j++) {
            auto req = blockReqParas[j];
            constexpr size_t kResInfoTupleIdx = 2;
            std::vector<ResInfo>& resInfos = std::get<kResInfoTupleIdx>(req);
            auto resType = std::get<0>(req);
            std::vector<BlockInfo>& blocks = resBlocks_[i][resType];
            if (resInfos.size() == 0 || blocks.size() == 0) {
                continue;
            }
            ReleaseBlockRes(std::get<1>(req), blocks, resInfos);
        }
    }
}

using ResTypeResInfo = std::pair<ResType, std::vector<ResInfo>*>;
static auto EraseReverse(std::vector<ResInfo>& vec, std::vector<ResInfo>::reverse_iterator it)
    -> std::vector<ResInfo>::reverse_iterator
{
    return std::vector<ResInfo>::reverse_iterator(vec.erase(std::next(it).base()));
}

static HcclResult
DoReleaseNonBlockTypeRes(int32_t userDevId, uint8_t dieId, std::array<ResTypeResInfo, NON_BLOCK_TYPE_NUM>& infoParas)
{
    CcuComponent& ccuComponent = CcuComponent::GetInstance(userDevId);

    for (auto& infos : infoParas) {
        const ResType resType = infos.first;
        std::vector<ResInfo>* resInfosPtr = infos.second;
        if (resInfosPtr == nullptr || resInfosPtr->empty()) {
            continue;
        }
        std::vector<ResInfo>& resInfos = *resInfosPtr;
        // 倒序删除，减少vector元素移动
        for (auto it = resInfos.rbegin(); it != resInfos.rend();) {
            const uint32_t num = it->num;
            if (num == 0) {
                it = EraseReverse(resInfos, it);
                continue;
            }

            const uint32_t startId = it->startId;
            auto ret = ccuComponent.ReleaseRes(dieId, resType, startId, num);
            if (ret != HcclResult::HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[CcuResBatchAllocator][%s] failed, userDevId[%d] dieId[%u], "
                    "failed to release %s resource, startId[%u], num[%u].",
                    __func__, userDevId, dieId, resType.Describe().c_str(), startId, num);
                return ret;
            }

            it = EraseReverse(resInfos, it);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResBatchAllocator::ReleaseNonBlockTypeRes(std::unique_ptr<CcuResRepository>& resRepoPtr) const
{
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (!dieEnableFlags_[dieId]) {
            continue;
        }

        std::array<ResTypeResInfo, NON_BLOCK_TYPE_NUM> infoParas
            = {{{ResType::LOOP, &resRepoPtr->loopEngine[dieId]},
                {ResType::MS, &resRepoPtr->ms[dieId]},
                {ResType::CKE, &resRepoPtr->cke[dieId]},
                {ResType::XN, &resRepoPtr->xn[dieId]},
                {ResType::GSA, &resRepoPtr->gsa[dieId]}}};

        CHK_RET(DoReleaseNonBlockTypeRes(userDevId_, dieId, infoParas));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResBatchAllocator::GetResource(const CcuResHandle& handle, CcuResRepository& ccuResRepo)
{
    std::unique_lock<std::mutex> lock(innerMutex_);

    uintptr_t handleKey = reinterpret_cast<uintptr_t>(handle);
    if (handleMap_.find(handleKey) == handleMap_.end()) {
        HCCL_ERROR(
            "[CcuResBatchAllocator][%s] userDevId[%d], failed to find "
            "resource repository, invalid resource handle(uintptr_t)[%lu]",
            __func__, userDevId_, handleKey);
        return HcclResult::HCCL_E_PARA;
    }

    ccuResRepo = *(handleMap_[handleKey].get());
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult PreAllocMissionRes(
    int32_t userDevId, std::array<bool, CCU_MAX_IODIE_NUM>& dieEnableFlags,
    std::array<uint32_t, CCU_MAX_IODIE_NUM>& missionNums, std::array<uint32_t, CCU_MAX_IODIE_NUM>& missionStartIds)
{
    auto& ccuResSepcs = CcuResSpecifications::GetInstance(userDevId);
    auto& ccuComponent = CcuComponent::GetInstance(userDevId);
    for (uint8_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        if (!dieEnableFlags[i]) {
            missionNums[i] = 0;
            missionStartIds[i] = 0;
            continue;
        }

        (void)ccuResSepcs.GetMissionNum(i, missionNums[i]);
        std::vector<ResInfo> tempResInfos;
        auto ret = ccuComponent.AllocRes(i, ResType::MISSION, missionNums[i], true, tempResInfos);
        if (ret == HcclResult::HCCL_E_UNAVAIL) {
            HCCL_WARNING(
                "[CcuMissionMgr][%s] userDevId[%d] dieId[%u], failed[%u] "
                "to pre allocate mission resource, num[%u]",
                __func__, userDevId, i, ret, missionNums[i]);
            return ret;
        }
        CHK_RET(ret);

        missionStartIds[i] = tempResInfos[0].startId;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResBatchAllocator::CcuMissionMgr::PreAlloc(
    const int32_t userDevId, const uint32_t blockSize, const std::array<bool, CCU_MAX_IODIE_NUM>& dieFlags)
{
    dieEnableFlags_ = dieFlags;
    std::array<uint32_t, CCU_MAX_IODIE_NUM> missionNums;
    std::array<uint32_t, CCU_MAX_IODIE_NUM> missionStartIds;

    auto ret = PreAllocMissionRes(userDevId, dieEnableFlags_, missionNums, missionStartIds);
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }

    uint32_t missionNum = 0;
    if (dieEnableFlags_[0]) {
        missionNum = missionNums[0];
    } else if (dieEnableFlags_[1]) {
        missionNum = missionNums[1];
    }

    if (dieEnableFlags_[0] && dieEnableFlags_[1] && missionStartIds[0] != missionStartIds[1]) {
        // 当前 FUSION_MULTIPLE_DIE 要求多Die ID一致
        HCCL_ERROR(
            "[CcuMissionMgr][%s] userDevId[%d] die 0 allocated missions "
            "start with id %u, die 1 allocated missions start with id %u, the start "
            "id should be same.",
            __func__, userDevId, missionStartIds[0], missionStartIds[1]);
        return HcclResult::HCCL_E_INTERNAL;
    }

    strategy_ = blockSize;
    uint32_t blockNum = missionNum / strategy_;
    for (uint32_t i = 0; i < blockNum; i++) {
        BlockInfo blockInfo;
        blockInfo.id = i;
        blockInfo.startId = missionStartIds[0] + i * strategy_;
        blockInfo.num = strategy_;
        blockInfo.allocated = false;
        blockInfo.handle = 0;
        blocks_.emplace_back(blockInfo);
    }

    return HcclResult::HCCL_SUCCESS;
}

static uint32_t
Check2DieMissionReqNum(const MissionReq& missionReq, const std::array<bool, CCU_MAX_IODIE_NUM>& dieEnableFlags)
{
    uint32_t die0ReqNum = missionReq.req[0];
    uint32_t die1ReqNum = missionReq.req[1];

    if (dieEnableFlags[0] && dieEnableFlags[1]) {
        if (die0ReqNum != die1ReqNum) {
            HCCL_WARNING(
                "[CcuMissionMgr][Alloc] die 0 request %u, die 1 request %u, "
                "will choose the larger one.",
                die0ReqNum, die1ReqNum);
            return std::max(die0ReqNum, die1ReqNum);
        }

        return die0ReqNum;
    }

    if (dieEnableFlags[0]) {
        return die0ReqNum;
    }

    if (dieEnableFlags[1]) {
        return die1ReqNum;
    }

    return 0;
}

HcclResult CcuResBatchAllocator::CcuMissionMgr::Alloc(
    const uintptr_t handleKey, const MissionReq& missionReq, MissionResInfo& missionInfos)
{
    MissionReqType reqType = missionReq.reqType;
    constexpr MissionReqType defaultReqType = MissionReqType::FUSION_MULTIPLE_DIE;
    if (missionReq.reqType != MissionReqType::FUSION_MULTIPLE_DIE) {
        HCCL_WARNING(
            "[CcuMissionMgr][%s] mission reqType[%d], mission resources "
            "now only support %d.",
            __func__, reqType, defaultReqType);
        reqType = MissionReqType::FUSION_MULTIPLE_DIE;
    }

    uint32_t reqNum = Check2DieMissionReqNum(missionReq, dieEnableFlags_);
    if (reqNum == 0) {
        HCCL_INFO(
            "[CcuMissionMgr][%s] passed, request mission num is 0, "
            "will not allocate mission resource.",
            __func__);
        return HcclResult::HCCL_SUCCESS;
    }

    std::vector<ResInfo> resInfos;
    auto ret = HandleBlockRes(handleKey, reqNum, strategy_, blocks_, resInfos);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING(
            "[CcuMissionMgr][%s] failed, mission block resources are unavailable, "
            "reqNum[%u], strategy[%u], reqType[%d].",
            __func__, reqNum, strategy_, reqType);
        DumpBlockResInfo(ResType::MISSION, blocks_);
        return ret;
    }
    CHK_RET(ret);

    missionInfos.reqType = reqType;

    for (uint8_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        if (dieEnableFlags_[i]) {
            missionInfos.mission[i] = resInfos;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

void CcuResBatchAllocator::CcuMissionMgr::Release(MissionResInfo& missionInfos)
{
    // 目前支持 FUSION_MULTIPLE_DIE 类型，故多die同步释放
    for (uint8_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        if (dieEnableFlags_[i] && missionInfos.mission[i].size() != 0) {
            ReleaseBlockRes(strategy_, blocks_, missionInfos.mission[i]);
            break;
        }
    }

    for (uint8_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        missionInfos.mission[i].clear();
    }
}

void CcuResBatchAllocator::CcuMissionMgr::Reset() { blocks_.clear(); }

// 根据 resType 解析对应的 blocks 指针，将 MISSION 与普通块类型的分支收敛至此
// GetAllocatableMaxBlockResNum 内部通过 switch 校验块类型，非法类型直接返回错误
HcclResult
CcuResBatchAllocator::ResolveBlocksPtr(uint8_t dieId, ResType resType, const std::vector<BlockInfo>*& blocksPtr) const
{
    if (resType == ResType::MISSION) {
        blocksPtr = &missionMgr_.GetBlocks();
        return HCCL_SUCCESS;
    }

    uint32_t poolSize = 0;
    CHK_RET(GetAllocatableMaxBlockResNum(resType, dieId, poolSize));
    if (poolSize == 0) {
        blocksPtr = nullptr;
        return HCCL_SUCCESS;
    }

    auto it = resBlocks_[dieId].find(resType);
    if (it == resBlocks_[dieId].end()) {
        blocksPtr = nullptr;
        return HCCL_SUCCESS;
    }
    blocksPtr = &it->second;
    return HCCL_SUCCESS;
}

// 直接扫描 resBlocks 的 allocated 标志计算最大连续空闲块数 × blockSize
// Block 分配时 HandleBlockRes 同步更新 allocated, 无需绕道 handleMap
HcclResult CcuResBatchAllocator::QueryRemainRes(uint8_t dieId, ResType resType, uint32_t& remainNum) const
{
    const std::vector<BlockInfo>* blocksPtr = nullptr;
    uint32_t blockSize = 0;

    CHK_RET(ResolveBlocksPtr(dieId, resType, blocksPtr));
    if (blocksPtr == nullptr || blocksPtr->empty()) {
        remainNum = 0;
        return HCCL_SUCCESS;
    }

    blockSize = blocksPtr->front().num;
    uint32_t maxFreeBlocks = 0;
    uint32_t curFreeBlocks = 0;
    for (const auto& block : *blocksPtr) {
        if (!block.allocated) {
            curFreeBlocks++;
            continue;
        }
        maxFreeBlocks = curFreeBlocks > maxFreeBlocks ? curFreeBlocks : maxFreeBlocks;
        curFreeBlocks = 0;
    }
    if (curFreeBlocks > maxFreeBlocks) {
        maxFreeBlocks = curFreeBlocks;
    }

    remainNum = maxFreeBlocks * blockSize;
    HCCL_INFO(
        "[CcuResBatchAllocator][%s] resType[%s] maxFreeBlocks[%u] blockSize[%u] remainNum[%u]", __func__,
        resType.Describe().c_str(), maxFreeBlocks, blockSize, remainNum);
    return HCCL_SUCCESS;
}

}; // namespace hcomm

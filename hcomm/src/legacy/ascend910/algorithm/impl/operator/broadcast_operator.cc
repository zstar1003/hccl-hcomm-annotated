/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_operator.h"
#include "device_capacity.h"
#include "coll_alg_op_registry.h"

namespace hccl {
constexpr u64 AIV_MAX_DATASIZE = 4ULL * 1024 * 1024 * 1024;

BroadCastOperator::BroadCastOperator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_BROADCAST)
{
    // 由于bcast暂不支持server间ring，需继续使用HD或NHR
    if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR)
        && !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1)
        && !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING)
        && !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
        HCCL_WARNING("[BroadCastOperator][BroadCastOperator] do not support ring in AlgoLevel1 yet, reset algType=HD.");
    }
}
BroadCastOperator::~BroadCastOperator() {}

HcclResult
BroadCastOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag)
{
    HcclResult ret;
    isAivMode_ = false;
    if (isDiffDeviceType_) {
        ret = SelectAlgforMix(param, algName);
    } else if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        ret = SelectAlgfor310P3(param, algName);
    } else if (Is310PDevice() && topoType_ == TopoType::TOPO_TYPE_2P_MESH) {
        ret = SelectAlgfor310P(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else {
        HCCL_ERROR("BroadCastOperator[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastSelector][SelectAlg]tag[%s], broadcast failed, return[%d]", tag.c_str(), ret), ret);

    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = tag;
    } else if (
        isAivMode_
        || (Is310P3Common(isHaveCpuRank_, deviceType_) && (algType_.algoLevel1 != AlgTypeLevel1::ALG_LEVEL1_HD))) {
        newTag = tag + algName;
    } else if (
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD || algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
        newTag = tag + algName;
        std::string appendTag = "";
        u32 serverNumPerSuperPod = superPodNum_ == 0 ? moduleNum_ : moduleNum_ / superPodNum_;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
            u32 part1Size = FACTOR_TWO * (serverNumPerSuperPod - (1 << static_cast<u32>(log2(serverNumPerSuperPod))));
            u32 rootId = param.root / deviceNumPerAggregation_ % serverNumPerSuperPod;
            appendTag += "L1_" + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
            u32 part1Size = FACTOR_TWO * (superPodNum_ - (1 << static_cast<u32>(log2(superPodNum_))));
            u32 rootId = param.root / deviceNumPerAggregation_ / serverNumPerSuperPod;
            appendTag += (appendTag.empty() ? "L2_" : "_L2_")
                         + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        HCCL_DEBUG("[BroadCastOperator][SelectAlg]tag is [%s]", tag.c_str());
        newTag = newTag + '_' + appendTag;
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
        }
    } else {
        AlgTypeLevel1 algType1 = algType_.algoLevel1;
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        CHK_PRT_RET(
            level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.", algType1),
            HCCL_E_INTERNAL);
        newTag = tag + level1Iter->second + algName;
    }
    HCCL_DEBUG("[%s] SelectAlg for newTag", __func__);
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}

HcclResult BroadCastOperator::SelectAlgforMix(const OpParam& param, std::string& algName)
{
    (void)param;
    if (gcdDeviceNumPerAggregation_ > 1) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING("[BroadCastOperator][SelectAlgforMix] only support NHR in AlgoLevel1 yet, "
                     "default is algType=NHR.");
        algName = "BroadCastMixExecutor";
    } else {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        HCCL_WARNING("[BroadCastOperator][SelectAlgforMix] only support ring in AlgoComm yet, "
                     "default is algType=ring.");
        algName = "BroadCastComm";
    }

    HCCL_INFO("[SelectAlgforMix] broadcast SelectAlgforMix is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName) const
{
    (void)param;
    algName = "BroadCastCommFor310P";
    HCCL_INFO("[SelectAlgfor310P3] broadcast SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor310P([[maybe_unused]] const OpParam& param, std::string& algName) const
{
    algName = "BroadcastPlusBroadcast";
    HCCL_INFO("[SelectAlgfor310P] broadcast SelectAlgfor310P is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor910A(const OpParam& param, std::string& algName) const
{
    (void)param;
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "BroadCastMeshExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_4P_RING) {
        algName = "BroadCast4pRingExecutor";
    } else if (isRingTopo) {
        algName = "BroadCastRingExecutor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor910A] broadcast SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH
                      || topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    // 暂只支持单算子模式
    const u64 commInputSize = cclBufferManager_.GetInCCLbufferSize();
    const u64 commOutputSize = cclBufferManager_.GetOutCCLbufferSize();
    u64 dataSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    bool isCCLBufferGE16M = commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB;
    isAivMode_ = topoMatcher_->GetAivModeConfig() && isSingleMeshAggregation_ && isOpbase && isCCLBufferGE16M
                 && IsSupportAIVCopy(param.DataDes.dataType) && dataSize <= AIV_MAX_DATASIZE;
    if (isAivMode_) {
        algName = "BroadcastMeshAivExecutor";
    } else if (isMeshTopo) {
        algName = "BroadCastMeshExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_4P_RING) {
        algName = "BroadCast4pRingExecutor";
    } else if (isRingTopo) {
        algName = "BroadCastRingExecutor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor910B] broadcast SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
{
    // level 1重定向为NHR, 因scatter && broadcast只支持nhr/nb
    if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR)
        && !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING(
            "[BroadCastOperator][BroadCastOperator] do not support ring in AlgoLevel1 yet, reset algType=NHR.");
    }

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    if (dataSize >= cclBufferManager_.GetInCCLbufferSize()) {
        HCCL_WARNING(
            "The current inCCLbufferSize is [%llu] bytes, change the HCCL_BUFFSIZE environment variable "
            "to be greater than the current data volume[%llu] bytes to improve the performance of the 91093 "
            "environment.",
            cclBufferManager_.GetInCCLbufferSize(), dataSize);
    }

    const u64 commInputSize = cclBufferManager_.GetInCCLbufferSize();
    const u64 commOutputSize = cclBufferManager_.GetOutCCLbufferSize();
    bool isCCLBufferGE16M = commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB;
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    bool hasEnoughCCLBuffer = isSingleMeshAggregation_ && isOpbase && isCCLBufferGE16M && dataSize <= AIV_MAX_DATASIZE;
    bool hasSmallDataOrOnlyAiv = !isOpbase && (isOnlyAiv || dataSize <= HCCL_MID_COUNT_16_MB);
    bool isSatisfyDataSize = (userRankSize_ <= DEVICE_EIGHT && dataSize <= HCCL_SMALL_COUNT_512_KB)
                             || (userRankSize_ > DEVICE_EIGHT && dataSize <= HCCL_SMALL_COUNT_1_MB);

    bool isAivSingleNode = serverNum_ == 1 && (hasEnoughCCLBuffer || hasSmallDataOrOnlyAiv);
    bool isAivCrossNode = (superPodNum_ == 1) && (serverNum_ > 1) && !GetExternalInputInterHccsDisable()
                          && ((isOpbase && isCCLBufferGE16M && isSatisfyDataSize) || hasSmallDataOrOnlyAiv);

    isAivMode_ = topoMatcher_->GetAivModeConfig() && IsSupportAIVCopy(param.DataDes.dataType)
                 && (isAivSingleNode || isAivCrossNode);

    bool smallCountOptimSingleServer
        = (serverNum_ == 1)
          && ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
              || (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !param.aicpuUnfoldMode))
          && (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO)
          && (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_512_KB * userRankSize_);
    bool smallCountOptimMultiServer = (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) && (serverNum_ != 1)
                                      && (superPodNum_ == 1)
                                      && (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType]
                                          <= HCCL_SMALL_COUNT_1_MB * deviceNumPerAggregation_);
    bool is2Pod2ServerTopo = (superPodNum_ == 2 && serverNum_ == 2); // 针对 A3背靠背机型
    bool smallCountOptimMultiPod
        = (superPodNum_ > 1 || (GetExternalInputInterHccsDisable() && serverNum_ > 1)) && !is2Pod2ServerTopo
          && (param.DataDes.count * unitSize <= HCCL_SMALL_COUNT_16_KB * deviceNumPerAggregation_)
          && !retryEnable_; // 涉及ROCE平面
    bool isBack2BackFor91093 = is2Pod2ServerTopo && (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
                               && (param.DataDes.count * unitSize <= HCCL_SMALL_COUNT_16_KB * deviceNumPerAggregation_)
                               && !retryEnable_;
    if (isAivMode_) {
        if (isAivSingleNode) {
            algName = "BroadcastMeshAivExecutor";
        } else {
            algName = "BroadcastMeshAivFor91093Executor";
        }
    } else if (multiModuleDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_) {
        algName = "BroadCastComm";
    } else if (smallCountOptimMultiServer || smallCountOptimMultiPod) {
        algName = "BroadCastComm";
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    } else if (isBack2BackFor91093) {
        algName = "BroadcastMidCountFor91093Executor";
    } else if (smallCountOptimSingleServer) {
        algName = "BroadCastSmallCountExecutor";
    } else if (
        param.supportZeroCopy
        && (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING
            || param.DataDes.count * unitSize > HCCL_MID_COUNT_16_MB * serverNum_)) {
        algName = "BroadCastRingZerocopyExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        algName = "BroadCastRingFor91093Executor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor91093] broadcast SelectAlgfor91093 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_BROADCAST, Broadcast, BroadCastOperator);
} // namespace hccl

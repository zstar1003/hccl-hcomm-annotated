/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_v_operator.h"
#include "device_capacity.h"
#include "coll_alg_op_registry.h"
#include "hccl_aiv.h"

namespace hccl {

constexpr u64 MAX_310P_RANK_SIZE = 4;
constexpr u32 MODULE_NUM_FOUR = 4;

AllGatherVOperator::AllGatherVOperator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER_V)
{}

AllGatherVOperator::~AllGatherVOperator() {}

HcclResult
AllGatherVOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag)
{
    HcclResult ret;
    HCCL_DEBUG("[%s] SelectAlg begins", __func__);
    if (isDiffDeviceType_) {
        HCCL_ERROR("[AllGatherVOperator][SelectAlg] AllGatherV not support diffDeviceType");
        return HCCL_E_NOT_SUPPORT;
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        ret = SelectAlgfor310P3(param, algName);
    } else {
        HCCL_ERROR("[AllGatherVOperator][SelectAlg] AllGatherV only support A3, A2 and 310P.");
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherVOperator][SelectAlg]tag[%s], AllGatherV failed, return[%d]", tag.c_str(), ret), ret);

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        newTag = tag + algName;
    } else {
        AlgTypeLevel1 algType1 = algType_.algoLevel1;
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        CHK_PRT_RET(
            level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.", algType1),
            HCCL_E_INTERNAL);

        newTag = tag + level1Iter->second + algName;
    }

    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}

HcclResult AllGatherVOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
{
    const HcclDataType dataType = param.VDataDes.dataType;
    const auto* countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    const auto countsPerRank = std::vector<u64>(countsPtr, countsPtr + userRankSize_);
    const u64 maxCount = *std::max_element(countsPerRank.begin(), countsPerRank.end());
    const u32 unitSize = SIZE_TABLE[dataType];
    const u64 dataSize = maxCount * unitSize; // 单位：字节
    if (dataSize >= cclBufferManager_.GetInCCLbufferSize()) {
        HCCL_WARNING(
            "The current inCCLbufferSize is [%llu] bytes, change the HCCL_BUFFSIZE environment variable to "
            "be greater than the current data volume[%llu] bytes to improve the performance of the 91093 environment.",
            cclBufferManager_.GetInCCLbufferSize(), dataSize);
    }

    if (multiModuleDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_) {
        HCCL_ERROR(
            "[AllGatherVOperator][SelectAlgfor91093]not support mode, multiModuleDiffDeviceNumMode_[%u], "
            "multiSuperPodDiffServerNumMode_[%u]",
            multiModuleDiffDeviceNumMode_, multiSuperPodDiffServerNumMode_);
        return HCCL_E_NOT_SUPPORT;
    } else {
        if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[AllGatherVOperator][SelectAlgfor91093] only support ring, NB and NHR in AlgoLevel1 yet, "
                         "default is algType=NHR.");
        }
        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            algName = "AlignedAllGatherVDoubleRingFor91093Executor";
        } else {
            algName = "AllGatherVRingFor91093Executor";
        }
    }

    HCCL_INFO("[SelectAlgfor91093] AllGatherV SelectAlgfor91093 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherVOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    const auto* countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsPerRank = std::vector<u64>(countsPtr, countsPtr + userRankSize_);
    u64 maxCount = *std::max_element(countsPerRank.begin(), countsPerRank.end());
    u32 unitSize = SIZE_TABLE[param.VDataDes.dataType];
    u64 dataSize = maxCount * unitSize;
    bool isBigData = false;

    if (dataSize > AIV_ALL_GATHER_SMALL_SIZE) {
        isBigData = true;
    }

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isSingleMeshAggregation_) {
        u64 cclBufferSize = cclBufferManager_.GetOutCCLbufferSize() / userRankSize_;
        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_ALLGATHER_V, dataSize, cclBufferSize, algTypeLevel1Tag));
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到NHR算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_ALLGATHER_V);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING(
                "[AllGatherVOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+ graph[%u], "
                "reset algorithm to NHR.",
                contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    bool isAivMode = topoMatcher_->GetAivModeConfig() && isSingleMeshAggregation_
                     && IsSupportAIVCopy(param.VDataDes.dataType) && dataSize <= AIV_BIG_SIZE;

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (isAivMode) {
            algName = isBigData ? "AllGatherVMeshAivExecutor" : "AllGatherVMeshAivSmallCountExecutor";
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE && !isSingleMeshAggregation_) {
            algName = "AllGatherVMeshOpbasePipelineExecutor";
        } else {
            algName = "AllGatherVMeshExecutor";
        }
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        if (isSingleMeshAggregation_) {
            algName = "AllGatherVMeshGraphExecutor";
        } else if (
            deviceNumPerAggregation_ > 1
            && (dataSize > HCCL_SMALL_COUNT_1_MB || moduleNum_ <= MODULE_NUM_FOUR
                || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE)) {
            algName = "AllGatherVMeshGraphPipelineExecutor";
        } else {
            algName = "AllGatherVMeshExecutor";
        }
    }
    HCCL_INFO("[SelectAlgfor910B] AllGatherV SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherVOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    (void)param;
    CHK_PRT_RET(
        userRankSize_ > MAX_310P_RANK_SIZE,
        HCCL_ERROR(
            "[AllGatherVOperator][SelectAlgfor310P3]rankSize[%u] is not supported.AllGatherV does not support the "
            "scenario where the rankSize is greater than 4.",
            userRankSize_),
        HCCL_E_NOT_SUPPORT);
    algName = "AllGatherVFor310PExecutor";
    HCCL_INFO("[SelectAlgfor310P3] AllGatherV SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_ALLGATHER_V, AllGatherV, AllGatherVOperator);

} // namespace hccl

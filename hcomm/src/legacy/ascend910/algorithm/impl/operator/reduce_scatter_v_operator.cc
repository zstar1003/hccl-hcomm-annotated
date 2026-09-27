/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_v_operator.h"
#include "device_capacity.h"
#include "hccl_aiv.h"

namespace hccl {

constexpr u64 MAX_310P_RANK_SIZE = 4;

ReduceScatterVOperator::ReduceScatterVOperator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlgOperator(
          algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V)
{}

ReduceScatterVOperator::~ReduceScatterVOperator() {}

HcclResult ReduceScatterVOperator::SelectAlg(
    const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag)
{
    HcclResult ret;

    if (isDiffDeviceType_) {
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlg] ReduceScatterV not support diffDeviceType");
        return HCCL_E_NOT_SUPPORT;
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        ret = SelectAlgfor310P3(param, algName);
    } else {
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlg] ReduceScatterV only support A3, A2 and 310P.");
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlg]tag[%s], ReduceScatterV failed, return[%d]", tag.c_str(), ret),
        ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else {
        if (deviceType_ == DevType::DEV_TYPE_310P3) {
            newTag = tag + algName;
        } else {
            AlgTypeLevel1 algType1 = algType_.algoLevel1;
            auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
            CHK_PRT_RET(
                level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.", algType1),
                HCCL_E_INTERNAL);
            newTag = tag + level1Iter->second + algName;
        }

        bool isInlineReduce = IsSupportSDMAReduce(
            cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(),
            param.VDataDes.dataType, param.reduceType);
        const std::string REDUCE_SCATTER_V_NO_INLINE = "_no_inline";
        newTag = isInlineReduce ? newTag : newTag + REDUCE_SCATTER_V_NO_INLINE;
    }

    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}

HcclResult ReduceScatterVOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
{
    const auto* countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsPerRank = std::vector<u64>(countsPtr, countsPtr + userRankSize_);
    u64 maxCount = *std::max_element(countsPerRank.begin(), countsPerRank.end());
    u32 unitSize = SIZE_TABLE[param.VDataDes.dataType];
    u64 dataSize = maxCount * unitSize; // 单位：字节
    if (dataSize >= cclBufferManager_.GetInCCLbufferSize()) {
        HCCL_WARNING(
            "The current inCCLbufferSize is [%llu] bytes, change the HCCL_BUFFSIZE environment variable to "
            "be greater than the current data volume[%llu] bytes to improve the performance of the 91093 environment.",
            cclBufferManager_.GetInCCLbufferSize(), dataSize);
    }

    if (multiModuleDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_) {
        HCCL_ERROR(
            "[ReduceScatterVOperator][SelectAlgfor91093] not support mode, multiModuleDiffDeviceNumMode_[%u], "
            "multiSuperPodDiffServerNumMode_[%u]",
            multiModuleDiffDeviceNumMode_, multiSuperPodDiffServerNumMode_);
        return HCCL_E_NOT_SUPPORT;
    } else {
        if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
            algName = "ReduceScatterVRingFor91093Executor";
        } else if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            const s32 HCCS_PORT_NUM_910_93_7 = 7;
            if (hccsPortNum_ == HCCS_PORT_NUM_910_93_7) {
                algName = "ReduceScatterVFastDoubleRingFor91093Executor";
            } else {
                algName = "AlignedReduceScatterVDoubleRingFor91093Executor";
            }
        } else {
            HCCL_ERROR("[ReduceScatterVOperator][SelectAlgfor91093] not support topoType_[%u]", topoType_);
            return HCCL_E_NOT_SUPPORT;
        }
    }

    const bool isWholeRing = (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING)
                             && (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING);
    if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || isWholeRing
          || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)) {
        // 910_93超节点只支持server间ring,NB和NHR，默认需继续使用NHR
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING("[ReduceScatterVOperator][SelectAlgfor91093] only support ring, NB and NHR in AlgoLevel1 yet,"
                     " default algType is NHR.");
    }

    HCCL_INFO("[SelectAlgfor91093] ReduceScatterV SelectAlgfor91093 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    // 图模式切入确定性
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && !isSingleMeshAggregation_) {
        if (!multiModuleDiffDeviceNumMode_) {
            algName = "ReduceScatterVDeterExecutor"; // 多机图模式当前默认选中确定性算法
            if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING
                  || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB
                  || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR)) {
                // 只支持server间ring,NB和NHR，默认使能NHR
                algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
                HCCL_WARNING(
                    "[ReduceScatterVOperator][SelectAlgfor910B] only support ring, NB and NHR in AlgoLevel1 yet,"
                    " default algType is NHR.");
            }
            HCCL_INFO("[SelectAlgfor910B] ReduceScatterV SelectAlgfor910B algName is [%s]", algName.c_str());
            return HCCL_SUCCESS;
        } else {
            HCCL_ERROR(
                "[ReduceScatterVOperator][SelectAlgfor910B] ReduceScatterV not support uneven devices in multiServer.");
            return HCCL_E_NOT_SUPPORT;
        }
    }
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH
                      || topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    // Deterministic 确定性分支
    bool isDeterministic
        = topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE && isMeshTopo
          && !multiModuleDiffDeviceNumMode_
          && (param.VDataDes.dataType == HCCL_DATA_TYPE_FP16 || param.VDataDes.dataType == HCCL_DATA_TYPE_FP32
              || param.VDataDes.dataType == HCCL_DATA_TYPE_BFP16);
    if (isDeterministic) {
        // 只有浮点数存在不确定性
        algName = "ReduceScatterVDeterExecutor";
        HCCL_INFO("[SelectAlgfor910B] ReduceScatterV SelectAlgfor910B algName is [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }

    // pipeline算法回退: task数量多，如果超出FFTS子图限制，则重定向到NHR算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING(
                "[ReduceScatterVOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+ "
                "graph[%u], reset algorithm to NHR.",
                contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    // Pipeline
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE
        && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE
        && IsMultiMeshInlineReduce(
            cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(),
            param.VDataDes.dataType, param.reduceType)) {
        algName = "ReduceScatterVMeshOpbasePipelineExecutor";
        HCCL_INFO("[SelectAlgfor910B] ReduceScatterV SelectAlgfor910B algName is [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }

    const auto* countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsPerRank = std::vector<u64>(countsPtr, countsPtr + userRankSize_);
    u64 maxCount = *std::max_element(countsPerRank.begin(), countsPerRank.end());
    u32 unitSize = SIZE_TABLE[param.VDataDes.dataType];
    u64 maxDataSize = maxCount * unitSize; // 单位：字节
    // 910B单机AIV模式下ReduceScatterV算子当前仅支持单卡数据量不大于256M的场景，大于256M暂不支持
    bool isAivMode = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && topoMatcher_->GetAivModeConfig()
                     && isSingleMeshAggregation_ && maxDataSize <= AIV_BIG_SIZE
                     && IsSupportAIVReduce(param.VDataDes.dataType, param.reduceType)
                     && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE;
    HCCL_INFO(
        "[ReduceScatterVOperator][SelectAlgfor910B]isAivMode[%d], maxCount[%llu], maxDataSize[%llu], "
        "deterministic[%u], isSingleMeshAggregation[%d].",
        isAivMode, maxCount, maxDataSize, topoMatcher_->GetDeterministicConfig(), isSingleMeshAggregation_);

    if (isAivMode) {
        if (maxDataSize > AIV_REDUCE_SCATTER_MID_SIZE) {
            algName = "ReduceScatterVAIVBigCountExecutor";
        } else {
            algName = "ReduceScatterVMeshAivSmallCountExecutor";
        }
    } else if (
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && IsSupportSDMAReduce(
            cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(),
            param.VDataDes.dataType, param.reduceType)) {
        algName = "ReduceScatterVMeshOpbaseExecutor";
        if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB
              || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR)) {
            // 只支持server间ring,NB和NHR，默认使能NHR
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[ReduceScatterVOperator][SelectAlgfor910B] only support ring, NB and NHR in AlgoLevel1 yet,"
                         " default algType is NHR.");
        }
    } else if (
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB
        && IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.VDataDes.dataType, param.reduceType)) {
        algName = "ReduceScatterVMeshExecutor";
    } else {
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlgfor910B] ReduceScatterV only support inlinereduce.");
        return HCCL_E_NOT_SUPPORT;
    }

    HCCL_INFO("[SelectAlgfor910B] ReduceScatterV SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    (void)param;
    CHK_PRT_RET(
        userRankSize_ > MAX_310P_RANK_SIZE,
        HCCL_ERROR(
            "[ReduceScatterVOperator][SelectAlgfor310P3]rankSize[%u] is not supported.ReduceScatterV does not "
            "support the scenario where the rankSize is greater than 4.",
            userRankSize_),
        HCCL_E_NOT_SUPPORT);
    algName = "ReduceScatterVFor310PRing";
    HCCL_INFO("[SelectAlgfor310P3] ReduceScatterV SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, ReduceScatterV, ReduceScatterVOperator);

} // namespace hccl

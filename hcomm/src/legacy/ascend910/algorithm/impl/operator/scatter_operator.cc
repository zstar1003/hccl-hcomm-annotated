/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_operator.h"
#include "coll_alg_utils.h"

namespace hccl {

ScatterOperator::ScatterOperator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_SCATTER)
{
    // 由于scatter只支持server间ring、nb和nhr，其他算法需要重定向到ring
    if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR)
        && !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)
        && !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        HCCL_INFO(
            "[ScatterOperator][ScatterOperator] algType[%s] is not supported, reset algType=ring",
            AlgTypeToStr(algType_).c_str());
    }
}

ScatterOperator::~ScatterOperator() {}

HcclResult
ScatterOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag)
{
    if (isDiffDeviceType_) {
        HCCL_ERROR("[ScatterOperator][SelectAlg] Scatter not support diffDeviceType");
        return HCCL_E_NOT_SUPPORT;
    }
    if (userRankSize_ == 1) {
        algName = "ScatterSingleExecutor";
        HCCL_INFO("[SelectAlg] Scatter SelectAlg is algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }
    newTag = param.tag;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD
            || algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD)) {
        std::string appendTag = "";
        u32 serverNumPerSuperPod = superPodNum_ == 0 ? moduleNum_ : moduleNum_ / superPodNum_;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
            u32 part1Size = FACTOR_TWO * (serverNumPerSuperPod - (1 << static_cast<u32>(log2(serverNumPerSuperPod))));
            u32 rootId = param.root / deviceNumPerAggregation_ % serverNumPerSuperPod;
            appendTag += "L1_" + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        HCCL_DEBUG("[ScatterOperator]SelectAlg for algoLevel1");
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
            u32 part1Size = FACTOR_TWO * (superPodNum_ - (1 << static_cast<u32>(log2(superPodNum_))));
            u32 rootId = param.root / deviceNumPerAggregation_ / serverNumPerSuperPod;
            appendTag += (appendTag.empty() ? "L2_" : "_L2_")
                         + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        HCCL_DEBUG("[ScatterOperator][SelectAlg]tag is [%s]", tag.c_str());
        newTag = newTag + '_' + appendTag;
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
        }
    }

    // 由于scatter只支持server间ring,nb和NHR，如果不是需要重定向到ring；910_93仅支持server间ring
    if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR)
        && !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)
        && !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)) {
        HCCL_INFO(
            "[ScatterOperator][Scatter] algType[%s] is not supported, reset algType=ring",
            AlgTypeToStr(algType_).c_str());
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    }

    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH
                      || topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING
                      || topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    if (multiModuleDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_) {
        algName = "ScatterCommExecutor";
    } else if (isMeshTopo) {
        algName = "ScatterMeshExecutor";
    } else if (isRingTopo) {
        if (deviceType_ == DevType::DEV_TYPE_910_93) {
            algName = "ScatterRingFor91093Executor";
        } else {
            algName = "ScatterRingExecutor";
        }
    } else {
        algName = "ScatterCommExecutor";
    }
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = newTag + algName;
        HCCL_INFO("[SelectAlg] Scatter newTag is [%s] algName is [%s]", newTag.c_str(), algName.c_str());
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_SCATTER, Scatter, ScatterOperator);
} // namespace hccl

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_operator.h"

namespace hccl {

ReduceOperator::ReduceOperator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE)
{
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1
        || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB
        || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        HCCL_WARNING("[ReduceOperator][ReduceOperator] nonuniform-hierarchical-ring and nonuniform-bruck and pipeline "
                     "algorithms do not support Reduce yet, reset algo to halving-doubling");
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
    }
}

ReduceOperator::~ReduceOperator() {}

HcclResult
ReduceOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag)
{
    HcclResult ret = HCCL_SUCCESS;

    if (userRankSize_ == 1) {
        algName = "ReduceSingleExecutor";
        HCCL_INFO("[SelectAlg] Reduce SelectAlg is algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }

    newTag = param.tag;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD
            || algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD)) {
        std::string appendTag = "";
        u32 serverNumPerSuperPod = superPodNum_ == 0 ? moduleNum_ : moduleNum_ / superPodNum_;
        HCCL_DEBUG("[ReduceOperator][SelectAlg]serverNumPerSuperPod is %u", serverNumPerSuperPod);
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
            u32 part1Size = FACTOR_TWO * (serverNumPerSuperPod - (1 << static_cast<u32>(log2(serverNumPerSuperPod))));
            u32 rootId = param.root / deviceNumPerAggregation_ % serverNumPerSuperPod;
            appendTag += "L1_" + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        HCCL_DEBUG("[ReduceOperator]SelectAlg for algoLevel1");
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
            u32 part1Size = FACTOR_TWO * (superPodNum_ - (1 << static_cast<u32>(log2(superPodNum_))));
            u32 rootId = param.root / deviceNumPerAggregation_ / serverNumPerSuperPod;
            appendTag += (appendTag.empty() ? "L2_" : "_L2_")
                         + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        HCCL_DEBUG("[ReduceOperator][SelectAlg]tag is [%s]", tag.c_str());
        newTag = newTag + '_' + appendTag;
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
        }
    }

    if (isDiffDeviceType_) {
        HCCL_ERROR("[ReduceOperator][SelectAlg] Reduce not support diffDeviceType");
        return HCCL_E_NOT_SUPPORT;
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else {
        HCCL_ERROR("ReduceOperator[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceSelector][SelectAlg]tag[%s], reduce failed, return[%d]", tag.c_str(), ret), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType_.algoLevel1);
        CHK_PRT_RET(
            level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
            HCCL_ERROR("level1: algType1[%u] is invalid.", algType_.algoLevel1), HCCL_E_INTERNAL);
        newTag = newTag + level1Iter->second + algName;
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}

HcclResult ReduceOperator::SelectAlgfor910A(const OpParam& param, std::string& algName) const
{
    (void)param;
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "ReduceMeshExecutor";
    } else if (isRingTopo) {
        algName = "ReduceRingPlusHd";
    } else {
        algName = "ReduceComm";
    }

    HCCL_INFO("[SelectAlgfor910A] reduce SelectAlgfor910A is algName[%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceOperator::SelectAlgfor910B(const OpParam& param, std::string& algName) const
{
    (void)param;
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH
                      || topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    if (isMeshTopo) {
        algName = "ReduceMeshExecutor";
    } else if (isRingTopo) {
        algName = "ReduceRingPlusHd";
    } else {
        algName = "ReduceComm";
    }

    HCCL_INFO("[SelectAlgfor910B] reduce SelectAlgfor910B is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
{
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;
    bool isDoubleRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    if (dataSize >= cclBufferManager_.GetInCCLbufferSize()) {
        HCCL_WARNING(
            "The current inCCLbufferSize is [%llu] bytes, change the HCCL_BUFFSIZE environment variable "
            "to be greater than the current data volume[%llu] bytes to improve the performance of the 91093 "
            "environment.",
            cclBufferManager_.GetInCCLbufferSize(), dataSize);
    }

    if (multiModuleDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_) {
        algName = "ReduceComm";
    } else if (isRingTopo || isDoubleRingTopo) {
        algName = "ReduceRingFor91093Executor";
    } else {
        algName = "ReduceComm";
    }
    if ((!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)
         && !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD))
        || (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) && topoMatcher_->GetTopoInfo().superPodNum > 1)
        || !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING)) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        HCCL_WARNING("[ReduceOperator][SelectAlgfor91093][Superpod] inter-server only support ring yet, "
                     "default is algType=RING.");
    }
    HCCL_INFO("[SelectAlgfor91093] reduce SelectAlgfor91093 is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_REDUCE, Reduce, ReduceOperator);

} // namespace hccl

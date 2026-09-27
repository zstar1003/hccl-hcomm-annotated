/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "mc2_selector.h"
#include "selector_registry.h"
#include "coll_operator.h"
#include "coll_alg_params.h"

namespace Hccl {
const std::map<OpType, std::string> MC2_CCU_1D_DEFAULT_ALG_MAP = {
    {OpType::ALLGATHER, "CcuAllGatherMesh1D"},        {OpType::REDUCESCATTER, "CcuReduceScatterMesh1D"},
    {OpType::ALLREDUCE, "CcuAllReduceMesh1D"},        {OpType::REDUCE, "CcuReduceMesh1D"},
    {OpType::ALLTOALL, "CcuAlltoAllMesh1D"},          {OpType::ALLTOALLV, "CcuAlltoAllVMesh1D"},
    {OpType::HALFALLTOALLV, "CcuHalfAll2AllVMesh1D"},
};

const std::map<OpType, std::string> MC2_CCU_SCHED_1D_DEFAULT_ALG_MAP = {
    {OpType::ALLGATHER, "CcuAllGatherMeshMem2Mem1D"}, {OpType::REDUCESCATTER, "CcuReduceScatterMeshMem2Mem1D"},
    {OpType::ALLREDUCE, "CcuAllReduceMeshMem2Mem1D"}, {OpType::ALLTOALL, "CcuAlltoAllMesh1D"},
    {OpType::ALLTOALLV, "CcuAlltoAllVMesh1D"},        {OpType::HALFALLTOALLV, "CcuHalfAll2AllVMesh1D"},
};

const std::map<OpType, std::string> MC2_CCU_2D_DEFAULT_ALG_MAP = {
    {OpType::ALLGATHER, "CcuAllGatherMesh2D"},        {OpType::REDUCESCATTER, "CcuReduceScatterMesh2D"},
    {OpType::ALLREDUCE, "CcuAllReduceMesh2DOneShot"}, {OpType::REDUCE, "CcuReduceMesh2D"},
    {OpType::ALLTOALL, "CcuAlltoAllMesh2D"},
};

const std::map<OpType, std::string> MC2_AICPU_1D_DEFAULT_ALG_MAP = {
    {OpType::ALLGATHER, "InsAllGatherMesh"},
    {OpType::REDUCESCATTER, "InsReduceScatterNHR"},
    {OpType::ALLREDUCE, "InsAllReduceNHR"},
    {OpType::REDUCE, "InsReduceNHR"},
    {OpType::ALLTOALL, "InsAlltoAllMesh"},
    {OpType::ALLTOALLV, "InsAlltoAllvMesh"},
    {OpType::BATCHSENDRECV, "InsBatchSendRecv"},
    {OpType::BROADCAST, "InsBroadcastNHR"},
    {OpType::SCATTER, "InsScatterNHR"},
    {OpType::SEND, "InsSend"},
    {OpType::RECV, "InsRecv"},
};

AlgorithmType Mc2Selector::GetAlgorithmTypeForMC2CCU(const std::string& name)
{
    auto it = algorithmMap_.find(name);
    if (it == algorithmMap_.end()) {
        THROW<InvalidParamsException>(StringFormat("Unknown algorithm name: [%s] ", name.c_str()));
    }
    return it->second;
}

const std::map<OpType, std::string> MC2_AICPU_2D_DEFAULT_ALG_MAP = {};

SelectorStatus Mc2Selector::SelectDefaultCcuMsAlgo(
    const CollAlgOperator& op, const CollAlgParams& params, std::string& primQueueGenName) const
{
    (void)params;
    TopoInfo topoInfo;
    CalcTopoShape(topoInfo);
    std::map<OpType, string> algMap;
    if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
        algMap = MC2_CCU_1D_DEFAULT_ALG_MAP;
    } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
        algMap = MC2_CCU_2D_DEFAULT_ALG_MAP;
    } else {
        HCCL_ERROR("[Algo][Mc2Selector][SelectDefaultCcuMsAlgo] only support 1D mesh and 2D mesh algo.");
        return SelectorStatus::NOT_MATCH;
    }
    auto it = algMap.find(op.opType);
    if (it != algMap.end()) {
        primQueueGenName = it->second;
    } else {
        HCCL_ERROR(
            "[Algo][Mc2Selector][SelectDefaultCcuMsAlgo] op.opType[%s] Level0Shape[%d] does not have any default mc2 "
            "algo.",
            op.opType.Describe().c_str(), topoInfo.level0Shape);
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus Mc2Selector::SelectDefaultCcuSchedAlgo(
    const CollAlgOperator& op, const CollAlgParams& params, std::string& primQueueGenName) const
{
    (void)params;
    TopoInfo topoInfo;
    CalcTopoShape(topoInfo);
    std::map<OpType, string> algMap;
    if (topoInfo.level0Shape == Level0Shape::MESH_1D || topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
        algMap = MC2_CCU_SCHED_1D_DEFAULT_ALG_MAP;
    } else {
        HCCL_ERROR("[Algo][Mc2Selector][SelectDefaultCcuSchedAlgo] only support 1D mesh algo.");
        return SelectorStatus::NOT_MATCH;
    }
    auto it = algMap.find(op.opType);
    if (it != algMap.end()) {
        primQueueGenName = it->second;
    } else {
        HCCL_ERROR(
            "[Algo][Mc2Selector][SelectDefaultCcuSchedAlgo] op.opType[%s] Level0Shape[%d] does not have any default "
            "mc2 algo.",
            op.opType.Describe().c_str(), topoInfo.level0Shape);
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus Mc2Selector::SelectDefaultAicpuAlgo(
    const CollAlgOperator& op, const CollAlgParams& params, std::string& primQueueGenName) const
{
    (void)params;
    TopoInfo topoInfo;
    CalcTopoShape(topoInfo);
    std::map<OpType, string> algMap;
    if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
        algMap = MC2_AICPU_1D_DEFAULT_ALG_MAP;
    } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
        algMap = MC2_AICPU_2D_DEFAULT_ALG_MAP;
    } else {
        HCCL_ERROR("[Algo][Mc2Selector][SelectDefaultAicpuAlgo] only support 1D mesh and 2D mesh algo.");
        return SelectorStatus::NOT_MATCH;
    }
    auto it = algMap.find(op.opType);
    if (it != algMap.end()) {
        primQueueGenName = it->second;
    } else {
        HCCL_ERROR(
            "[Algo][Mc2Selector][SelectDefaultAicpuAlgo] op.opType[%s] Level0Shape[%d] does not have any default mc2 "
            "algo.",
            op.opType.Describe().c_str(), topoInfo.level0Shape);
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus
Mc2Selector::SelectCcuMsAlgo(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName) const
{
    // 校验 algConfig 是否为空
    if (params.algConfig.empty()) {
        HCCL_INFO("[Algo][Mc2Selector][SelectCcuMsAlgo] algConfig is [%s].", params.algConfig.c_str());
        // 没有配置算法类型，返回默认算法
        HCCL_INFO("[Algo][Mc2Selector][SelectCcuMsAlgo] MC2 CCU MS does not support algConfig yet.");
    }

    // 当前 ccu 模式只有默认算法选择，不支持配置 algConfig
    return SelectDefaultCcuMsAlgo(op, params, primQueueGenName);
}

SelectorStatus
Mc2Selector::SelectCcuSchedAlgo(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName) const
{
    // 校验 algConfig 是否为空
    if (params.algConfig.empty()) {
        HCCL_INFO("[Algo][Mc2Selector][SelectCcuSchedAlgo] algConfig is [%s].", params.algConfig.c_str());
        // 没有配置算法类型，返回默认算法
        HCCL_INFO("[Algo][Mc2Selector][SelectCcuSchedAlgo] MC2 CCU Sched does not support algConfig yet.");
    }

    // 当前 ccu 模式只有默认算法选择，不支持配置 algConfig
    return SelectDefaultCcuSchedAlgo(op, params, primQueueGenName);
}

SelectorStatus
Mc2Selector::SelectAicpuAlgo(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName) const
{
    // 校验 algConfig 是否为空
    if (params.algConfig.empty()) {
        HCCL_INFO("[Algo][Mc2Selector][SelectAicpuAlgo] algConfig is [%s].", params.algConfig.c_str());
        // 没有配置算法类型，返回默认算法
        HCCL_INFO("[Algo][Mc2Selector][SelectAicpuAlgo] MC2 AICPU does not support algConfig yet.");
    }

    // 当前 ccu 模式只有默认算法选择，不支持配置 algConfig
    return SelectDefaultAicpuAlgo(op, params, primQueueGenName);
}

SelectorStatus Mc2Selector::Select(const CollAlgOperator& op, CollAlgParams& params, std::string& primQueueGenName)
{
    if (rankGraph_ == nullptr) {
        HCCL_ERROR("[Algo][Mc2Selector] rankGraph_ is nullptr.");
        return SelectorStatus::NOT_MATCH;
    }

    if (params.opExecuteConfig.accState == AcceleratorState::CCU_MS) {
        return SelectCcuMsAlgo(op, params, primQueueGenName);
    } else if (params.opExecuteConfig.accState == AcceleratorState::CCU_SCHED) {
        return SelectCcuSchedAlgo(op, params, primQueueGenName);
    } else if (params.opExecuteConfig.accState == AcceleratorState::AICPU_TS) {
        return SelectAicpuAlgo(op, params, primQueueGenName);
    } else {
        // 当前 MC2 场景不支持回退，当遇到不支持的 AcceleratorState 时直接报错
        HCCL_ERROR(
            "[Algo][Mc2Selector] AcceleratorState[%s] is not supported, match failed",
            params.opExecuteConfig.accState.Describe().c_str());
        return SelectorStatus::NOT_MATCH;
    }
}

REGISTER_SELECTOR(18, Mc2Selector);
} // namespace Hccl

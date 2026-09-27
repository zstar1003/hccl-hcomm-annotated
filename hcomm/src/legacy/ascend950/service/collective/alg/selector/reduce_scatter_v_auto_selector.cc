/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_v_auto_selector.h"
#include "selector_registry.h"
#include "coll_operator.h"

namespace Hccl {
SelectorStatus ReduceScatterVAutoSelector::SelectCcuMsAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[ReduceScatterVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    // MS 模式不支持 int8
    CHK_PRT_RET(
        op.dataType == DataType::INT8,
        HCCL_WARNING(
            "[Algo][ReduceScatterVAutoSelector] dataType[%s] is not supported yet for ccu_ms mode.",
            op.dataType.Describe().c_str()),
        SelectorStatus::NOT_MATCH);

    if (topoInfo.levelNum > 1) {
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (Is2DieFullMesh()) {
                HCCL_WARNING(
                    "[Algo][ReduceScatterVAutoSelector] 2DieFullMesh[%d] is not supported yet for ccu_ms mode.",
                    topoInfo.level0Shape);
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuReduceScatterVMesh1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            HCCL_WARNING(
                "[Algo][ReduceScatterVAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                primQueueGenName = "CcuReduceScatterVMesh1D";
            } else if (topoInfo.level0PcieMix) {
                HCCL_WARNING(
                    "[Algo][ReduceScatterVAutoSelector] level0 PCIE mix is not supported yet for ccu_ms mode.");
                return SelectorStatus::NOT_MATCH;
            } else { // MS 不支持
                HCCL_WARNING(
                    "[Algo][ReduceScatterVAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                    topoInfo.level0Shape);
                return SelectorStatus::NOT_MATCH;
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][ReduceScatterVAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING(
                "[Algo][ReduceScatterVAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][ReduceScatterVAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterVAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfo& topoInfo, [[maybe_unused]] const CollAlgOperator& op,
    [[maybe_unused]] const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap,
    std::string& primQueueGenName) const
{
    HCCL_DEBUG("[ReduceScatterVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    if (topoInfo.levelNum > 1) {
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] levelNum > 1 is not supported yet for ccu_schedule mode.");
        return SelectorStatus::NOT_MATCH;
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (Is2DieFullMesh()) {
                HCCL_WARNING(
                    "[Algo][ReduceScatterVAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuReduceScatterVMeshMem2Mem1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            HCCL_WARNING(
                "[Algo][ReduceScatterVAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                primQueueGenName = "CcuReduceScatterVMeshMem2Mem1D";
            } else if (topoInfo.level0PcieMix) {
                HCCL_WARNING(
                    "[Algo][ReduceScatterVAutoSelector] level0 PCIE mix is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                HCCL_WARNING(
                    "[Algo][ReduceScatterVAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                    topoInfo.level0Shape);
                return SelectorStatus::NOT_MATCH;
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][ReduceScatterVAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING(
                "[Algo][ReduceScatterVAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }

    HCCL_INFO("[Algo][ReduceScatterVAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterVAutoSelector::SelectAicpuAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)topoInfo;
    (void)op;
    (void)configAlgMap;
    (void)primQueueGenName;

    // 暂时没有 aicpu 算法
    HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] No aicpu algorithm for aicpu mode. Auto select failed.");
    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(OpType::REDUCESCATTERV, 18, ReduceScatterVAutoSelector);
} // namespace Hccl

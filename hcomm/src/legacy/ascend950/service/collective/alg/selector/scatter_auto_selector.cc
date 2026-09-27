/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_auto_selector.h"
#include "selector_registry.h"
#include "coll_operator.h"

namespace Hccl {
SelectorStatus ScatterAutoSelector::SelectCcuMsAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)topoInfo;
    (void)op;
    (void)configAlgMap;
    (void)primQueueGenName;
    HCCL_WARNING("[Algo][ScatterAutoSelector] not supported yet for ccu_ms mode, reset to default.");
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus ScatterAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[ScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    if (topoInfo.levelNum > 1) {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (topoInfo.netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                // 每框出 1 卡
                primQueueGenName = "CcuScatterNHRMem2Mem1D";
            } else if (Is2DieFullMesh()) {
                HCCL_WARNING("[Algo][ScatterAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuScatterParallelMesh1DNHR";
            }
        } else {
            HCCL_WARNING(
                "[Algo][ScatterAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (Is2DieFullMesh()) {
                HCCL_WARNING("[Algo][ScatterAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuScatterMesh1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "CcuScatterMesh2D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                primQueueGenName = "CcuScatterMesh1D";
            } else if (topoInfo.level0PcieMix) {
                HCCL_WARNING("[Algo][ScatterAutoSelector] level0 PCIE mix is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuAllGatherParallelMesh1DNHR";
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][ScatterAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING(
                "[Algo][ScatterAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][ScatterAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ScatterAutoSelector::SelectAicpuAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[ScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    if (topoInfo.levelNum > 1) {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (topoInfo.netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                primQueueGenName = "InsScatterNHR";
            } else {
                primQueueGenName = "InsScatterParallelMesh1DNHR";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            HCCL_WARNING(
                "[Algo][ScatterAutoSelector] level0Shape[%d] is not supported yet for aicpu levelNum > 1.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][ScatterAutoSelector] level0Shape[%d] is not supported yet for aicpu levelNum > 1.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            primQueueGenName = "InsScatterMesh1D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "InsScatterMesh2D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                primQueueGenName = "InsScatterMesh1D";
            } else {
                if (topoInfo.level0PcieMix) {
                    // 预留PCIE mix入口，如果要更新算法可以直接改
                    primQueueGenName = "InsScatterParallelMesh1DNHRPcie";
                }
                primQueueGenName = "InsScatterMesh1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            if (topoInfo.level0PcieMix) {
                primQueueGenName = "InsScatterNHR";
            } else {
                primQueueGenName = "InsScatterMesh1D";
            }
        } else {
            HCCL_WARNING("[ScatterAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][ScatterAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ScatterAutoSelector::SelectAivAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[ScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    // aiv 直接走打平 mesh
    primQueueGenName = "AivScatterMesh1D";

    HCCL_INFO("[Algo][ScatterAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(OpType::SCATTER, 18, ScatterAutoSelector);
} // namespace Hccl

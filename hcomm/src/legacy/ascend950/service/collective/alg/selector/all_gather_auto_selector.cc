/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_auto_selector.h"
#include "selector_registry.h"
#include "coll_operator.h"

namespace Hccl {

SelectorStatus AllGatherAutoSelector::SelectCcuMsAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);
    u32 rankSize_2P = 2;
    u32 rankSize_4P = 4;
    HcclDetourType detourType = EnvConfig::GetInstance().GetDetourConfig().GetDetourType();
    CHK_PRT_RET(
        (detourType == HcclDetourType::HCCL_DETOUR_ENABLE_2P && rankSize_ != rankSize_2P)
            || (detourType == HcclDetourType::HCCL_DETOUR_ENABLE_4P && rankSize_ != rankSize_4P),
        HCCL_WARNING("[Algo][AllGatherAutoSelector] detourType not match for rankSize."), SelectorStatus::NOT_MATCH);

    CHK_PRT_RET(
        detourType == HcclDetourType::HCCL_DETOUR_ENABLE_2P_AND_4P,
        HCCL_WARNING("[Algo][AllGatherAutoSelector] HCCL_DETOUR_ENABLE_2P_AND_4P is not supported yet."),
        SelectorStatus::NOT_MATCH);

    if (topoInfo.levelNum > 1) {
        HCCL_WARNING("[Algo][AllGatherAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (Is2DieFullMesh()) {
                primQueueGenName = "CcuAllGatherMesh1D2Die";
            } else if (
                (detourType == HcclDetourType::HCCL_DETOUR_ENABLE_2P && rankSize_ == rankSize_2P)
                || (detourType == HcclDetourType::HCCL_DETOUR_ENABLE_4P && rankSize_ == rankSize_4P)) {
                primQueueGenName = "CcuAllGatherMeshDetour1D";
            } else {
                primQueueGenName = "CcuAllGatherMesh1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "CcuAllGatherMesh2D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                if ((detourType == HcclDetourType::HCCL_DETOUR_ENABLE_2P && rankSize_ == rankSize_2P)
                    || (detourType == HcclDetourType::HCCL_DETOUR_ENABLE_4P && rankSize_ == rankSize_4P)) {
                    primQueueGenName = "CcuAllGatherMeshDetour1D";
                } else if (topoInfo.level0PcieMix) {
                    HCCL_WARNING("[Algo][AllGatherAutoSelector] level0 PCIE mix is not supported yet for ccu_ms mode.");
                    return SelectorStatus::NOT_MATCH;
                } else {
                    primQueueGenName = "CcuAllGatherMesh1D";
                }
            } else { // MS 不支持
                HCCL_WARNING(
                    "[Algo][AllGatherAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                    topoInfo.level0Shape);
                return SelectorStatus::NOT_MATCH;
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][AllGatherAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING(
                "[Algo][AllGatherAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][AllGatherAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    if (topoInfo.levelNum > 1) {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (topoInfo.netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                // 每框出 1 卡
                primQueueGenName = "CcuAllGatherNHR1D";
            } else if (Is2DieFullMesh()) {
                HCCL_WARNING("[Algo][AllGatherAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuAllGatherParallelMesh1DNHR";
            }
        } else {
            HCCL_WARNING(
                "[Algo][AllGatherAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (Is2DieFullMesh()) {
                HCCL_WARNING("[Algo][AllGatherAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuAllGatherMeshMem2Mem1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "CcuAllGatherMeshMem2Mem2D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                primQueueGenName = "CcuAllGatherMeshMem2Mem1D";
            } else if (topoInfo.level0PcieMix) {
                HCCL_WARNING(
                    "[Algo][AllGatherAutoSelector] level0 PCIE mix is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuAllGatherParallelMesh1DNHR";
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][AllGatherAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING(
                "[Algo][AllGatherAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][AllGatherAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectAicpuAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    if (topoInfo.levelNum > 1) {
        if (topoInfo.Level1Nhr) {
            primQueueGenName = "InsAllGatherNHR";
        } else if (topoInfo.Level0Nhr) {
            primQueueGenName = "InsAllGatherParallelNHRNHR";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (topoInfo.netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                primQueueGenName = "InsAllGatherNHR";
            } else {
                primQueueGenName = "InsAllGatherParallelMesh1DNHR";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "InsAllGatherParallelMesh2DNHR";
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            primQueueGenName = "InsAllGatherParallelNHRNHR";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            primQueueGenName = "InsAllGatherMesh";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "InsAllGatherMesh2D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                primQueueGenName = "InsAllGatherMesh";
            } else if (topoInfo.level0PcieMix) {
                primQueueGenName = "InsAllGatherParallelMesh1DNHRPcie";
            } else {
                primQueueGenName = "InsAllGatherParallelMesh1DNHR";
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            primQueueGenName = "InsAllGatherNHR";
        } else {
            HCCL_WARNING("[AllGatherAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][AllGatherAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectAivAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    // aiv 直接走打平 mesh
    primQueueGenName = "AivAllGatherMesh1D";

    HCCL_INFO("[Algo][AllGatherAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(OpType::ALLGATHER, 18, AllGatherAutoSelector);
} // namespace Hccl

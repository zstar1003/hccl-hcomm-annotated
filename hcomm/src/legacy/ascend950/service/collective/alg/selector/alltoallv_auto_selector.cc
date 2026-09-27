/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_auto_selector.h"
#include "selector_registry.h"
#include "coll_operator.h"

namespace Hccl {

SelectorStatus AlltoAllVAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    if (topoInfo.levelNum > 1) {
        HCCL_WARNING("[Algo][AlltoAllVAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (Is2DieFullMesh()) {
                primQueueGenName = "CcuAlltoAllVMesh2Die";
            } else {
                primQueueGenName = "CcuAlltoAllVMesh1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "CcuAlltoAllVMesh2D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                if (Is2DieFullMesh()) {
                    primQueueGenName = "CcuAlltoAllVMesh2Die";
                } else {
                    primQueueGenName = "CcuAlltoAllVMesh1D";
                }
            } else if (topoInfo.level0PcieMix) {
                HCCL_WARNING(
                    "[Algo][AlltoAllVAutoSelector] level0 PCIE mix is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuAlltoAllVMesh1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][AlltoAllVAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING(
                "[Algo][AlltoAllVAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][AlltoAllVAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllVAutoSelector::SelectAicpuAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    // 目前只有 InsAlltoAllvMesh 算法
    if (topoInfo.levelNum > 1) {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            primQueueGenName = "InsAlltoAllvMesh";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "InsAlltoAllvMesh";
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            primQueueGenName = "InsAlltoAllvMesh";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            primQueueGenName = "InsAlltoAllvMesh";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "InsAlltoAllvMesh";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                primQueueGenName = "InsAlltoAllvMesh";
            } else if (topoInfo.level0PcieMix) {
                primQueueGenName = "InsAlltoAllvMesh";
            } else {
                primQueueGenName = "InsAlltoAllvMesh";
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            primQueueGenName = "InsAlltoAllvMesh";
        } else {
            HCCL_WARNING("[AlltoAllVAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][AlltoAllVAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllVAutoSelector::SelectAivAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    // aiv 直接走打平 mesh
    primQueueGenName = "AivAlltoAllVMesh1D";

    HCCL_INFO("[Algo][AlltoAllVAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(OpType::ALLTOALLV, 18, AlltoAllVAutoSelector);
} // namespace Hccl

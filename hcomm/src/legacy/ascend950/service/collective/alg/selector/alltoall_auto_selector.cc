/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoall_auto_selector.h"
#include "selector_registry.h"
#include "coll_operator.h"

namespace Hccl {
SelectorStatus AlltoAllAutoSelector::SelectCcuMsAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)topoInfo;
    (void)op;
    (void)configAlgMap;
    (void)primQueueGenName;
    HCCL_WARNING("[Algo][AlltoAllAutoSelector] is not supported yet for ccu_ms mode, reset to default.");
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectCcuScheduleAlgoLevel1(
    const TopoInfo& topoInfo, const CollAlgOperator& op, std::string& primQueueGenName) const
{
    if (isMc2_) {
        HCCL_WARNING("[Algo][AlltoAllAutoSelector] levelNum > 1 is not supported yet for ccu_schedule mode.");
        return SelectorStatus::NOT_MATCH;
    }
    if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
        if (topoInfo.netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            primQueueGenName = "CcuAlltoAllMesh1D";
        } else {
            if (op.dataType == DataType::INT8) {
                HCCL_WARNING("[Algo][AlltoAllAutoSelector] int8 is not supported yet for ccu_schedule mode.");
                return SelectorStatus::NOT_MATCH;
            }
            primQueueGenName = "CcuAllToAllMesh1D2Die";
        }
    } else {
        HCCL_WARNING(
            "[Algo][AlltoAllAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
            topoInfo.level0Shape);
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[AlltoAllAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    if (topoInfo.levelNum > 1) {
        SelectorStatus ret = SelectCcuScheduleAlgoLevel1(topoInfo, op, primQueueGenName);
        return ret;
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (Is2DieFullMesh()) {
                primQueueGenName = "CcuAllToAllMesh1D2Die";
            } else {
                primQueueGenName = "CcuAlltoAllMesh1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "CcuAlltoAllMesh2D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                if (Is2DieFullMesh()) {
                    primQueueGenName = "CcuAllToAllMesh1D2Die";
                } else {
                    primQueueGenName = "CcuAlltoAllMesh1D";
                }
            } else if (topoInfo.level0PcieMix) {
                HCCL_WARNING(
                    "[Algo][AlltoAllAutoSelector] level0 PCIE mix is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                HCCL_WARNING(
                    "[Algo][AlltoAllAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                    topoInfo.level0Shape);
                return SelectorStatus::NOT_MATCH;
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][AlltoAllAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING(
                "[Algo][AlltoAllAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][AlltoAllAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectAicpuAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[AlltoAllAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    if (topoInfo.levelNum > 1) {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            primQueueGenName = "InsAlltoAllMesh";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "InsAlltoAllMesh";
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            primQueueGenName = "InsAlltoAllMesh";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            primQueueGenName = "InsAlltoAllMesh";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "InsAlltoAllMesh2D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                primQueueGenName = "InsAlltoAllMesh";
            } else if (topoInfo.level0PcieMix) {
                primQueueGenName = "InsAlltoAllMesh";
            } else {
                primQueueGenName = "InsAlltoAllMesh";
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            primQueueGenName = "InsAlltoAllMesh";
        } else {
            HCCL_WARNING("[AlltoAllAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][AlltoAllAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectAivAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)op;
    (void)configAlgMap;
    HCCL_DEBUG("[AlltoAllAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    // aiv 直接走打平 mesh
    primQueueGenName = "AivAlltoAllMesh1D";

    HCCL_INFO("[Algo][AlltoAllAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(OpType::ALLTOALL, 18, AlltoAllAutoSelector);
} // namespace Hccl

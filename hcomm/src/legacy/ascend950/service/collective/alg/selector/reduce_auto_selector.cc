/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_auto_selector.h"
#include "selector_registry.h"
#include "coll_operator.h"

namespace Hccl {

constexpr u64 REDUCE_AICPU_1D_MAX_DATA_SIZE = 16 * 1024 * 1024;
constexpr u64 REDUCE_CCU_1D_MAX_DATA_SIZE = 64 * 1024 * 1024;
constexpr u64 REDUCE_CCU_1D_MAX_DATA_SIZE_INT8 = 16 * 1024 * 1024;

SelectorStatus ReduceAutoSelector::SelectCcuMsAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);
    // MS 模式不支持 int8
    CHK_PRT_RET(
        op.dataType == DataType::INT8,
        HCCL_WARNING(
            "[Algo][ReduceAutoSelector] dataType[%s] is not supported yet for ccu_ms mode.",
            op.dataType.Describe().c_str()),
        SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    CHK_PRT_RET(
        op.reduceOp == ReduceOp::PROD,
        HCCL_WARNING(
            "[Algo][ReduceAutoSelector] ReduceOp[%s] is not supported yet for ccu_ms mode.",
            op.reduceOp.Describe().c_str()),
        SelectorStatus::NOT_MATCH);

    CHK_PRT_RET(
        op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64,
        HCCL_WARNING("[Algo][ReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64."),
        SelectorStatus::NOT_MATCH);

    if (topoInfo.levelNum > 1) {
        HCCL_WARNING("[Algo][ReduceAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (IsInputOutputOverlap(op.inputMem, op.outputMem) == true) {
                // 不支持 inplace 场景
                return SelectorStatus::NOT_MATCH;
            }
            if (Is2DieFullMesh()) {
                HCCL_WARNING("[Algo][ReduceAutoSelector] 2DieFullMesh is not supported yet for ccu_ms mode.");
                return SelectorStatus::NOT_MATCH;
            } else if (dataSize_ >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
                HCCL_INFO("[Algo][ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize_);
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuReduceMesh1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "CcuReduceMesh2D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                if (dataSize_ >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
                    HCCL_INFO("[Algo][ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize_);
                    return SelectorStatus::NOT_MATCH;
                } else {
                    primQueueGenName = "CcuReduceMesh1D";
                }
            } else if (topoInfo.level0PcieMix) {
                HCCL_WARNING("[Algo][ReduceAutoSelector] level0 PCIE mix is not supported yet for ccu_ms mode.");
                return SelectorStatus::NOT_MATCH;
            } else { // MS 不支持
                HCCL_WARNING(
                    "[Algo][ReduceAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                    topoInfo.level0Shape);
                return SelectorStatus::NOT_MATCH;
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][ReduceAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING(
                "[Algo][ReduceAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][ReduceAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);
    // ccu 模式不支持 inplace 场景
    CHK_PRT_RET(
        IsInputOutputOverlap(op.inputMem, op.outputMem) == true,
        HCCL_WARNING("[Algo][ReduceAutoSelector] ccu schedule does not support inplace allreduce."),
        SelectorStatus::NOT_MATCH);

    // ccu 模式不支持 PROD
    CHK_PRT_RET(
        op.reduceOp == ReduceOp::PROD,
        HCCL_WARNING(
            "[Algo][ReduceAutoSelector] ReduceOp[%s] is not supported yet for ccu schedule mode.",
            op.reduceOp.Describe().c_str()),
        SelectorStatus::NOT_MATCH);

    CHK_PRT_RET(
        op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64,
        HCCL_WARNING("[Algo][ReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64."),
        SelectorStatus::NOT_MATCH);

    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    if (topoInfo.levelNum > 1) {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (topoInfo.netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                // 每框出 1 卡
                primQueueGenName = "CcuReduceNHR1D";
            } else if (Is2DieFullMesh()) {
                HCCL_WARNING("[Algo][ReduceAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuReduceParallelMesh1DNHR";
            }
        } else {
            HCCL_WARNING(
                "[Algo][ReduceAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            double ratio;
            if (rankSize_ == 0) {
                HCCL_WARNING("[Algo][ReduceAutoSelector] the selector is not set RankSize_");
                ratio = 1;
            } else {
                if (op.dataType == DataType::INT8) {
                    ratio = DEFAULT_RANK_SIZE / rankSize_;
                } else {
                    ratio = (DEFAULT_RANK_SIZE / rankSize_) * (DEFAULT_RANK_SIZE / rankSize_);
                }
            }
            if (Is2DieFullMesh()) {
                HCCL_WARNING("[Algo][ReduceAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            }
            if (op.dataType == DataType::INT8) {
                if (dataSize_ * ratio >= REDUCE_CCU_1D_MAX_DATA_SIZE_INT8) {
                    HCCL_INFO("[Algo][ReduceAutoSelector] fallback to aicpu.");
                    return SelectorStatus::NOT_MATCH;
                } else {
                    primQueueGenName = "CcuReduceMeshMem2Mem1D";
                }
            } else {
                if (dataSize_ * ratio >= REDUCE_CCU_1D_MAX_DATA_SIZE) {
                    HCCL_INFO("[Algo][ReduceAutoSelector] fallback to aicpu.");
                    return SelectorStatus::NOT_MATCH;
                } else {
                    primQueueGenName = "CcuReduceMeshTwoShotMem2Mem1D";
                }
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "CcuReduceMeshMem2Mem2D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                double ratio;
                if (rankSize_ == 0) {
                    HCCL_WARNING("[Algo][ReduceAutoSelector] the selector is not set RankSize_");
                    ratio = 1;
                } else {
                    if (op.dataType == DataType::INT8) {
                        ratio = DEFAULT_RANK_SIZE / rankSize_;
                    } else {
                        ratio = (DEFAULT_RANK_SIZE / rankSize_) * (DEFAULT_RANK_SIZE / rankSize_);
                    }
                }
                if (op.dataType == DataType::INT8) {
                    if (dataSize_ * ratio >= REDUCE_CCU_1D_MAX_DATA_SIZE_INT8) {
                        HCCL_INFO("[Algo][ReduceAutoSelector] fallback to aicpu.");
                        return SelectorStatus::NOT_MATCH;
                    } else {
                        primQueueGenName = "CcuReduceMeshMem2Mem1D";
                    }
                } else {
                    if (dataSize_ * ratio >= REDUCE_CCU_1D_MAX_DATA_SIZE) {
                        HCCL_INFO("[Algo][ReduceAutoSelector] fallback to aicpu.");
                        return SelectorStatus::NOT_MATCH;
                    } else {
                        primQueueGenName = "CcuReduceMeshTwoShotMem2Mem1D";
                    }
                }
            } else if (topoInfo.level0PcieMix) {
                HCCL_WARNING("[Algo][ReduceAutoSelector] level0 PCIE mix is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                primQueueGenName = "CcuReduceNHR1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][ReduceAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING(
                "[Algo][ReduceAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][ReduceAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectAicpuAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    if (topoInfo.levelNum > 1) {
        CHK_PRT_RET(
            op.reduceOp == ReduceOp::PROD,
            HCCL_WARNING(
                "[Algo][ReduceAutoSelector] ReduceOp[%s] is not supported yet for aicpu levelNum > 1.",
                op.reduceOp.Describe().c_str()),
            SelectorStatus::NOT_MATCH);

        CHK_PRT_RET(
            op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64,
            HCCL_WARNING("[Algo][ReduceAutoSelector] aicpu levelNum > 1 not support INT64, UINT64, FP64."),
            SelectorStatus::NOT_MATCH);

        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (topoInfo.netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                primQueueGenName = "InsReduceNHR";
            } else {
                primQueueGenName = "InsReduceParallelMesh1DNHR";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "InsReduceNHR";
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            primQueueGenName = "InsReduceNHR";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64
                || op.reduceOp == ReduceOp::PROD) {
                primQueueGenName = "InsReduceAicpuReduce";
            } else if (dataSize_ >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
                primQueueGenName = "InsReduceMesh1DTwoShot";
            } else {
                primQueueGenName = "InsReduceMesh1D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            if (op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64
                || op.reduceOp == ReduceOp::PROD) {
                primQueueGenName = "InsReduceAicpuReduceMesh2D";
            } else {
                primQueueGenName = "InsReduceMesh2D";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                if (op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64
                    || op.reduceOp == ReduceOp::PROD) {
                    primQueueGenName = "InsReduceAicpuReduce";
                } else {
                    primQueueGenName = "InsReduceParallelMesh1DNHR";
                }
            } else {
                if (topoInfo.level0PcieMix) {
                    // 预留PCIE mix入口，如果要更新算法可以直接改
                    primQueueGenName = "InsReduceParallelMesh1DNHRPcie";
                }
                if (op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64
                    || op.reduceOp == ReduceOp::PROD) {
                    primQueueGenName = "InsReduceAicpuReduce";
                } else {
                    primQueueGenName = "InsReduceNHR";
                }
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            if (op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64
                || op.reduceOp == ReduceOp::PROD) {
                primQueueGenName = "InsReduceAicpuReduce";
            } else {
                primQueueGenName = "InsReduceNHR";
            }
        } else {
            HCCL_WARNING("[Algo][ReduceAutoSelector] level0Shape[%d] is not supported yet.", topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][ReduceAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectAivAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    // aiv 模式不支持 PROD
    CHK_PRT_RET(
        op.reduceOp == ReduceOp::PROD,
        HCCL_WARNING(
            "[Algo][ReduceAutoSelector] ReduceOp[%s] is not supported yet for aiv mode.",
            op.reduceOp.Describe().c_str()),
        SelectorStatus::NOT_MATCH);

    if (op.dataType == DataType::UINT64 || op.dataType == DataType::FP64) {
        HCCL_WARNING("[Algo][ReduceAutoSelector] aiv mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    // aiv 直接走打平 mesh
    primQueueGenName = "AivReduceMesh1D";

    HCCL_INFO("[Algo][ReduceAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(OpType::REDUCE, 18, ReduceAutoSelector);
} // namespace Hccl

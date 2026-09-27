/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_auto_selector.h"
#include "selector_registry.h"
#include "coll_operator.h"

namespace Hccl {
constexpr u64 AR_M2M_1D_MAX_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_SMALL_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_MAX_DATA_SIZE = 16 * 1024 * 1024;
constexpr u64 AR_ONESHOT_1D_MAX_DATA_SIZE = 16 * 1024;

SelectorStatus AllReduceAutoSelector::SelectCcuMsAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);
    u32 rankSize_2P = 2;
    u32 rankSize_4P = 4;
    // MS 模式不支持 int8
    CHK_PRT_RET(
        op.dataType == DataType::INT8,
        HCCL_WARNING(
            "[Algo][AllReduceAutoSelector] dataType[%s] is not supported yet for ccu_ms mode.",
            op.dataType.Describe().c_str()),
        SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    CHK_PRT_RET(
        op.reduceOp == ReduceOp::PROD,
        HCCL_WARNING(
            "[Algo][AllReduceAutoSelector] ReduceOp[%s] is not supported yet for ccu_ms mode.",
            op.reduceOp.Describe().c_str()),
        SelectorStatus::NOT_MATCH);

    CHK_PRT_RET(
        op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64,
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64."),
        SelectorStatus::NOT_MATCH);

    HcclDetourType detourType = EnvConfig::GetInstance().GetDetourConfig().GetDetourType();
    CHK_PRT_RET(
        (detourType == HcclDetourType::HCCL_DETOUR_ENABLE_2P && rankSize_ != rankSize_2P)
            || (detourType == HcclDetourType::HCCL_DETOUR_ENABLE_4P && rankSize_ != rankSize_4P),
        HCCL_WARNING("[Algo][AllReduceAutoSelector] detourType not match for rankSize."), SelectorStatus::NOT_MATCH);

    CHK_PRT_RET(
        detourType == HcclDetourType::HCCL_DETOUR_ENABLE_2P_AND_4P,
        HCCL_WARNING("[Algo][AllReduceAutoSelector] HCCL_DETOUR_ENABLE_2P_AND_4P is not supported yet."),
        SelectorStatus::NOT_MATCH);

    if (topoInfo.levelNum > 1) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (IsInputOutputOverlap(op.inputMem, op.outputMem) == true) {
                // 不支持 inplace 场景
                return SelectorStatus::NOT_MATCH;
            }
            if (Is2DieFullMesh()) {
                HCCL_WARNING("[Algo][AllReduceAutoSelector] 2DieFullMesh is not supported yet for ccu_ms mode.");
                return SelectorStatus::NOT_MATCH;
            } else if (
                (detourType == HcclDetourType::HCCL_DETOUR_ENABLE_2P && rankSize_ == rankSize_2P)
                || (detourType == HcclDetourType::HCCL_DETOUR_ENABLE_4P && rankSize_ == rankSize_4P)) {
                primQueueGenName = "CcuAllReduceMeshDetour1D";
            } else if (dataSize_ / rankSize_ > AR_ONESHOT_1D_MAX_DATA_SIZE) {
                primQueueGenName = "CcuAllReduceMesh1D";
            } else {
                primQueueGenName = "CcuAllReduceMesh1DOneShot";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            if (IsSmallData(dataSize_) && IsInputOutputOverlap(op.inputMem, op.outputMem) != true) {
                primQueueGenName = "CcuAllReduceMesh2DOneShot";
            } else {
                primQueueGenName = "CcuAllReduceMesh2DTwoShot";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                if (IsInputOutputOverlap(op.inputMem, op.outputMem) == true) {
                    // 不支持 inplace 场景
                    return SelectorStatus::NOT_MATCH;
                }
                if ((detourType == HcclDetourType::HCCL_DETOUR_ENABLE_2P && rankSize_ == rankSize_2P)
                    || (detourType == HcclDetourType::HCCL_DETOUR_ENABLE_4P && rankSize_ == rankSize_4P)) {
                    primQueueGenName = "CcuAllReduceMeshDetour1D";
                } else if (dataSize_ / rankSize_ > AR_ONESHOT_1D_MAX_DATA_SIZE) {
                    primQueueGenName = "CcuAllReduceMesh1D";
                } else {
                    primQueueGenName = "CcuAllReduceMesh1DOneShot";
                }
            } else if (topoInfo.level0PcieMix) {
                HCCL_WARNING("[Algo][AllReduceAutoSelector] level0 PCIE mix is not supported yet for ccu_ms mode.");
                return SelectorStatus::NOT_MATCH;
            } else { // MS 不支持
                HCCL_WARNING(
                    "[Algo][AllReduceAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                    topoInfo.level0Shape);
                return SelectorStatus::NOT_MATCH;
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][AllReduceAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING(
                "[Algo][AllReduceAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][AllReduceAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);
    // ccu 模式不支持 PROD
    CHK_PRT_RET(
        op.reduceOp == ReduceOp::PROD,
        HCCL_WARNING(
            "[Algo][AllReduceAutoSelector] ReduceOp[%s] is not supported yet for ccu schedule mode.",
            op.reduceOp.Describe().c_str()),
        SelectorStatus::NOT_MATCH);

    CHK_PRT_RET(
        op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64,
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64."),
        SelectorStatus::NOT_MATCH);

    if (topoInfo.levelNum > 1) {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (topoInfo.netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                // 每框出 1 卡
                primQueueGenName = "CcuAllReduceNHR1D";
            } else if (Is2DieFullMesh()) {
                HCCL_WARNING("[Algo][AllReduceAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                CHK_PRT_RET(
                    op.dataType == DataType::INT8,
                    HCCL_WARNING(
                        "[Algo][AllReduceAutoSelector] dataType[%s] is not supported yet for ccu schedule "
                        "mode with ms reduce. levelNum[%u]",
                        op.dataType.Describe().c_str(), topoInfo.levelNum),
                    SelectorStatus::NOT_MATCH);
                if (IsSmallDataCCU(dataSize_, rankSize_)) {
                    primQueueGenName = "CcuAllReduceParallelMesh1DNHR"; // 64M以下跑ccu
                } else {
                    return SelectorStatus::NOT_MATCH; // 64M以上切为aicpu
                }
            }
        } else {
            HCCL_WARNING(
                "[Algo][AllReduceAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            CHK_PRT_RET(
                op.dataType == DataType::INT8,
                HCCL_WARNING(
                    "[Algo][AllReduceAutoSelector] dataType[%s] is not supported yet for ccu schedule mode "
                    "with ms reduce.",
                    op.dataType.Describe().c_str()),
                SelectorStatus::NOT_MATCH);
            double ratio; // 以8卡为基线确定ratio，用来表示不同卡数对下发的影响系数
            if (rankSize_ == 0) {
                HCCL_WARNING("[AllReduceAutoSelector] the selector is not set rankSize_.");
                ratio = 1;
            } else {
                ratio = DEFAULT_RANK_SIZE / rankSize_ / rankSize_;
            }
            if (Is2DieFullMesh()) {
                HCCL_WARNING("[Algo][AllReduceAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else if (dataSize_ * ratio > AR_M2M_1D_MAX_DATA_SIZE) {
                return SelectorStatus::NOT_MATCH;
            }
            primQueueGenName = "CcuAllReduceMeshMem2Mem1D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "CcuAllReduceMeshTwoShotMem2Mem2D";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            CHK_PRT_RET(
                op.dataType == DataType::INT8,
                HCCL_WARNING(
                    "[Algo][AllReduceAutoSelector] dataType[%s] is not supported yet for ccu schedule mode "
                    "with ms reduce.",
                    op.dataType.Describe().c_str()),
                SelectorStatus::NOT_MATCH);
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                double ratio; // 以8卡为基线确定ratio，用来表示不同卡数对下发的影响系数
                if (rankSize_ == 0) {
                    HCCL_WARNING("[AllReduceAutoSelector] the selector is not set rankSize_.");
                    ratio = 1;
                } else {
                    ratio = DEFAULT_RANK_SIZE / rankSize_ / rankSize_;
                }
                if (dataSize_ * ratio > AR_M2M_1D_MAX_DATA_SIZE) {
                    return SelectorStatus::NOT_MATCH;
                }
                primQueueGenName = "CcuAllReduceMeshMem2Mem1D";
            } else if (topoInfo.level0PcieMix) {
                HCCL_WARNING(
                    "[Algo][AllReduceAutoSelector] level0 PCIE mix is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                if (IsSmallDataCCU(dataSize_, rankSize_)) {
                    primQueueGenName = "CcuAllReduceParallelMesh1DNHR"; // 64M以下跑ccu
                } else {
                    return SelectorStatus::NOT_MATCH; // 64M以上切为aicpu
                }
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            HCCL_WARNING(
                "[Algo][AllReduceAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING(
                "[Algo][AllReduceAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo.level0Shape);
            return SelectorStatus::NOT_MATCH;
        }
    }

    HCCL_INFO("[Algo][AllReduceAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectAicpuAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    if (topoInfo.levelNum > 1) {
        if (op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64
            || op.reduceOp == ReduceOp::PROD) {
            HCCL_ERROR(
                "[SelectAicpuAlgo] INT64, UINT64, FP64 and reduceop::prod only support in-box fullmesh algo type now.");
            return SelectorStatus::NOT_MATCH;
        }
        if (topoInfo.Level1Nhr) {
            primQueueGenName = "InsAllReduceNHR";
        } else if (topoInfo.Level0Nhr) {
            primQueueGenName = "InsAllReduceParallelNHRNHR";
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            if (topoInfo.netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                primQueueGenName = "InsAllReduceNHR";
            } else {
                primQueueGenName = "InsAllReduceFourTemplateMesh1DNHR";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            primQueueGenName = "InsAllReduceParallelMesh2DNHR";
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            primQueueGenName = "InsAllReduceParallelNHRNHR";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo.level0Shape == Level0Shape::MESH_1D) {
            double ratio; // 以8卡为基线确定ratio，用来表示不同卡数对下发的影响系数
            if (rankSize_ == 0) {
                HCCL_WARNING("[AllReduceAutoSelector] the selector is not set rankSize_.");
                ratio = 1;
            } else {
                ratio = DEFAULT_RANK_SIZE / rankSize_ / rankSize_;
            }
            if (op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64
                || op.reduceOp == ReduceOp::PROD) {
                primQueueGenName = "InsAllReduceAicpuReduce";
            } else if (dataSize_ <= AR_AICPU_1D_SMALL_DATA_SIZE) {
                primQueueGenName = "InsAllReduceMesh1DOneShot";
            } else if (dataSize_ * ratio > AR_AICPU_1D_MAX_DATA_SIZE) {
                primQueueGenName = "InsAllReduceMesh1DTwoShotMeshChunk";
            } else {
                primQueueGenName = "InsAllReduceMesh1DTwoShot";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_2D) {
            if (op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64
                || op.reduceOp == ReduceOp::PROD) {
                primQueueGenName = "InsAllReduceAicpuReduceMesh2D";
            } else {
                primQueueGenName = "InsAllReduceMesh2DTwoShot";
            }
        } else if (topoInfo.level0Shape == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, TopoType::MESH_1D)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                double ratio; // 以8卡为基线确定ratio，用来表示不同卡数对下发的影响系数
                if (rankSize_ == 0) {
                    HCCL_WARNING("[AllReduceAutoSelector] the selector is not set rankSize_.");
                    ratio = 1;
                } else {
                    ratio = DEFAULT_RANK_SIZE / rankSize_ / rankSize_;
                }
                if (op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64
                    || op.reduceOp == ReduceOp::PROD) {
                    primQueueGenName = "InsAllReduceAicpuReduce";
                } else if (dataSize_ <= AR_AICPU_1D_SMALL_DATA_SIZE) {
                    primQueueGenName = "InsAllReduceMesh1DOneShot";
                } else if (dataSize_ * ratio > AR_AICPU_1D_MAX_DATA_SIZE) {
                    primQueueGenName = "InsAllReduceMesh1DTwoShotMeshChunk";
                } else {
                    primQueueGenName = "InsAllReduceMesh1DTwoShot";
                }
            } else {
                if (op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64
                    || op.reduceOp == ReduceOp::PROD) {
                    HCCL_WARNING(
                        "[Algo][AllReduceAutoSelector] INT64, UINT64, FP64, ReduceOp::PROD level0Shape[%d] is "
                        "not supported "
                        "yet for aicpu mode.",
                        topoInfo.level0Shape);
                    return SelectorStatus::NOT_MATCH;
                } else if (topoInfo.level0PcieMix) {
                    primQueueGenName = "InsAllReduceParallelMesh1DNHRPcie";
                } else {
                    primQueueGenName = "InsAllReduceParallelMesh1DNHR";
                }
            }
        } else if (topoInfo.level0Shape == Level0Shape::CLOS) {
            if (op.dataType == DataType::INT64 || op.dataType == DataType::UINT64 || op.dataType == DataType::FP64
                || op.reduceOp == ReduceOp::PROD) {
                primQueueGenName = "InsAllReduceAicpuReduce";
            } else {
                primQueueGenName = "InsAllReduceNHR";
            }
        } else {
            HCCL_WARNING("[AllReduceAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[Algo][AllReduceAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectAivAlgo(
    const TopoInfo& topoInfo, const CollAlgOperator& op,
    const std::map<OpType, std::vector<HcclAlgoType>>& configAlgMap, std::string& primQueueGenName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo.levelNum);

    // aiv 模式不支持 PROD
    CHK_PRT_RET(
        op.reduceOp == ReduceOp::PROD,
        HCCL_WARNING(
            "[Algo][AllReduceAutoSelector] ReduceOp[%s] is not supported yet for aiv mode.",
            op.reduceOp.Describe().c_str()),
        SelectorStatus::NOT_MATCH);

    if (op.dataType == DataType::UINT64 || op.dataType == DataType::FP64) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] aiv mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    // aiv 直接走打平 mesh
    if (dataSize_ <= AIV_ALL_REDUCE_SMALL_COUNT) {
        primQueueGenName = "AivAllReduceMesh1DOneShot";
    } else {
        primQueueGenName = "AivAllReduceMesh1DTwoShot";
    }

    HCCL_INFO("[Algo][AllReduceAutoSelector][%s] Algo match [%s]", __func__, primQueueGenName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(OpType::ALLREDUCE, 18, AllReduceAutoSelector);
} // namespace Hccl
